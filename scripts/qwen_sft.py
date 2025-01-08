import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft.tuners.lora import LoraConfig
from trl import SFTTrainer
import wandb
from typing import Dict, Any


def setup_wandb(project_name: str = "qwen-sft") -> None:
    """Initialize Weights & Biases tracking."""
    wandb.init(project=project_name)


def get_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2-0.5B-Instruct", device_map: str = "auto"
) -> tuple:
    """
    Load the model and tokenizer with 4-bit quantization and LORA config.

    Args:
        model_id: HuggingFace model identifier
        device_map: Device mapping strategy
    Returns:
        tuple: (model, tokenizer)
    """
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config, device_map=device_map
    )

    return model, tokenizer


def get_training_args() -> TrainingArguments:
    """
    Return training arguments using the HuggingFace TrainingArguments class.
    Adjust these based on your GPU memory constraints.
    """
    return TrainingArguments(
        output_dir="./qwen_sft_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_steps=-1,  # No limit on steps
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="wandb",  # Enable wandb logging
        remove_unused_columns=True,
        gradient_checkpointing=True,  # Added gradient checkpointing
        fp16=True,
    )


def prepare_dataset():
    """
    Load and prepare the dataset.
    Returns:
        dataset: Prepared dataset for training
    """
    dataset = load_dataset("trl-lib/Capybara", split="train[:1000]")
    return dataset


def main():
    # Initialize wandb
    setup_wandb()

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # Get training arguments
    training_args = get_training_args()

    # Prepare dataset
    dataset = prepare_dataset()

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        # formatting_func=lambda x: x["input"] + "\n" + x["output"],
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        args=training_args,  # Use training_args instead of unpacking config
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model("../models/qwen_sft_final")

    # End wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
