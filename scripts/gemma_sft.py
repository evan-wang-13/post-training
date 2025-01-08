import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft.tuners.lora import LoraConfig
from trl import SFTTrainer
import wandb
from typing import Dict, Any


def setup_wandb(project_name: str = "gemma-sft") -> None:
    """Initialize Weights & Biases tracking."""
    wandb.init(project=project_name)


def get_model_and_tokenizer(
    model_id: str = "google/gemma-2-9b-it", device_map: str = "auto"
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

    # Configure LoRA
    # lora_config = LoraConfig(
    #     r=16,  # Rank
    #     lora_alpha=32,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config, device_map=device_map
    )

    return model, tokenizer


def get_training_config() -> Dict[str, Any]:
    """
    Return training configuration parameters.
    Adjust these based on your GPU memory constraints.
    """
    return {
        "output_dir": "./gemma_sft_output",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "max_seq_length": 512,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
    }


def prepare_dataset():
    """
    Load and prepare the Capybara dataset.
    Returns:
        dataset: Prepared dataset for training
    """
    dataset = load_dataset("HuggingFaceH4/capybara_train", split="train")

    # Filter out examples that are too long to avoid OOM
    dataset = dataset.filter(lambda x: len(x["input"]) + len(x["output"]) < 1000)

    # Take a small subset for testing
    dataset = dataset.select(range(min(len(dataset), 1000)))

    return dataset


def main():
    # Initialize wandb
    setup_wandb()

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # Get training configuration
    training_config = get_training_config()

    # Prepare dataset
    dataset = prepare_dataset()

    # Initialize SFT trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=lambda x: x["input"] + "\n" + x["output"],
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        **training_config,
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model("../models/gemma_sft_final")

    # End wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
