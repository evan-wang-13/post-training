import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft.peft_model import PeftModel
from peft.config import PeftConfig
from peft.tuners.lora import LoraConfig
from trl import DPOTrainer, DPOConfig
import wandb
from typing import Dict, Any


def setup_wandb(project_name: str = "qwen-dpo", use_wandb: bool = True) -> None:
    """Initialize Weights & Biases tracking."""
    if use_wandb:
        wandb.init(project=project_name)


def get_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2-0.5B-Instruct",
    sft_model_path: str = "./qwen_sft_final",
    device_map: str = "auto",
) -> tuple:
    """
    Load the SFT model and tokenizer with 4-bit quantization for DPO training.
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

    # Load the SFT model as the reference model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quantization_config, device_map=device_map
    )
    # Freeze the base model's parameters
    model.config.use_cache = False  # This is important for training
    for param in model.parameters():
        param.requires_grad = False  # Make sure base model is frozen

    # Load the SFT weights and prepare for training
    model = PeftModel.from_pretrained(
        model,
        sft_model_path,
        is_trainable=True,  # Make sure PEFT layers are trainable
    )

    # Double check that LoRA parameters are trainable
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    return model, tokenizer


def get_training_args(use_wandb: bool = True) -> DPOConfig:
    """
    Return training arguments for DPO.
    """
    return DPOConfig(
        output_dir="./qwen_dpo_output",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        max_steps=-1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=True,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=100,
        beta=0.1,
        max_length=512,
        max_prompt_length=128,
    )


def prepare_dataset():
    """
    Load and prepare the Ultrafeedback dataset for DPO.
    """
    train_dataset = load_dataset(
        "trl-lib/ultrafeedback_binarized", split="train[:1000]"
    )
    eval_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="test[:1000]")
    return train_dataset, eval_dataset


def main(use_wandb: bool = True):
    # Initialize wandb
    setup_wandb(use_wandb=use_wandb)

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer()

    # Get training arguments
    training_args = get_training_args(use_wandb=use_wandb)

    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset()

    # Initialize DPO trainer
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # Train the model
    trainer.train()

    # Save the final model
    trainer.save_model("../models/qwen_dpo_final")

    # End wandb run
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main(use_wandb=True)
