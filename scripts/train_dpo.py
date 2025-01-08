# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft.mapping import get_peft_model
from peft.tuners.lora import LoraConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",  # Normal Float 4 format
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

# Update model loading
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

for param in model.parameters():
    if param.dtype in [torch.float16, torch.float32, torch.float64]:
        param.requires_grad = True
# Add this line to enable gradients
model.config.use_cache = False  # Required for gradient checkpointing
model.train()  # Set to training mode

# Add LoRA configuration
peft_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=32,  # Alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],  # Typical attention modules
)

# Apply LoRA to the model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train[:1000]")

training_args = DPOConfig(
    output_dir="Qwen2-0.5B-DPO",
    logging_steps=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=100,
    gradient_checkpointing=True,
)
trainer = DPOTrainer(
    model=model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
)
trainer.train()
