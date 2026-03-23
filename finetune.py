import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from dotenv import load_dotenv

load_dotenv()

# ---- Config ----
MODEL_ID   = "microsoft/Phi-3.5-mini-instruct"
OUTPUT_DIR = Path("models/cre-llm-v1")
PAIRS_PATH = Path("data/training/pairs.jsonl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Device ----
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ---- Load training pairs ----
def load_pairs():
    pairs = []
    with open(PAIRS_PATH) as f:
        for line in f:
            pair = json.loads(line.strip())
            text = f"""<|system|>
You are an expert in Canadian commercial real estate lending and underwriting.<|end|>
<|user|>
{pair['prompt']}<|end|>
<|assistant|>
{pair['completion']}<|end|>"""
            pairs.append({"text": text})
    print(f"Loaded {len(pairs):,} training pairs")
    return Dataset.from_list(pairs)

# ---- LoRA Config ----
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,               # reduced from 16 - less memory
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

# ---- Main ----
def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model directly to MPS...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # float16 cuts memory in half vs float32
        device_map={"": device},    # load directly to MPS, skip CPU
    )

    # apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # load data
    dataset = load_pairs()

    # ---- SFT Config ----
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=2,   # increased from 1
        gradient_accumulation_steps=2,   # reduced from 4
        learning_rate=2e-4,
        fp16=True,                       # enable fp16 for speed on MPS
        bf16=False,
        logging_steps=5,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=256,                  # reduced from 512
        dataloader_pin_memory=False,     # disable pin_memory warning
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting fine-tuning...")
    print(f"Training pairs: {len(dataset):,}")
    print(f"Epochs: {sft_config.num_train_epochs}")
    print(f"Device: {device}")
    print()

    trainer.train()

    print("\nSaving LoRA adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()