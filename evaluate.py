import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_ID   = "microsoft/Phi-3.5-mini-instruct"
ADAPTER    = "models/cre-llm-v1"
device     = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": device},
)

print("Loading CRE adapter...")
model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()

def ask(question):
    prompt = f"""<|system|>
You are an expert in Canadian commercial real estate lending and underwriting.<|end|>
<|user|>
{question}<|end|>
<|assistant|>"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # extract just the assistant response
    response = response.split("<|assistant|>")[-1].strip()
    return response

# ---- Test questions ----
questions = [
    "What is a typical cap rate for Class A multifamily in Toronto?",
    "What does DSCR mean and what is the minimum threshold for Canadian commercial lending?",
    "What is the difference between a PAC and a CAP in commercial mortgage lending?",
    "What are typical GoC spread pricing conventions for Canadian commercial mortgages?",
]

print("\n" + "="*60)
print("CRE-LLM v1 Evaluation")
print("="*60)

for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")
    print("-"*60)