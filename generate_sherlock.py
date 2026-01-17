"""
Generate Sherlock Holmes-style text using the fine-tuned model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model(base_model_name="google/gemma-3-27b-it", peft_model_path="sherlock-gemma-model"):
    """Load the fine-tuned model."""
    print("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load fine-tuned LoRA weights
    model = PeftModel.from_pretrained(base_model, peft_model_path)
    model.eval()
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.8, top_p=0.9):
    """Generate text continuation from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    # Load the fine-tuned model
    model, tokenizer = load_model()
    
    print("\n" + "="*80)
    print("Sherlock Holmes Text Generator")
    print("="*80)
    print("\nEnter a prompt to start a Sherlock Holmes story.")
    print("Type 'quit' to exit.\n")
    
    while True:
        prompt = input("\nPrompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        if not prompt:
            print("Please enter a prompt.")
            continue
        
        print("\nGenerating...\n")
        generated = generate_text(model, tokenizer, prompt, max_length=400)
        
        print("-" * 80)
        print(generated)
        print("-" * 80)


if __name__ == "__main__":
    # Example prompts to try:
    example_prompts = [
        "To Sherlock Holmes, the case was",
        "I had not seen my friend Holmes for",
        "The mystery of the",
        "Watson, come quickly! I have discovered",
    ]
    
    print("\nExample prompts you can try:")
    for i, prompt in enumerate(example_prompts, 1):
        print(f"{i}. {prompt}")
    
    main()
