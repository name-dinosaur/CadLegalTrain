from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load fine tuned model
model_path = "./gpt2-legal-final"

print("Loading model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# gpt prompt
prompt = """Is punching someone a criminal offence under Canadian law?
Answer:
"""

#  Tokenize and generate 
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=250,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGPT-2 Output:\n")
print(generated_text.split("Answer:")[-1].strip())
