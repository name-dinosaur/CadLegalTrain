import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

# Load Tokenized Dataset  
print("Loading tokenized dataset...")
dataset = load_dataset("json", data_files="tokenized_legal_dataset_gpt2.json", split="train")

# Load GPT-2 tokenizer and model 
print("Loading GPT-2 model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))
model.eval()  # Safer default unless training

# Format dataset for GPT-2
print("Formatting dataset...")

def format_sample(example):
    return {
        "input_ids": example["tokens"],
        "attention_mask": [1] * len(example["tokens"])
    }

# Use a small subset first (1000 examples)
dataset = dataset.shuffle(seed=42).select(range(1000))
dataset = dataset.map(format_sample)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training configuration
print("Setting training arguments...")
training_args = TrainingArguments(
    output_dir="./gpt2-legal-checkpoint",
    per_device_train_batch_size=1,          
    gradient_accumulation_steps=2,          
    num_train_epochs=10,                    
    logging_steps=50,
    save_steps=200,
    save_total_limit=1,
    fp16=True,                              
    evaluation_strategy="no",
    report_to="none"
)

print("Starting training (1 epoch, 1000 samples)...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# Save final model
print("✅ Training complete. Saving model...")
trainer.save_model("./gpt2-legal-final")
tokenizer.save_pretrained("./gpt2-legal-final")

print("All done! Your model is safely fine-tuned and saved.")
