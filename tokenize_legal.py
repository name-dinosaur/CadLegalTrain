import pandas as pd
from transformers import AutoTokenizer

### ✅ Step 1: Load the Clustered Dataset ###
print("Loading clustered dataset...")
df_reduced = pd.read_csv("clustered_legal_data.csv")

print(f"Dataset contains {len(df_reduced)} legal cases.")

### ✅ Step 2: Load an Open-Source Tokenizer (LLaMA 1) ###
print("Loading open-source tokenizer (LLaMA 1)...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ✅ Manually set the padding token (LLaMA models don’t have one by default)
tokenizer.pad_token = tokenizer.eos_token

### ✅ Step 3: Tokenization ###
print("Tokenizing dataset...")
df_reduced["tokens"] = df_reduced["unofficial_text"].apply(
    lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=512)["input_ids"]
)

### ✅ Step 4: Save the Tokenized Dataset ###
df_reduced.to_json("tokenized_legal_dataset.json", orient="records")

print("✅ Tokenization complete! Ready for GPT 2 fine-tuning. 🚀")
