import pandas as pd
from transformers import GPT2Tokenizer

#Load the Clustered Dataset
print("Loading clustered dataset...")
df_reduced = pd.read_csv("clustered_legal_data.csv")

print(f"Dataset contains {len(df_reduced)} legal cases.")

# Load GPT-2 Tokenizer
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

#set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token

# Tokenization
print("Tokenizing dataset...")
df_reduced["tokens"] = df_reduced["unofficial_text"].apply(
    lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=512)["input_ids"]
)

#Save the Tokenized Dataset
df_reduced.to_json("tokenized_legal_dataset_gpt2.json", orient="records")

print("Tokenization complete")
