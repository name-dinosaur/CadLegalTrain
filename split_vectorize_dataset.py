import pandas as pd
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import torch

# Load Dataset
print("Loading dataset...")
dataset = load_from_disk("canadian_legal_data")["train"]

# Convert to DataFrame and drop rows without text
df = pd.DataFrame(dataset)
df = df[["unofficial_text"]].dropna()

print(f"Total rows: {df.shape[0]}")
print(f"Dataset Size in RAM: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# Split 80/20
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
split_idx = int(0.8 * len(df))
train_df = df[:split_idx]
test_df = df[split_idx:]

# Save both splits to disk
train_df.to_parquet("legal_train_80.parquet", index=False)
print(f"Saved 80% training set with {len(train_df)} rows to 'legal_train_80.parquet'")

test_df.to_parquet("legal_test_20.parquet", index=False)
print(f"Saved 20% test set with {len(test_df)} rows to 'legal_test_20.parquet'")

# Load Model
print("Loading embedding model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Vectorize Training Texts
print("Encoding text into vectors...")
texts = train_df["unofficial_text"].astype(str).tolist()
batch_size = 32
embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

# Save as NumPy array
embeddings_array = np.array(embeddings)
np.save("legal_text_embeddings_80.npy", embeddings_array)

# store in DataFrame
train_df["embeddings"] = list(embeddings)
print(f"Saved embeddings with shape: {embeddings_array.shape}")
