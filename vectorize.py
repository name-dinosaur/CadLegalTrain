import pandas as pd
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

### ✅ Step 1: Load Dataset ###
print("Loading dataset...")
dataset = load_from_disk("canadian_legal_data")["train"]

# Convert to Pandas
df = pd.DataFrame(dataset)
df = df[["unofficial_text"]].dropna()  # Keep only text column

print(f"Total rows: {df.shape[0]}")
print(f"Dataset Size in RAM: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

### ✅ Step 2: Vectorize Using Sentence Embeddings ###
print("Loading embedding model...")
device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically use GPU if available
print(f"Using device: {device}")

model = SentenceTransformer("all-MiniLM-L6-v2", device=device)  # Load model onto GPU
print("Encoding text into vectors...")

# Convert all text into a list
texts = df["unofficial_text"].astype(str).tolist()

# Encode in batches
batch_size = 32  # Adjust based on available memory
embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

# Convert to NumPy array and save
embeddings_array = np.array(embeddings)
np.save("legal_text_embeddings.npy", embeddings_array)

# Store embeddings in DataFrame
df["embeddings"] = list(embeddings)

print(f"Finished encoding! Saved embeddings with shape: {embeddings_array.shape}")

