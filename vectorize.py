import pandas as pd
import numpy as np
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from transformers import AutoTokenizer

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
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding text into vectors...")
df["embeddings"] = df["unofficial_text"].astype(str).apply(lambda x: model.encode(x))

# Convert to NumPy array
embeddings_array = np.vstack(df["embeddings"].values)

print(f"Generated embeddings shape: {embeddings_array.shape}")

### ✅ Step 3: Clustering (Reduce Dataset Size) ###
num_clusters = 5000  # Adjust to control dataset size
print(f"Clustering {embeddings_array.shape[0]} samples into {num_clusters} clusters...")

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings_array)

# Keep the most representative case per cluster
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_array)
df_reduced = df.iloc[closest_indices]

print(f"Reduced dataset from {len(df)} to {len(df_reduced)} samples")

### ✅ Step 4: Tokenization for LLaMA 3 ###
print("Loading LLaMA tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")

# Tokenize dataset
df_reduced["tokens"] = df_reduced["unofficial_text"].apply(lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=512)["input_ids"])

### ✅ Step 5: Save Final Dataset for LLaMA Training ###
df_reduced.to_json("llama3_legal_dataset.json", orient="records")

print("Dataset is fully processed and saved as 'llama3_legal_dataset.json'. Ready for LLaMA 3 training! 🚀")
