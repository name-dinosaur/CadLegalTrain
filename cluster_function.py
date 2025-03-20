import pandas as pd
import numpy as np
from datasets import load_from_disk
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

### ✅ Step 1: Load Saved Embeddings ###
print("Loading saved embeddings...")
embeddings_array = np.load("legal_text_embeddings.npy")  # Ensure this file exists!

print(f"✅ Embeddings loaded! Shape: {embeddings_array.shape}")

### ✅ Step 2: Load Dataset Directly Instead of CSV ###
print("Loading original dataset from Hugging Face disk storage...")
dataset = load_from_disk("canadian_legal_data")["train"]

# Convert to Pandas and keep only text column
df = pd.DataFrame(dataset)
df = df[["unofficial_text"]].dropna()

print(f"✅ Loaded dataset with {len(df)} samples.")

### ✅ Step 3: Perform Clustering ###
num_clusters = 3500  # Adjust based on dataset size
print(f"Clustering {len(df)} samples into {num_clusters} clusters...")

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings_array)

# Keep the most representative case per cluster
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_array)
df_reduced = df.iloc[closest_indices].reset_index(drop=True)

print(f"✅ Reduced dataset from {len(df)} to {len(df_reduced)} samples.")

### ✅ Step 4: Save Clustered Dataset ###
df_reduced.to_csv("clustered_legal_data.csv", index=False)
print("✅ Clustered dataset saved as 'clustered_legal_data.csv'. Ready for tokenization! 🚀")
