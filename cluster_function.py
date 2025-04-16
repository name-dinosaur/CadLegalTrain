import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# Load Saved Embeddings
print("Loading saved embeddings...")
embeddings_array = np.load("legal_text_embeddings_80.npy")  # Matches 80% training set
print(f"Embeddings loaded! Shape: {embeddings_array.shape}")

# Step 2: Load Matching Text Data
print("Loading training text data...")
df = pd.read_parquet("legal_train_80.parquet")  # Must match the embeddings shape
print(f" Loaded text data with {len(df)} samples.")

# Perform Clustering
num_clusters = 10000  # Adjust as needed
print(f"Clustering {len(df)} samples into {num_clusters} clusters...")

kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(embeddings_array)

# Find the most representative sample per cluster
closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, embeddings_array)
df_reduced = df.iloc[closest_indices].reset_index(drop=True)

print(f"Reduced dataset from {len(df)} to {len(df_reduced)} samples.")

# Save Clustered Representative Samples
df_reduced.to_csv("clustered_legal_data.csv", index=False)
print("Clustered dataset saved as 'clustered_legal_data.csv'. Ready for tokenization! 🚀")
