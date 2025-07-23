import yaml
import numpy as np
import os
from sklearn.cluster import KMeans

# Parameter configuration
K = 12  # Number of clusters
RANDOM_STATE = 42

# Load data
with open("train_set.yaml", "r") as f:
    train_set = yaml.safe_load(f)

scale2str = {0.06:"006", 0.08:"008", 0.10:"010", 0.12:"012", 0.15:"015"}

# Store features and corresponding metadata
pc_feats = []
metadata = []  # Each element is a tuple (key, scale)

for key in train_set.keys():
    for scale in train_set[key]:
        feat_path = os.path.join(
            "assets/meshdatav3_pc_feat", 
            key, 
            f"pc_feat_{scale2str[scale]}.npy"
        )
        pc_feats.append(np.load(feat_path))
        metadata.append((key, scale))

# Convert to feature matrix [N x D]
X = np.vstack(pc_feats)

# Perform K-means clustering
kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(X)

# Find the closest sample to each cluster center
center_features = kmeans.cluster_centers_
cluster_results = []

for cluster_id in range(K):
    # Compute distances from the center to all samples
    distances = np.linalg.norm(X - center_features[cluster_id], axis=1)
    
    # Sort by distance to get sample indices
    sorted_indices = np.argsort(distances)
    
    # Closest sample
    nearest_idx = sorted_indices[0]
    nearest_key, nearest_scale = metadata[nearest_idx]
    
    # Top 10 samples closest to the center
    top10_indices = sorted_indices[:10]
    top10_samples = [(metadata[i][0], metadata[i][1]) for i in top10_indices]
    
    # All sample indices belonging to this cluster
    sample_indices = np.where(kmeans.labels_ == cluster_id)[0]
    
    cluster_results.append({
        "cluster_id": cluster_id,
        "nearest_sample_key": nearest_key,
        "nearest_sample_scale": nearest_scale,
        "top10_samples": top10_samples,
        "sample_indices": sample_indices
    })

# Print the results
for res in cluster_results:
    print(f"Cluster {res['cluster_id']}:")
    print(f"  Nearest Sample -> Key: {res['nearest_sample_key']}, Scale: {res['nearest_sample_scale']:.3f}")
    print(f"  Num Samples in Cluster: {len(res['sample_indices'])}")
    print("  Top 10 closest samples:")
    for idx, (key, scale) in enumerate(res['top10_samples'], start=1):
        print(f"    {idx}. Key: {key}, Scale: {scale:.3f}")
    print()
