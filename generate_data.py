import numpy as np
from sklearn import datasets


def generate_data(no_clusters: int, cluster_std: int, n_data: int = 100) -> np.ndarray:
    return datasets.make_blobs(
        n_samples=n_data,
        n_features=2,
        centers=no_clusters,
        cluster_std=cluster_std,
        center_box=(-20, 20),
        random_state=42,
    )
