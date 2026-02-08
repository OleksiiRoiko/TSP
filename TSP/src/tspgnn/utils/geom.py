from __future__ import annotations
import numpy as np

def complete_edges(n: int) -> np.ndarray:
    i, j = np.triu_indices(n, k=1)
    return np.stack([i, j], axis=1).astype(np.int64)


def edge_features(coords: np.ndarray, edges: np.ndarray, feature_dim: int | None = None) -> np.ndarray:
    """
    Vectorized: build baseline 10-D edge features, then truncate/zero-pad
    to `feature_dim` if provided.

    Baseline: [xa, ya, xb, yb, d, dx, dy, angle, dx^2, dy^2]
    """
    C = np.asarray(coords, dtype=np.float32)
    E = np.asarray(edges, dtype=np.int64)
    a = E[:, 0]
    b = E[:, 1]

    A = C[a]                    # [m,2]
    B = C[b]                    # [m,2]
    D = B - A                   # [m,2] (dx, dy)
    d = np.linalg.norm(D, axis=1, keepdims=True)             # [m,1]
    ang = np.arctan2(D[:, 1:2], D[:, 0:1])                   # [m,1]
    dx2dy2 = D * D                                            # [m,2]

    X = np.concatenate([A, B, d, D, ang, dx2dy2], axis=1).astype(np.float32)  # [m,10]

    if feature_dim is None or feature_dim == X.shape[1]:
        return X
    if feature_dim < X.shape[1]:
        return X[:, :feature_dim].copy()
    pad = np.zeros((X.shape[0], feature_dim - X.shape[1]), dtype=np.float32)
    return np.hstack([X, pad])
