from __future__ import annotations
import numpy as np

def pairwise_dist(coords: np.ndarray) -> np.ndarray:
    X = np.asarray(coords, dtype=np.float32)
    q = (X*X).sum(axis=1, keepdims=True)
    D2 = q + q.T - 2.0*(X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    np.sqrt(D2, out=D2)
    return D2.astype(np.float32)

def complete_edges(n: int) -> np.ndarray:
    i, j = np.triu_indices(n, k=1)
    return np.stack([i, j], axis=1).astype(np.int64)


def knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    D = pairwise_dist(coords)
    np.fill_diagonal(D, np.inf)
    k_eff = max(1, min(k, n-1))
    nbrs = np.argpartition(D, kth=k_eff, axis=1)[:, :k_eff]
    edges=set()
    for i in range(n):
        for j in nbrs[i]:
            a,b = (i,int(j))
            if a>b: a,b=b,a
            edges.add((a,b))
    return np.asarray(sorted(edges), dtype=np.int64)

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



def union_edges_unique(*arrays) -> np.ndarray:
    """
    Take one or more [m,2] edge arrays (undirected), return a unique-sorted
    [M,2] int64 array with (min(u,v), max(u,v)) ordering.
    """
    S = set()
    for arr in arrays:
        if arr is None: 
            continue
        A = np.asarray(arr, dtype=np.int64)
        if A.ndim != 2 or A.shape[1] != 2:
            continue
        for a, b in A:
            a = int(a); b = int(b)
            if a > b: a, b = b, a
            S.add((a, b))
    if not S:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(sorted(S), dtype=np.int64)
