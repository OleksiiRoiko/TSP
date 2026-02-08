from __future__ import annotations
from pathlib import Path
import hashlib, json
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.io import load_npz
from ..utils.geom import complete_edges, edge_features
from ..utils.tour import tour_edges_undirected


def _hash_key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()[:16]


class NPZTSPDataset(Dataset):
    """
    Loads *.npz TSP graphs recursively.
    Builds complete-graph edge features (vectorized) and labels.
    Optional on-disk caching of features/labels to avoid recomputation.
    """
    def __init__(
        self,
        root: str,
        feature_dim: int = 10,
        cache_dir: str | None = "runs/cache/features",  # None disables cache
    ):
        self.root = Path(root)
        self.files = sorted(p for p in self.root.rglob("*.npz") if p.is_file())
        self.feature_dim = int(feature_dim)

        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is not None:
            (self.cache_dir / str(self.feature_dim) / "complete").mkdir(parents=True, exist_ok=True)

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npz files found under '{self.root}'.")

    def __len__(self) -> int:
        return len(self.files)

    # --------- caching helpers ----------
    def _cache_paths_for(self, npz_path: Path) -> tuple[Path, Path]:
        # Make a stable key from path + basic params
        if self.cache_dir is None:
            raise RuntimeError("cache_dir is None; caching is disabled")
        key = _hash_key(str(npz_path.resolve()), f"fd={self.feature_dim}", "cand=complete")
        base = (self.cache_dir / str(self.feature_dim) / "complete") / key
        return base.with_suffix(".npz"), base.with_suffix(".json")

    def _try_load_cache(self, npz_path: Path):
        if self.cache_dir is None:
            return None
        fx, meta = self._cache_paths_for(npz_path)
        if fx.exists() and meta.exists():
            try:
                z = np.load(fx, allow_pickle=False)
                X = z["X"]
                y = z["y"] if "y" in z.files else None
                return X, y
            except Exception:
                return None
        return None

    def _save_cache(self, npz_path: Path, X: np.ndarray, y: np.ndarray | None):
        if self.cache_dir is None:
            return
        fx, meta = self._cache_paths_for(npz_path)
        fx.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fx, X=X.astype(np.float32), y=y if y is not None else np.array([]))
        meta.write_text(json.dumps({"source": str(npz_path)}), encoding="utf-8")

    # --------- main API ----------
    def __getitem__(self, idx: int):
        f = self.files[idx]

        # 1) try cache
        cached = self._try_load_cache(f)
        if cached is not None:
            X, y = cached
            return {
                "edge_feats": torch.from_numpy(X).float(),
                "labels": None if (y is None or y.size == 0) else torch.from_numpy(y.astype(np.float32)).float(),
            }

        # 2) compute fresh
        d = load_npz(f)
        C = d["coords"].astype(np.float32)
        T = d.get("label_tour", None)
        T = T.astype(np.int64) if T is not None else None
        n = C.shape[0]

        # Always use the complete graph
        E = complete_edges(n)

        X = edge_features(C, E, feature_dim=self.feature_dim)

        y = None
        if T is not None:
            gt = set((min(int(a), int(b)), max(int(a), int(b))) for a, b in tour_edges_undirected(T))
            y = np.array([1.0 if (min(a, b), max(a, b)) in gt else 0.0 for a, b in E], dtype=np.float32)

        # 3) cache for next epoch
        self._save_cache(f, X, y)

        return {
            "edge_feats": torch.from_numpy(X).float(),
            "labels": None if y is None else torch.from_numpy(y).float(),
        }


def collate_edge_batches(batch):
    feats = torch.cat([b["edge_feats"] for b in batch], dim=0)
    labels = None
    if batch[0]["labels"] is not None:
        labels = torch.cat([b["labels"] for b in batch], dim=0)
    return {"edge_feats": feats, "labels": labels}
