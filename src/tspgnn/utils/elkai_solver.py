from __future__ import annotations

import importlib

import numpy as np


def solve_with_elkai(coords: np.ndarray, scale: int = 10_000) -> np.ndarray | None:
    try:
        elkai = importlib.import_module("elkai")
    except Exception:
        return None
    try:
        dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
        mat = (dist * int(scale)).astype(int)
        return np.array(elkai.solve_int_matrix(mat.tolist()), dtype=np.int64)
    except Exception:
        return None
