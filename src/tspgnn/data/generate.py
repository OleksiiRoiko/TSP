from __future__ import annotations
from pathlib import Path
import importlib
import numpy as np
from ..config import GenerateCfg
from ..utils.io import save_npz
from ..utils.tour import tour_length, two_opt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from tqdm import tqdm

def _make_one(args):
    idx, n, seed, dist, use_elkai = args
    rng = np.random.default_rng(seed + idx * 1337)
    coords = _sample(rng, n, dist)
    tour = _elkai_tour(coords) if use_elkai else None
    if tour is None:
        tour = _nn2opt(coords, max_passes=50)
    L = tour_length(coords, tour)
    return idx, coords.astype(np.float32), tour.astype(np.int32), float(L), dist

def _elkai_tour(coords: np.ndarray):
    try:
        elkai = importlib.import_module("elkai")
    except Exception:
        return None
    try:
        dist = np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
        mat = (dist*10_000).astype(int)
        return np.array(elkai.solve_int_matrix(mat.tolist()), dtype=np.int64)
    except Exception:
        return None

def _nn2opt(coords: np.ndarray, max_passes=50):
    n=coords.shape[0]; unvis=set(range(n)); cur=0; tour=[cur]; unvis.remove(cur)
    while unvis:
        rest=np.array(sorted(list(unvis)))
        d=np.linalg.norm(coords[rest]-coords[cur], axis=1)
        nxt=int(rest[np.argmin(d)]); tour.append(nxt); unvis.remove(nxt); cur=nxt
    return two_opt(coords, np.asarray(tour, dtype=np.int64), max_passes=max_passes)

def _sample(rng: np.random.Generator, n: int, kind: str):
    if kind=="uniform": return rng.random((n,2), dtype=np.float32)
    if kind=="clustered":
        centers=rng.random((4,2)); assign=rng.integers(0,4,size=n)
        pts=centers[assign] + 0.08*rng.standard_normal((n,2))
        return np.clip(pts,0,1).astype(np.float32)
    if kind=="ring":
        # Points around a ring with small radial noise
        angles = rng.random(size=n) * (2.0 * np.pi)
        r = 0.35 + 0.03 * rng.standard_normal(size=n)
        x = 0.5 + r * np.cos(angles)
        y = 0.5 + r * np.sin(angles)
        pts = np.stack([x, y], axis=1)
        return np.clip(pts, 0.0, 1.0).astype(np.float32)
    if kind=="grid_jitter":
        g=int(np.sqrt(n))
        xs,ys=np.meshgrid(np.linspace(0.05,0.95,g), np.linspace(0.05,0.95,g))
        pts=np.stack([xs.ravel(),ys.ravel()],axis=1)
        if pts.shape[0]>n: pts=pts[:n]
        if pts.shape[0]<n: pts=np.vstack([pts, rng.random((n-pts.shape[0],2))])
        pts+=0.02*rng.standard_normal(pts.shape)
        return np.clip(pts,0,1).astype(np.float32)
    raise ValueError(kind)

def run(cfg: GenerateCfg, logger):
    rng = np.random.default_rng(int(cfg.seed))
    root = Path(cfg.out_root)
    dists = list(cfg.dist_names)
    probs = np.array(cfg.dist_probs, dtype=np.float64)
    if len(dists) != len(probs):
        raise ValueError("dist_names and dist_probs must have the same length")
    if probs.sum() <= 0:
        raise ValueError("dist_probs must sum to a positive value")
    probs /= probs.sum()

    # sensible default for Windows too
    workers = max(1, min(4, (os.cpu_count() or 1) // 2))

    for n in cfg.n_list:
        for split, count in {
            "train": cfg.per_size_train,
            "val":   cfg.per_size_val,
            "test":  cfg.per_size_test,
        }.items():
            out_dir = root / split / f"N{n}"
            out_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[gen] N={n} split={split} count={count} workers={workers}")

            # build task list once (we keep a stable idx for file names)
            tasks = []
            start_idx = 0  # could be len(existing) if you append later
            for i in range(int(count)):
                dist = str(rng.choice(dists, p=probs))
                use_elkai = bool(rng.random() < cfg.elkai_frac)
                tasks.append((start_idx + i, n, int(cfg.seed), dist, use_elkai))

            # run with a nice progress bar
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_make_one, t) for t in tasks]
                with tqdm(total=len(futures), ncols=100,
                          desc=f"N={n} | {split}", leave=True) as bar:
                    for fut in as_completed(futures):
                        idx, coords, tour, L, dist = fut.result()
                        # deterministic filename from idx (no glob scans)
                        save_npz(
                            out_dir / f"syn_{idx:06d}.npz",
                            id=np.array(f"syn_{idx:06d}"),
                            n=np.int32(n),
                            coords=coords,
                            label_tour=tour,
                            label_len_norm=L,
                            distribution=np.array(dist),
                            metric=np.array("EUC_2D"),
                            source=np.array("synthetic"),
                            seed=np.int32(cfg.seed),
                        )
                        bar.update(1)

            logger.info(f"  wrote {count} files to {out_dir}\n")
