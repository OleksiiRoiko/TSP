from __future__ import annotations
from pathlib import Path
import importlib
from typing import Iterable, Tuple, cast
import numpy as np
import tsplib95
from ..config import TsplibCfg
from ..utils.io import save_npz, download_tsplib_file
from ..utils.tour import tour_length

def _parse_opt_tour(path: Path):
    if not path.exists(): return None
    seq=[]; in_sec=False
    for line in path.read_text("utf-8","ignore").splitlines():
        s=line.strip()
        if s.upper().startswith("TOUR_SECTION"): in_sec=True; continue
        if s.startswith("-1") or s.upper().startswith("EOF"): break
        if in_sec and s:
            for tok in s.split():
                if tok.isdigit():
                    val = int(tok)
                    if val > 0:
                        seq.append(val - 1)
    return seq if seq else None

def _elkai(coords: np.ndarray):
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

def run(cfg: TsplibCfg, logger):
    raw_dir=Path(cfg.raw_root); out_dir=Path(cfg.out_root)
    raw_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.download:
        for nm in cfg.names:
            download_tsplib_file(nm, raw_dir, "tsp")
            download_tsplib_file(nm, raw_dir, "opt.tour")

    for nm in cfg.names:
        tsp = raw_dir / f"{nm}.tsp"
        if not tsp.exists():
            logger.warning(f"skip {nm}: missing {tsp}"); continue
        problem = tsplib95.load(str(tsp))
        if not getattr(problem, "node_coords", None):
            logger.error(f"skip {nm}: missing node_coords")
            continue
        coords_map = problem.node_coords
        xs, ys = [], []
        raw_items = getattr(coords_map, "items", None)
        if callable(raw_items):
            items = cast(Iterable[tuple[int, tuple[float, float]]], raw_items())
        else:
            items = cast(Iterable[tuple[int, tuple[float, float]]], coords_map)
        for _, (x, y) in sorted(items, key=lambda kv: kv[0]):
            xs.append(float(x))
            ys.append(float(y))
        coords_orig = np.stack([np.array(xs), np.array(ys)], axis=1).astype(np.float32)
        minv = coords_orig.min(axis=0); span = np.maximum(coords_orig.max(axis=0)-minv, 1e-9)
        coords = ((coords_orig-minv)/span).astype(np.float32)
        seq = _parse_opt_tour(raw_dir / f"{nm}.opt.tour")
        source="opt.tour"
        if not seq:
            t=_elkai(coords)
            if t is None: logger.error(f"{nm}: no opt.tour and elkai unavailable"); continue
            seq=t.tolist(); source="elkai"
        tour=np.asarray(seq, dtype=np.int64)

        def wlen(tt):
            L=0.0; n=len(tt)
            for i in range(n):
                a,b=int(tt[i])+1, int(tt[(i+1)%n])+1
                L += float(problem.get_weight(a,b))
            return L

        save_npz(out_dir / f"{nm}.npz",
                 id=np.array(nm), n=np.int32(coords.shape[0]),
                 coords=coords, coords_orig=coords_orig,
                 label_tour=tour.astype(np.int32),
                 label_len_norm=float(tour_length(coords, tour)),
                 label_len_tsplib=float(wlen(tour)),
                 metric=np.array(str(problem.edge_weight_type)),
                 label_source=np.array(source),
                 source=np.array("tsplib"),
                 scale_min=minv.astype(np.float32),
                 scale_span=span.astype(np.float32))
        logger.info(f"processed {nm}")
