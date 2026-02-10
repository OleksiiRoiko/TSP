from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from ..config import GenerateCfg
from ..utils.io import save_npz
from ..utils.tour import tour_length, two_opt, tour_length_tsplib, verify_tour
from ..utils.elkai_solver import solve_with_elkai
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def _seed_for_instance(base_seed: int, n: int, split: str, idx: int) -> int:
    """
    Build a deterministic per-instance seed that separates train/val/test streams.
    This prevents accidental split overlap when idx numbering restarts per split.
    """
    key = f"{int(base_seed)}|{int(n)}|{str(split).lower()}|{int(idx)}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _make_one(args):
    (
        idx, n, split, seed, dist, solver, elkai_frac,
        concorde_cmd, concorde_scale, concorde_timeout, concorde_keep_tmp,
        concorde_require_optimal_proof,
    ) = args
    rng = np.random.default_rng(_seed_for_instance(seed, n, split, idx))
    coords = _sample(rng, n, dist)
    coords_orig = None
    label_len_tsplib = None
    label_source = None
    concorde_optimal_proved = None

    solver = str(solver).lower()
    if solver == "concorde":
        tour, coords_orig, label_len_tsplib, concorde_optimal_proved = _concorde_tour(
            coords,
            cmd=concorde_cmd,
            scale=concorde_scale,
            timeout=concorde_timeout,
            keep_tmp=concorde_keep_tmp,
        )
        if bool(concorde_require_optimal_proof):
            if tour is None:
                raise RuntimeError("Concorde did not produce a valid tour")
            if concorde_optimal_proved is not True:
                raise RuntimeError("Concorde output does not confirm optimal proof")
        if tour is not None:
            label_source = "concorde"
    elif solver == "elkai":
        tour = solve_with_elkai(coords)
        if tour is not None:
            label_source = "elkai"
    elif solver == "nn2opt":
        tour = None
    else:
        # auto: previous behavior (elkai if available, else NN+2opt)
        use_elkai = bool(rng.random() < float(elkai_frac))
        tour = solve_with_elkai(coords) if use_elkai else None
        if tour is not None:
            label_source = "elkai"

    if tour is None:
        tour = _nn2opt(coords, max_passes=50)
        label_source = "nn2opt"

    # If concorde/tsplib metric exists, compute TSPLIB length too
    if coords_orig is not None and label_len_tsplib is None:
        label_len_tsplib = float(tour_length_tsplib(coords_orig, tour, "EUC_2D"))

    L = tour_length(coords, tour)
    return (
        idx,
        coords.astype(np.float32),
        tour.astype(np.int32),
        float(L),
        dist,
        coords_orig,
        label_len_tsplib,
        label_source,
        concorde_optimal_proved,
    )

def _write_tsplib(path: Path, coords: np.ndarray, name: str = "synthetic"):
    n = coords.shape[0]
    lines = [
        f"NAME: {name}",
        "TYPE: TSP",
        f"DIMENSION: {n}",
        "EDGE_WEIGHT_TYPE: EUC_2D",
        "NODE_COORD_SECTION",
    ]
    for i, (x, y) in enumerate(coords, start=1):
        lines.append(f"{i} {int(x)} {int(y)}")
    lines.append("EOF")
    path.write_text("\n".join(lines), encoding="utf-8")

def _parse_tour_file(path: Path, n: int):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception:
            return None
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    if not nums:
        return None
    if -1 in nums:
        nums = nums[:nums.index(-1)]
    if not nums:
        return None
    # Some formats start with dimension
    if nums[0] == n and len(nums) >= n + 1:
        nums = nums[1:]
    # Some formats repeat the start node at the end
    if len(nums) >= n + 1 and nums[0] == nums[-1]:
        nums = nums[:-1]
    if len(nums) > n:
        nums = nums[:n]
    if not nums or len(nums) != n:
        return None
    if min(nums) >= 1 and max(nums) == n:
        nums = [x - 1 for x in nums]
    if len(nums) != n or len(set(nums)) != n:
        return None
    if min(nums) < 0 or max(nums) >= n:
        return None
    tour = np.asarray(nums, dtype=np.int64)
    return tour if verify_tour(tour, n) else None


def _concorde_log_has_optimal_proof(log_text: str) -> bool:
    t = (log_text or "").lower()
    if not t:
        return False
    negative_markers = (
        "not proven",
        "no proof",
        "time limit",
        "timelimit",
        "timeout",
        "interrupted",
        "aborted",
    )
    if any(m in t for m in negative_markers):
        return False
    return "optimal" in t


def _concorde_tour(
    coords: np.ndarray,
    cmd: str,
    scale: int,
    timeout: int,
    keep_tmp: bool,
):
    n = coords.shape[0]
    coords_scaled = np.rint(coords * float(scale)).astype(np.int64)
    tmp_root = None
    if keep_tmp:
        tmp_root = Path("runs/tmp/concorde")
        tmp_root.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="cc_", dir=str(tmp_root) if tmp_root else None))
    try:
        tsp_path = tmpdir / "prob.tsp"
        _write_tsplib(tsp_path, coords_scaled, name="synthetic")
        out_tour = tmpdir / "prob.tour"
        cmd_list = shlex.split(str(cmd)) if str(cmd).strip() else ["concorde"]
        run_logs: list[str] = []

        try:
            # Try explicit output file first
            res = subprocess.run(
                cmd_list + ["-o", str(out_tour), str(tsp_path)],
                cwd=str(tmpdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=float(timeout) if timeout else None,
            )
            run_logs.append((res.stdout or "") + "\n" + (res.stderr or ""))
            if res.returncode != 0 or not out_tour.exists():
                # Fallback: default concorde output (prob.sol)
                res2 = subprocess.run(
                    cmd_list + [str(tsp_path)],
                    cwd=str(tmpdir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=float(timeout) if timeout else None,
                )
                run_logs.append((res2.stdout or "") + "\n" + (res2.stderr or ""))
        except FileNotFoundError:
            return None, None, None, False
        except subprocess.TimeoutExpired:
            return None, None, None, False

        cand = None
        if out_tour.exists():
            cand = out_tour
        else:
            sol = tsp_path.with_suffix(".sol")
            tour = tsp_path.with_suffix(".tour")
            if sol.exists():
                cand = sol
            elif tour.exists():
                cand = tour
        if cand is None:
            return None, None, None, _concorde_log_has_optimal_proof("\n".join(run_logs))

        tour = _parse_tour_file(cand, n)
        if tour is None:
            return None, None, None, _concorde_log_has_optimal_proof("\n".join(run_logs))

        coords_orig = coords_scaled.astype(np.float32)
        label_len_tsplib = float(tour_length_tsplib(coords_orig, tour, "EUC_2D"))
        optimal_proved = _concorde_log_has_optimal_proof("\n".join(run_logs))
        return tour, coords_orig, label_len_tsplib, optimal_proved
    finally:
        if not keep_tmp:
            shutil.rmtree(tmpdir, ignore_errors=True)

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


def _build_payload(
    *,
    idx: int,
    n: int,
    coords: np.ndarray,
    tour: np.ndarray,
    length_norm: float,
    dist: str,
    seed: int,
    label_source: str | None,
    coords_orig: np.ndarray | None,
    length_tsplib: float | None,
    concorde_optimal_proved: bool | None,
) -> dict:
    payload = {
        "id": np.array(f"syn_{idx:06d}"),
        "n": np.int32(n),
        "coords": coords,
        "label_tour": tour,
        "label_len_norm": float(length_norm),
        "distribution": np.array(dist),
        "metric": np.array("EUC_2D"),
        "source": np.array("synthetic"),
        "seed": np.int32(seed),
        "label_source": np.array(label_source or "unknown"),
    }
    if coords_orig is not None:
        payload["coords_orig"] = coords_orig
    if length_tsplib is not None:
        payload["label_len_tsplib"] = float(length_tsplib)
    if concorde_optimal_proved is not None:
        payload["concorde_optimal_proved"] = bool(concorde_optimal_proved)
    return payload

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

    # worker count: cap auto mode to avoid oversubscription in constrained environments
    cfg_workers = getattr(cfg, "workers", None)
    if cfg_workers is None:
        workers = max(1, min(8, (os.cpu_count() or 1)))
    else:
        workers = max(1, int(cfg_workers))
    solver = str(cfg.tour_solver).lower()
    strict_opt = bool(getattr(cfg, "concorde_require_optimal_proof", False))
    logger.info(f"[gen] tour_solver={solver} concorde_require_optimal_proof={strict_opt}")

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
                tasks.append((
                    start_idx + i, n, split, int(cfg.seed), dist,
                    solver, float(cfg.elkai_frac),
                    str(cfg.concorde_cmd), int(cfg.concorde_scale),
                    int(cfg.concorde_timeout_sec), bool(cfg.concorde_keep_tmp),
                    bool(getattr(cfg, "concorde_require_optimal_proof", False)),
                ))

            # run with a nice progress bar (fallback to single process if needed)
            try:
                with ProcessPoolExecutor(max_workers=workers) as ex:
                    futures = [ex.submit(_make_one, t) for t in tasks]
                    with tqdm(total=len(futures), ncols=100,
                        desc=f"N={n} | {split}", leave=True) as bar:
                        for fut in as_completed(futures):
                            idx, coords, tour, L, dist, coords_orig, L_tsplib, label_source, concorde_optimal_proved = fut.result()
                            # deterministic filename from idx (no glob scans)
                            payload = _build_payload(
                                idx=idx,
                                n=n,
                                coords=coords,
                                tour=tour,
                                length_norm=L,
                                dist=dist,
                                seed=int(cfg.seed),
                                label_source=label_source,
                                coords_orig=coords_orig,
                                length_tsplib=L_tsplib,
                                concorde_optimal_proved=concorde_optimal_proved,
                            )
                            save_npz(out_dir / f"syn_{idx:06d}.npz", **payload)
                            bar.update(1)
            except PermissionError:
                # Some environments disallow multiprocessing; fall back to single-process.
                with tqdm(total=len(tasks), ncols=100,
                          desc=f"N={n} | {split} (single)", leave=True) as bar:
                    for t in tasks:
                        idx, coords, tour, L, dist, coords_orig, L_tsplib, label_source, concorde_optimal_proved = _make_one(t)
                        payload = _build_payload(
                            idx=idx,
                            n=n,
                            coords=coords,
                            tour=tour,
                            length_norm=L,
                            dist=dist,
                            seed=int(cfg.seed),
                            label_source=label_source,
                            coords_orig=coords_orig,
                            length_tsplib=L_tsplib,
                            concorde_optimal_proved=concorde_optimal_proved,
                        )
                        save_npz(out_dir / f"syn_{idx:06d}.npz", **payload)
                        bar.update(1)

            logger.info(f"  wrote {count} files to {out_dir}\n")
