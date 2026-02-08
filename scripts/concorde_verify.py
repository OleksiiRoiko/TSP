from __future__ import annotations

import argparse
import json
import random
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from tsplib95.loaders import parse as _tsplib_parse


def _read_text_maybe_gzip(path: Path) -> str:
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        try:
            import gzip
            data = gzip.decompress(data)
        except Exception:
            pass
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def _parse_opt_tour(path: Path):
    if not path.exists():
        return None
    seq = []
    in_sec = False
    for line in _read_text_maybe_gzip(path).splitlines():
        s = line.strip()
        if s.upper().startswith("TOUR_SECTION"):
            in_sec = True
            continue
        if s.startswith("-1") or s.upper().startswith("EOF"):
            break
        if in_sec and s:
            for tok in s.split():
                if tok.isdigit():
                    val = int(tok)
                    if val > 0:
                        seq.append(val - 1)
    return seq if seq else None


def _parse_tour_file(path: Path, n: int):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        try:
            text = path.read_text(encoding="latin-1")
        except Exception:
            return None
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    if -1 in nums:
        nums = nums[:nums.index(-1)]
    if not nums:
        return None
    if nums[0] == n and len(nums) >= n + 1:
        nums = nums[1:]
    if len(nums) >= n + 1 and nums[0] == nums[-1]:
        nums = nums[:-1]
    if len(nums) > n:
        nums = nums[:n]
    if len(nums) != n:
        return None
    if min(nums) >= 1 and max(nums) == n:
        nums = [x - 1 for x in nums]
    if len(set(nums)) != n:
        return None
    if min(nums) < 0 or max(nums) >= n:
        return None
    return np.asarray(nums, dtype=np.int64)


def _tour_len(problem, tour: np.ndarray) -> float:
    n = len(tour)
    L = 0.0
    for i in range(n):
        a = int(tour[i]) + 1
        b = int(tour[(i + 1) % n]) + 1
        L += float(problem.get_weight(a, b))
    return L


def _run_concorde(tsp_path: Path, cmd: str, timeout: int | None) -> tuple[np.ndarray | None, float, str]:
    cmd_list = shlex.split(cmd) if cmd.strip() else ["concorde"]
    start = time.time()
    with tempfile.TemporaryDirectory(prefix="cc_verify_") as tmp:
        tmpdir = Path(tmp)
        local_tsp = tmpdir / "prob.tsp"
        shutil.copy2(tsp_path, local_tsp)
        out_tour = tmpdir / "prob.tour"
        try:
            subprocess.run(
                cmd_list + ["-o", str(out_tour), str(local_tsp)],
                cwd=str(tmpdir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            return None, time.time() - start, "concorde_not_found"
        except subprocess.TimeoutExpired:
            return None, time.time() - start, "timeout"

        if not out_tour.exists():
            sol = local_tsp.with_suffix(".sol")
            if sol.exists():
                out_tour = sol
            else:
                return None, time.time() - start, "no_tour_output"

        tour = _parse_tour_file(out_tour, _infer_dim(tsp_path))
        if tour is None:
            return None, time.time() - start, "parse_failed"
        return tour, time.time() - start, "ok"


def _infer_dim(tsp_path: Path) -> int:
    text = _read_text_maybe_gzip(tsp_path)
    for line in text.splitlines():
        if line.upper().startswith("DIMENSION"):
            return int(line.split(":")[-1].strip())
    raise ValueError(f"Cannot infer DIMENSION in {tsp_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsp-root", default="runs/data/tsplib/raw", help="Folder with .tsp files")
    ap.add_argument("--out-dir", default="runs/experiments/concorde_verify", help="Output folder")
    ap.add_argument("--names", nargs="*", default=None, help="Optional list of instance names")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit / random sample")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for sampling")
    ap.add_argument("--concorde-cmd", default="concorde", help="Concorde command name/path")
    ap.add_argument("--timeout", type=int, default=120, help="Per-instance timeout (seconds)")
    args = ap.parse_args()

    tsp_root = Path(args.tsp_root)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.names:
        files = [tsp_root / f"{n}.tsp" for n in args.names]
    else:
        files = sorted(tsp_root.glob("*.tsp"))

    files = [f for f in files if f.exists()]
    if not files:
        raise FileNotFoundError(f"No .tsp files found in {tsp_root}")

    if args.limit:
        rng = random.Random(args.seed)
        files = rng.sample(files, min(args.limit, len(files)))

    results = []
    for f in files:
        name = f.stem
        text = _read_text_maybe_gzip(f)
        problem = _tsplib_parse(text)
        try:
            dim = int(str(problem.dimension))
        except Exception:
            dim = int(_infer_dim(f))
        opt_path = f.with_suffix(".opt.tour")
        opt_tour = _parse_opt_tour(opt_path) if opt_path.exists() else None
        opt_len = None
        if opt_tour:
            opt_len = _tour_len(problem, np.asarray(opt_tour, dtype=np.int64))

        tour, sec, status = _run_concorde(f, args.concorde_cmd, args.timeout)
        conc_len = None
        if tour is not None:
            conc_len = _tour_len(problem, tour)

        gap = None
        if opt_len is not None and conc_len is not None and opt_len > 0:
            gap = (conc_len - opt_len) / opt_len * 100.0

        results.append({
            "name": name,
            "n": dim,
            "status": status,
            "concorde_len": conc_len,
            "opt_len": opt_len,
            "gap_pct": gap,
            "seconds": sec,
        })

    out_json = out_root / "concorde_verify.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {out_json}")

    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        gaps = [r["gap_pct"] for r in ok if r["gap_pct"] is not None]
        if gaps:
            print(f"mean gap: {np.mean(gaps):.6f}%")


if __name__ == "__main__":
    main()
