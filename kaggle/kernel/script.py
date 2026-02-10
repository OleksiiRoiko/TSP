from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


READ_ROOT = Path(__file__).resolve().parent
WORK_ROOT = Path("/kaggle/working/TSP")
RUNS_DIR = WORK_ROOT / "runs"
DATA_DIR = RUNS_DIR / "data"


class _Tee:
    def __init__(self, *streams) -> None:
        self._streams = streams

    def write(self, data: str) -> None:
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                # Avoid failing process shutdown if one stream is already closed.
                continue

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                continue


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], **kwargs) -> None:
    log(" ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def _dataset_has_data(path: Path) -> bool:
    if (path / "data.zip").exists() or (path / "data").exists() or (path / "runs" / "data").exists():
        return True
    if _looks_like_data_root(path):
        return True
    if list(path.glob("*.zip")):
        return True
    return False


def _find_dataset_dir(slug: str) -> Path | None:
    base = Path("/kaggle/input")
    if slug:
        candidate = base / slug
        if candidate.exists() and _dataset_has_data(candidate):
            return candidate
    if base.exists():
        for d in base.iterdir():
            if not d.is_dir():
                continue
            if _dataset_has_data(d):
                return d
    return None


def _looks_like_data_root(path: Path) -> bool:
    markers = ("synthetic", "synthetic_concorde_v1", "tsplib")
    return any((path / m).exists() for m in markers)


def ensure_data() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        log(f"Using existing data at {DATA_DIR}")
        return

    dataset_slug = os.environ.get("KAGGLE_DATASET_SLUG", "tsp-data-v3")
    dataset_dir = _find_dataset_dir(dataset_slug)
    if dataset_dir is None:
        available = []
        base = Path("/kaggle/input")
        if base.exists():
            available = [d.name for d in base.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"Dataset not found or missing data files. "
            f"Expected {dataset_slug} with data.zip, data/, or runs/data/. "
            f"Available: {available}"
        )

    log(f"Using dataset: {dataset_dir}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = WORK_ROOT / "_dataset_extract"
    if tmp.exists():
        shutil.rmtree(tmp)

    zip_path = dataset_dir / "data.zip"
    if zip_path.exists():
        log(f"Unzipping {zip_path}")
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        src = tmp / "data" if (tmp / "data").exists() else tmp
    elif (dataset_dir / "data").exists():
        src = dataset_dir / "data"
    elif (dataset_dir / "runs" / "data").exists():
        src = dataset_dir / "runs" / "data"
    elif _looks_like_data_root(dataset_dir):
        # Some dataset versions are published with split roots directly.
        src = dataset_dir
    else:
        # Fallback: extract all zip files from dataset root and probe again.
        zip_parts = list(dataset_dir.glob("*.zip"))
        if zip_parts:
            log(f"Extracting dataset zip parts: {[z.name for z in zip_parts]}")
            tmp.mkdir(parents=True, exist_ok=True)
            for zpath in zip_parts:
                with zipfile.ZipFile(zpath, "r") as zf:
                    zf.extractall(tmp)
            if (tmp / "data").exists():
                src = tmp / "data"
            elif (tmp / "runs" / "data").exists():
                src = tmp / "runs" / "data"
            elif _looks_like_data_root(tmp):
                src = tmp
            else:
                raise FileNotFoundError("Extracted zip parts, but no recognizable data root found.")
        else:
            raise FileNotFoundError("Dataset does not contain data.zip, data/, runs/data/, or known split roots.")

    log(f"Copying dataset from {src} to {DATA_DIR}")
    shutil.copytree(src, DATA_DIR, dirs_exist_ok=True)
    if tmp.exists():
        shutil.rmtree(tmp)


def install_deps() -> None:
    req = WORK_ROOT / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(req)])


def _find_project_root() -> Path | None:
    candidates = [
        READ_ROOT,
        READ_ROOT.parent,
        WORK_ROOT,
        Path("/kaggle/working"),
        Path("/kaggle/input"),
    ]
    for base in candidates:
        if (base / "src" / "tspgnn").exists():
            return base
    inp = Path("/kaggle/input")
    if inp.exists():
        for d in inp.iterdir():
            if d.is_dir() and (d / "src" / "tspgnn").exists():
                return d
    return None


def stage_workdir() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    proj_root = _find_project_root()
    if proj_root is None:
        bundle = READ_ROOT / "project.zip"
        if not bundle.exists():
            # Fallback: look for project.zip in any Kaggle input dataset
            inp = Path("/kaggle/input")
            if inp.exists():
                hits = list(inp.rglob("project.zip"))
                if hits:
                    bundle = hits[0]
        if bundle.exists():
            log(f"Extracting bundle: {bundle}")
            with zipfile.ZipFile(bundle, "r") as zf:
                zf.extractall(WORK_ROOT)
            proj_root = WORK_ROOT
        else:
            try:
                log(f"READ_ROOT contents: {[p.name for p in READ_ROOT.iterdir()]}")
            except Exception:
                log("READ_ROOT contents: <unavailable>")
            raise FileNotFoundError("Cannot find project root with src/tspgnn in Kaggle environment.")
    log(f"Project root: {proj_root}")

    # copy code + configs into writable working dir
    if proj_root != WORK_ROOT:
        shutil.copytree(proj_root / "src", WORK_ROOT / "src", dirs_exist_ok=True)
        shutil.copytree(proj_root / "configs", WORK_ROOT / "configs", dirs_exist_ok=True)

    def _copy_with_fallbacks(name: str) -> None:
        candidates = [
            proj_root / name,
            READ_ROOT / name,
            proj_root / "kaggle" / "kernel" / name,
            READ_ROOT / "kaggle" / "kernel" / name,
        ]
        for src in candidates:
            if not src.exists():
                continue
            dst = WORK_ROOT / name
            try:
                if src.resolve() == dst.resolve():
                    return
            except Exception:
                pass
            shutil.copy2(src, dst)
            return

    for name in ("requirements.txt", "configs_to_run.txt"):
        _copy_with_fallbacks(name)


def resolve_configs() -> list[str]:
    cfg_file = WORK_ROOT / "configs_to_run.txt"
    if cfg_file.exists():
        cfgs: list[str] = []
        for raw in cfg_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                path = WORK_ROOT / path
            cfgs.append(str(path))
        if cfgs:
            return cfgs

    env_list = os.environ.get("EXP_CFGS", "").strip()
    if env_list:
        return [c.strip() for c in env_list.split(",") if c.strip()]

    single = os.environ.get("EXP_CFG", "").strip()
    if single:
        return [single]

    default = WORK_ROOT / "configs" / "exp_edge_res_h128_d4_ccv1.yaml"
    if default.exists():
        return [str(default)]

    cfg_paths = sorted((WORK_ROOT / "configs").glob("*.yaml"))
    if not cfg_paths:
        raise FileNotFoundError("No configs found in configs/")
    return [str(cfg_paths[0])]


def prune_output_artifacts() -> None:
    # Keep only experiment artifacts/logs in Kaggle output to reduce archive size.
    for name in ("data", "cache"):
        path = RUNS_DIR / name
        if path.exists():
            log(f"Pruning output artifact: {path}")
            shutil.rmtree(path, ignore_errors=True)


def main() -> None:
    log_path = RUNS_DIR / "kaggle_run.log"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = _Tee(orig_stdout, log_file)
    sys.stderr = _Tee(orig_stderr, log_file)

    try:
        stage_workdir()
        install_deps()
        ensure_data()

        cfgs = resolve_configs()
        resolved_cfgs: list[str] = []
        for cfg in cfgs:
            p = Path(cfg)
            if not p.exists():
                raise FileNotFoundError(f"Config path does not exist: {cfg}")
            resolved_cfgs.append(str(p))
        cfgs = resolved_cfgs

        env = os.environ.copy()
        env["PYTHONPATH"] = str(WORK_ROOT / "src")

        for cfg in cfgs:
            run([sys.executable, "-m", "tspgnn.cli", "--config", cfg, "train"], env=env, cwd=str(WORK_ROOT))
            run([sys.executable, "-m", "tspgnn.cli", "--config", cfg, "eval"], env=env, cwd=str(WORK_ROOT))
        prune_output_artifacts()
    finally:
        # Restore original streams before closing file to prevent shutdown-time tracebacks.
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_file.flush()
        log_file.close()


if __name__ == "__main__":
    main()
