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


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: list[str], **kwargs) -> None:
    log(" ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def ensure_data() -> None:
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        log(f"Using existing data at {DATA_DIR}")
        return

    dataset_slug = os.environ.get("KAGGLE_DATASET_SLUG", "tsp-data")
    dataset_dir = Path("/kaggle/input") / dataset_slug
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "_dataset_extract"
    if tmp.exists():
        shutil.rmtree(tmp)

    zip_path = dataset_dir / "data.zip"
    if zip_path.exists():
        log(f"Unzipping {zip_path}")
        tmp.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        src = tmp / "data" if (tmp / "data").exists() else tmp
    else:
        src = dataset_dir / "data" if (dataset_dir / "data").exists() else dataset_dir

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
        if bundle.exists():
            log(f"Extracting bundle: {bundle}")
            with zipfile.ZipFile(bundle, "r") as zf:
                zf.extractall(WORK_ROOT)
            proj_root = WORK_ROOT
        else:
            raise FileNotFoundError("Cannot find project root with src/tspgnn in Kaggle environment.")
    log(f"Project root: {proj_root}")

    # copy code + configs into writable working dir
    if proj_root != WORK_ROOT:
        shutil.copytree(proj_root / "src", WORK_ROOT / "src", dirs_exist_ok=True)
        shutil.copytree(proj_root / "configs", WORK_ROOT / "configs", dirs_exist_ok=True)
    for name in ("requirements.txt", "configs_to_run.txt", "config.yaml"):
        src = proj_root / name
        if src.exists():
            shutil.copy2(src, WORK_ROOT / name)


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

    cfgs = sorted((WORK_ROOT / "configs").glob("*.yaml"))
    if not cfgs:
        raise FileNotFoundError("No configs found in configs/")
    return [str(cfgs[0])]


def main() -> None:
    log_path = RUNS_DIR / "kaggle_run.log"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = log_file
    sys.stderr = log_file

    stage_workdir()
    install_deps()
    ensure_data()

    cfgs = resolve_configs()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORK_ROOT / "src")

    for cfg in cfgs:
        if os.environ.get("SKIP_TRAIN") != "1":
            run([sys.executable, "-m", "tspgnn.cli", "--config", cfg, "train"], env=env, cwd=str(WORK_ROOT))
        if os.environ.get("SKIP_EVAL") != "1":
            run([sys.executable, "-m", "tspgnn.cli", "--config", cfg, "eval"], env=env, cwd=str(WORK_ROOT))

    log_file.flush()
    log_file.close()


if __name__ == "__main__":
    main()
