from __future__ import annotations
import argparse
from .config import load_config, load_config_data
from .logging_setup import setup_logger
from .data.generate import run as cmd_generate
from .data.tsplib import run as cmd_tsplib
from .training.train import run as cmd_train
from .eval.evaluate import run as cmd_eval, run_qa as cmd_qa
from .viz.plot import run as cmd_visualize

def main():
    p = argparse.ArgumentParser(prog="tspgnn", description="TSP-GNN unified CLI")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("generate")
    sp.add_parser("tsplib")
    sp.add_parser("eval")
    sp.add_parser("visualize")
    sp.add_parser("qa")
    sp.add_parser("train")
    args = p.parse_args()

    cfg = load_config(args.config)  # YAML -> validated dataclasses
    cfg_data = load_config_data(args.config)
    if args.cmd == "generate":
        logger = setup_logger("generate", "runs/logs/generate.log")
        cmd_generate(cfg.generate, logger)
    elif args.cmd == "tsplib":
        logger = setup_logger("tsplib", "runs/logs/tsplib.log")
        cmd_tsplib(cfg.tsplib, logger)
    elif args.cmd == "train":
        logger = setup_logger("train", f"runs/logs/train_{cfg.train.exp_id}.log")
        cmd_train(cfg.train, logger, full_config=cfg_data, config_path=args.config)
    elif args.cmd == "eval":
        logger = setup_logger("eval", "runs/logs/eval.log")
        cmd_eval(cfg.eval, logger)
    elif args.cmd == "visualize":
        logger = setup_logger("visualize", "runs/logs/visualize.log")
        cmd_visualize(cfg.visualize, logger)
    elif args.cmd == "qa":
        logger = setup_logger("qa", "runs/logs/qa.log")
        cmd_qa(cfg.qa, logger)

if __name__ == "__main__":
    main()
