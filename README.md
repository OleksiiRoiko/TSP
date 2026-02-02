# TSP-GNN (TSPGNN)

Minimal pipeline for generating TSP data, training edge-MLP models, evaluating, and visualizing results.

## Quickstart

1) Set Python path (PowerShell):
```
$env:PYTHONPATH = "$PWD/src"
```

2) Generate synthetic data:
```
python -m tspgnn.cli generate
```

3) Download/process TSPLIB:
```
python -m tspgnn.cli tsplib
```

4) QA the dataset:
```
python -m tspgnn.cli qa
```

5) Train:
```
python -m tspgnn.cli train
```

6) Evaluate / visualize:
```
python -m tspgnn.cli eval
python -m tspgnn.cli visualize
```

## Configs

`config.yaml` is the base config for data/QA. Use small per-experiment overrides via `base:`
for train/eval/visualize.

Example:
```
base: ../config.yaml
train:
  exp_id: exp_edge_mlp_h128_d2
  model_name: edge_mlp
  hidden: 128
  depth: 2
```

Run:
```
python -m tspgnn.cli --config configs/exp_edge_mlp_h128_d2.yaml train
```

Template:
- `configs/_template.yaml`

## Outputs

All training outputs go under:
```
runs/experiments/<exp_id>/<timestamp>/
```

The latest run pointer is:
```
runs/experiments/<exp_id>/latest.json
```

Eval/visualize outputs:
- If `eval.save_json: auto`, results are written under the run directory:
  `runs/experiments/<exp_id>/<timestamp>/evals/eval_<dataset>.json`
- If `visualize.out_dir: auto`, figures are written under:
  `runs/experiments/<exp_id>/<timestamp>/figs/<dataset>/`

Multi-dataset runs:
- `eval.data_roots` evaluates multiple datasets in one command
- `visualize.targets` renders multiple datasets in one command
