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

`config.yaml` is the base config. Use small per-experiment overrides via `base:`.

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

## Outputs

All training outputs go under:
```
runs/experiments/<exp_id>/<timestamp>/
```

The latest run pointer is:
```
runs/experiments/<exp_id>/latest.json
```
