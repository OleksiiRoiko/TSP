# TSP-GNN (TSPGNN)

Minimal pipeline for generating TSP data, training edge-MLP models, evaluating, and visualizing results.

## Quickstart

1) Set Python path (PowerShell):
```
$env:PYTHONPATH = "$PWD/src"
```

2) Generate Concorde-labeled synthetic data:
```
python -m tspgnn.cli --config configs/data/generate_concorde.yaml generate
```

3) (Optional) Generate regular synthetic data:
```
python -m tspgnn.cli --config configs/data/generate_synthetic.yaml generate
```

4) Download/process TSPLIB:
```
python -m tspgnn.cli --config configs/data/tsplib.yaml tsplib
```

5) QA all datasets:
```
python -m tspgnn.cli --config configs/data/qa.yaml qa
```
Strict Concorde QA:
```
python -m tspgnn.cli --config configs/data/qa_concorde.yaml qa
```
(`qa_concorde.yaml` also checks `concorde_optimal_proved` when present.)

6) Train:
```
python -m tspgnn.cli --config configs/experiments/exp_edge_res_h256_d4_ccv1.yaml train
```

7) Evaluate / visualize:
```
python -m tspgnn.cli --config configs/experiments/exp_edge_res_h256_d4_ccv1.yaml eval
python -m tspgnn.cli --config configs/experiments/exp_edge_res_h256_d4_ccv1.yaml visualize
```

## Configs

Root `config.yaml` is not part of the workflow. Pass `--config <file>` explicitly for each command.

Config layout:
- `configs/data/` -> generation, TSPLIB import, QA
- `configs/experiments/` -> frozen experiment/train specs
- `configs/benchmark/` -> eval profiles, fair baselines, fair analysis
- `configs/lists/` -> local config lists
- `kaggle/kernel/configs_to_run.txt` -> Kaggle config list

Core MLP/Res experiment set is reduced to 10 configs (5 synthetic + 5 ccv1) with simple defaults
(`epochs: 10`, `val_every: 1`, `lr_scheduler: plateau`, `lr_patience: 2`,
`early_stop: true`, `early_patience: 2`). Transformer configs are kept as separate
graph-aware experiments in `configs/experiments/`.

Run an experiment:
```
python -m tspgnn.cli --config configs/experiments/exp_edge_mlp_h128_d2_ccv1.yaml train
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

Eval/visualize outputs:
- If `eval.save_json: auto`, results are written under the run directory:
  `runs/experiments/<exp_id>/<timestamp>/evals/eval_<dataset>.json`
- If `visualize.out_dir: auto`, figures are written under:
  `runs/experiments/<exp_id>/<timestamp>/figs/<dataset>/`

Multi-dataset runs:
- `eval.data_roots` evaluates multiple datasets in one command
- `visualize.targets` renders multiple datasets in one command (required)

## Kaggle Output (Download/Merge)

Current Kaggle CLI layout in this project:
```
kaggle/output/TSP/...
```
(files are downloaded directly, not only as `_output_.zip`).

Download output using kernel id from metadata:
```
$id = (Get-Content kaggle/kernel/kernel-metadata.json | ConvertFrom-Json).id
kaggle kernels output $id -p kaggle/output --force
```

If Kaggle CLI fails with proxy errors, run once with proxy vars cleared:
```
$env:HTTP_PROXY=''; $env:HTTPS_PROXY=''; $env:ALL_PROXY=''
$env:http_proxy=''; $env:https_proxy=''; $env:all_proxy=''
kaggle kernels output $id -p kaggle/output --force
```

Merge Kaggle artifacts into local `runs`:
```
robocopy kaggle\output\TSP\runs\experiments runs\experiments /E
robocopy kaggle\output\TSP\runs\logs runs\logs /E
Copy-Item kaggle\output\TSP\runs\kaggle_run.log runs\logs\kaggle_run_latest.log -Force
```

Quick verify:
```
Get-ChildItem runs\experiments | Select-Object Name,LastWriteTime
Get-ChildItem runs\logs | Select-Object Name,LastWriteTime
```

Optional cleanup:
```
if (Test-Path kaggle\output\TSP) { Remove-Item -Recurse -Force kaggle\output\TSP }
if (Test-Path kaggle\output\tsp-train-eval.log) { Remove-Item -Force kaggle\output\tsp-train-eval.log }
if (Test-Path kaggle\output\_output_.zip) { Remove-Item -Force kaggle\output\_output_.zip }
```

Legacy fallback (if Kaggle returns `_output_.zip`):
```
Expand-Archive -LiteralPath kaggle/output/_output_.zip -DestinationPath kaggle/output/unpacked -Force
```
