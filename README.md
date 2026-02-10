# TSP-GNN (TSPGNN)

Minimal pipeline for generating TSP data, training edge-MLP models, evaluating, and visualizing results.

## Quickstart

1) Set Python path (PowerShell):
```
$env:PYTHONPATH = "$PWD/src"
```

2) Generate Concorde-labeled synthetic data:
```
python -m tspgnn.cli --config configs/generate_concorde.yaml generate
```

3) (Optional) Generate regular synthetic data:
```
python -m tspgnn.cli --config configs/generate_synthetic.yaml generate
```

4) Download/process TSPLIB:
```
python -m tspgnn.cli --config configs/tsplib.yaml tsplib
```

5) QA all datasets:
```
python -m tspgnn.cli --config configs/qa.yaml qa
```
Strict Concorde QA:
```
python -m tspgnn.cli --config configs/qa_concorde.yaml qa
```
(`qa_concorde.yaml` also checks `concorde_optimal_proved` when present.)

6) Train:
```
python -m tspgnn.cli --config configs/exp_edge_res_h256_d4_ccv1.yaml train
```

7) Evaluate / visualize:
```
python -m tspgnn.cli --config configs/exp_edge_res_h256_d4_ccv1.yaml eval
python -m tspgnn.cli --config configs/exp_edge_res_h256_d4_ccv1.yaml visualize
```

## Configs

Root `config.yaml` is not part of the workflow. Pass `--config <file>` explicitly for each command.

Data/QA configs:
- `configs/generate_concorde.yaml`
- `configs/generate_synthetic.yaml`
- `configs/tsplib.yaml`
- `configs/qa.yaml`
- `configs/qa_concorde.yaml` (strict Concorde checks)

Experiment configs:
- `configs/exp_*.yaml`
- `configs/_template.yaml`
- `configs/local_configs_to_run.txt` (4 local configs)
- `kaggle/kernel/configs_to_run.txt` (6 Kaggle configs)

Active experiment set is reduced to 10 configs (5 synthetic + 5 ccv1) with simple defaults
(`epochs: 10`, `val_every: 1`, `lr_scheduler: plateau`, `lr_patience: 2`,
`early_stop: true`, `early_patience: 2`).

Run an experiment:
```
python -m tspgnn.cli --config configs/exp_edge_mlp_h128_d2_ccv1.yaml train
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

## Kaggle Output (Download/Extract)

PowerShell UTF-8 (avoids `charmap` console errors):
```
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"
```

Download kernel output:
```
kaggle kernels output <owner>/<kernel-slug> -p kaggle/output --force
```

Extract only useful artifacts (fast, no full unpack):
```
tar -xf kaggle/output/_output_.zip -C kaggle/output/unpacked `
  TSP/runs/experiments TSP/runs/logs TSP/runs/kaggle_run.log
```

If full extraction is needed:
```
Expand-Archive -LiteralPath kaggle/output/_output_.zip `
  -DestinationPath kaggle/output/unpacked -Force
```

Merge Kaggle artifacts into local `runs` (single source of truth):
```
robocopy kaggle\output\unpacked\TSP\runs\experiments runs\experiments /E
robocopy kaggle\output\unpacked\TSP\runs\logs runs\logs /E
```

Quick verify after merge:
```
Get-ChildItem runs\experiments | Select-Object Name,LastWriteTime
```

Cleanup temporary Kaggle output files:
```
if (Test-Path kaggle\output\unpacked) { Remove-Item -Recurse -Force kaggle\output\unpacked }
if (Test-Path kaggle\output\_output_.zip) { Remove-Item -Force kaggle\output\_output_.zip }
if (Test-Path kaggle\output\tsp-train-eval.log) { Remove-Item -Force kaggle\output\tsp-train-eval.log }
```
