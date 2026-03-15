# Configs

This folder is organized by role. Use explicit config paths, e.g.
`python -m tspgnn.cli --config configs/<group>/<name>.yaml <cmd>`.

Structure:
- `configs/data/` -> generation, TSPLIB import, QA
- `configs/experiments/` -> frozen experiment/train specs
- `configs/benchmark/` -> decode profiles, fair baselines, fair analysis
- `configs/lists/` -> local helper lists

Data generation / QA:
- `data/generate_concorde.yaml` -> builds `runs/data/synthetic_concorde_v1`
- `data/generate_synthetic.yaml` -> builds `runs/data/synthetic`
- `data/tsplib.yaml` -> downloads/processes TSPLIB
- `data/qa.yaml` -> QA over `runs/data`
- `data/qa_concorde.yaml` -> strict QA for Concorde labels
  - `data/qa.yaml` is a global integrity check and can include mixed label sources.
    For strict optimality checks on Concorde dataset, use `data/qa_concorde.yaml`.

Generation config split:
- `data/generate_concorde.yaml` keeps `tour_solver: concorde` + `concorde_*` fields.
- `data/generate_synthetic.yaml` uses non-Concorde mode (`tour_solver: auto`, `elkai_frac`) and omits `concorde_*`.
- `data/generate_concorde.yaml` enables `concorde_require_optimal_proof: true` (fail-fast if Concorde output does not confirm optimality).
  If strict mode is enabled, tune `concorde_timeout_sec` carefully (timeouts will fail the run).

Template:
- Copy `configs/experiments/_template.yaml` and set `train.exp_id` plus your model params

Run an experiment:
```
python -m tspgnn.cli --config configs/experiments/exp_edge_mlp_h128_d2.yaml train
```

Notes:
- You can use `{exp_id}` in string fields (auto-replaced by train.exp_id)
- `eval.data_roots` lets you evaluate multiple datasets in a single run
- `eval.save_pred_tour` controls whether per-instance tours are stored in eval JSON (default: false)
- Decode defaults are backward-compatible (`decode_multistart: 1`, `decode_noise_std: 0.0`).
  Enable stronger decoding by setting e.g. `decode_multistart: 8` + `decode_noise_std: 0.01`.
  `decode_twoopt_passes: 0` means "run 2-opt until no further improvement".
- Canonical decode profiles are stored in `configs/benchmark/eval_profiles.yaml`:
  - `baseline`: `run_twoopt=true`, `multistart=1`, `noise=0.0`, `twoopt_passes=20`
  - `optimized`: `run_twoopt=true`, `multistart=8`, `noise=0.02`, `twoopt_passes=40`
- Fair-comparison helper configs:
  - `benchmark/baseline_fair_baseline.yaml` -> baselines on TSPLIB with strict shared decode
  - `benchmark/baseline_fair_optimized.yaml` -> baselines on TSPLIB with stronger shared decode
  - `benchmark/analyze_fair_baseline.yaml` -> analysis over `eval_tsplib_baseline.json`
  - `benchmark/analyze_fair_optimized.yaml` -> analysis over `eval_tsplib_optimized.json`
- Recommended methodology:
  - use `eval_tsplib_baseline.json` for strict architecture/data-quality comparison
  - use `eval_tsplib_optimized.json` for fair end-to-end comparison under stronger decoding
  - keep `eval_tsplib.json` only as a historical/default artifact, not as the main fair-comparison target
- `visualize.targets` is required and lets you render multiple datasets in a single run
- Available model names: `edge_mlp`, `edge_mlp_deep`, `edge_res_mlp`, `edge_transformer`
- `edge_transformer.edge_feat_mode` options:
  - `full` (legacy default)
  - `relative`
  - `relative_sincos` (legacy compatible layout used in older experiments)
  - `relative_sincos_v2` (new dense relative layout without zero-only channels)
- QA includes `qa.check_split_overlap` (enabled by default) to detect duplicate samples across train/val/test.
- Core MLP/Res experiment set is reduced to 10 configs (5 synthetic + 5 concorde/ccv1)
  to avoid redundant runs. Transformer configs are kept separately as graph-aware experiments.
- Split:
  - local (4): `configs/lists/local_configs_to_run.txt`
  - Kaggle (6): `kaggle/kernel/configs_to_run.txt`
- Train defaults in active configs are intentionally simple:
  `epochs: 10`, `val_every: 1`, `lr_scheduler: plateau`, `lr_patience: 2`,
  `early_stop: true`, `early_patience: 2`.

Concorde dataset:
- New configs with suffix `_ccv1` use `runs/data/synthetic_concorde_v1` for train/val/test.
- Generate Concorde data with `configs/data/generate_concorde.yaml`.
- Use `configs/data/qa_concorde.yaml` to run strict QA on the Concorde dataset.

Active experiments:
- Synthetic (5):
  `exp_edge_mlp_h128_d2.yaml`, `exp_edge_mlp_h256_d2.yaml`, `exp_edge_mlp_h256_d3.yaml`,
  `exp_edge_res_h128_d4.yaml`, `exp_edge_res_h256_d4.yaml`
- Concorde / ccv1 (5):
  `exp_edge_mlp_h128_d2_ccv1.yaml`, `exp_edge_mlp_h256_d2_ccv1.yaml`,
  `exp_edge_mlp_h256_d3_ccv1.yaml`, `exp_edge_res_h128_d4_ccv1.yaml`,
  `exp_edge_res_h256_d4_ccv1.yaml`

Optional graph-aware (ccv1):
- Baselines:
  - `exp_edge_tf_h128_d3_ccv1.yaml`
  - `exp_edge_tf_h256_d4_ccv1.yaml`
- Optimized variants (relative edge mode + TSPLIB-only eval):
  - `exp_edge_tf_h192_d4_relsc_ccv1.yaml`
  - `exp_edge_tf_h256_d5_relsc_ccv1.yaml`
  - `exp_edge_tf_h320_d3_relsc_ccv1.yaml`
- New variants (`relative_sincos_v2` + multi-start decode), preserving old experiment semantics:
  - `exp_edge_tf_h128_d3_relv2_ms_ccv1.yaml`
  - `exp_edge_tf_h256_d4_relv2_ms_ccv1.yaml`
  - `exp_edge_tf_h256_d5_relv2_ms_ccv1.yaml`
