# Configs

Use `config.yaml` as the base config for data/QA (generate/tsplib/qa). Per-experiment
configs should override train/eval/visualize.

Recommended pattern:
- Base defaults (data, qa, eval/visualize paths) live in `config.yaml`
- Each experiment file overrides only train parameters and, if needed, dataset roots

Template:
- Copy `configs/_template.yaml` and set `train.exp_id` plus your model params

Run an experiment:
```
python -m tspgnn.cli --config configs/exp_edge_mlp_h128_d2.yaml train
```

Notes:
- You can use `{exp_id}` in string fields (auto-replaced by train.exp_id)
- `eval.data_roots` lets you evaluate multiple datasets in a single run
- `visualize.targets` is required and lets you render multiple datasets in a single run
- Since base config omits eval/visualize, include `eval.model_path` and `visualize.model`
- Available model names: `edge_mlp`, `edge_mlp_deep`, `edge_res_mlp`

Concorde dataset:
- New configs with suffix `_ccv1` use `runs/data/synthetic_concorde_v1` for train/val/test.
- Generate Concorde data with `config.yaml` (now points to `synthetic_concorde_v1`).
- Use `configs/qa_concorde.yaml` to run strict QA on the Concorde dataset.

New residual-MLP experiments:
- `exp_edge_res_h128_d4.yaml`
- `exp_edge_res_h256_d4.yaml`
- `exp_edge_res_h128_d4_ccv1.yaml`
- `exp_edge_res_h256_d4_ccv1.yaml`
