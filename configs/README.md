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
- `visualize.targets` lets you render multiple datasets in a single run
- Since base config omits eval/visualize, include `eval.model_path` and `visualize.model`
