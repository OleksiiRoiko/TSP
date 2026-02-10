from __future__ import annotations

MODEL_NAME_ALIASES: dict[str, str] = {
    "edge_mlp": "edge_mlp",
    "edge_mlp_deep": "edge_mlp_deep",
    "deep": "edge_mlp_deep",
    "edge_res_mlp": "edge_res_mlp",
    "edge_res": "edge_res_mlp",
    "res_mlp": "edge_res_mlp",
    "res": "edge_res_mlp",
    "edge_transformer": "edge_transformer",
    "edge_tf": "edge_transformer",
    "transformer": "edge_transformer",
}


def allowed_model_names() -> list[str]:
    return sorted(MODEL_NAME_ALIASES.keys())


def canonical_model_name(name: str | None) -> str:
    key = str(name or "edge_mlp").lower()
    if key not in MODEL_NAME_ALIASES:
        raise ValueError(f"Unknown model '{name}'. Supported: {', '.join(allowed_model_names())}.")
    return MODEL_NAME_ALIASES[key]
