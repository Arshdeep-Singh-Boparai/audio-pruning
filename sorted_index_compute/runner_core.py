# runner_core.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Dict

from common_scores import operator_norm_pruning, L1_Imp_index, GM_Imp_index

METHODS = {"op": operator_norm_pruning, "l1": L1_Imp_index, "gm": GM_Imp_index}

def load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unsupported checkpoint format. Expected a dict or dict with 'model'.")

def score_one_layer(state: Dict[str, torch.Tensor], layer: str, method: str) -> np.ndarray:
    fn = METHODS[method]
    comps = ["r_weight", "i_weight", "j_weight", "k_weight"]
    scores = []
    for c in comps:
        key = f"{layer}.{c}"
        if key not in state:
            raise KeyError(f"Missing key in checkpoint: {key}")
        W = state[key].detach().cpu().numpy()
        scores.append(fn(W))
    scores = np.stack(scores, axis=0)  # [4, F]
    return scores.mean(axis=0)  # [F]

def run_simple(checkpoint: str, out_dir: str, method: str, layers: list[str]) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = load_state_dict(checkpoint)

    for layer in tqdm(layers, desc=f"Scoring ({method})"):
        mean_score = score_one_layer(state, layer, method)
        np.save(out / f"{layer}_mean_score.npy", mean_score)
        np.save(out / f"{layer}_sorted_index.npy", np.argsort(mean_score))

    meta = {"checkpoint": str(Path(checkpoint).resolve()), "method": method, "layers": layers}
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Saved outputs to: {out}")
