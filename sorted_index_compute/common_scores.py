# common_scores.py
# Minimal, shared scoring utilities (faithful to your originals)
from __future__ import annotations
import numpy as np

def _reshape_3d(W2D: np.ndarray) -> np.ndarray:
    if W2D.ndim == 4:
        F, C, H, W = W2D.shape
        return W2D.reshape(F, C, H * W)
    elif W2D.ndim == 3:
        return W2D
    raise ValueError(f"Expected 3D or 4D weight array, got shape {W2D.shape}")

def operator_norm_pruning(W2D: np.ndarray) -> np.ndarray:
    W = _reshape_3d(W2D)
    F, C, HW = W.shape
    C_M, mean_vec = [], []
    for c in range(C):
        A = W[:, c, :]
        A_mean = A.mean(axis=0)
        mean_vec.append(A_mean)
        A_centered = A - A_mean
        u, s, vT = np.linalg.svd(A_centered, full_matrices=False)
        u1 = u[:, :1]
        v1 = vT[:1, :].T
        c1 = (u1 @ v1.T)[0, :]
        c1_norm = c1 / (np.linalg.norm(c1) + 1e-12)
        C_M.append(c1_norm)
    C_M = np.stack(C_M, axis=0)
    mean_vec = np.stack(mean_vec, axis=0)
    Score = []
    for nf in range(F):
        Score.append(np.trace((W[nf, :, :] - mean_vec) @ C_M.T))
    Score = np.asarray(Score, dtype=np.float64) ** 2
    return Score / (np.max(Score) + 1e-12)

def L1_Imp_index(W2D: np.ndarray) -> np.ndarray:
    W = _reshape_3d(W2D)
    Score = np.sum(np.abs(W[:, :, :1]), axis=(1, 2))  # channel-0 slice, as in your snippet
    return Score / (np.max(Score) + 1e-12)

def GM_Imp_index(W2D: np.ndarray) -> np.ndarray:
    W = _reshape_3d(W2D)
    abs_all = np.abs(W).reshape(-1)
    eps = 1e-12
    G_GM = np.exp(np.mean(np.log(abs_all + eps)))
    Diff = []
    for nf in range(W.shape[0]):
        x = np.abs(W[nf]).reshape(-1)
        F_GM = np.exp(np.mean(np.log(x + eps)))
        Diff.append((G_GM - F_GM) ** 2)
    Diff = np.asarray(Diff, dtype=np.float64)
    return Diff / (np.max(Diff) + 1e-12)
