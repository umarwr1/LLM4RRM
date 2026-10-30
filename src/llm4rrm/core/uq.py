"""Uncertainty-quantification metrics (paper Sec. IV-B, eqs. (13)-(14)).

Output-based sampling: for each of Psi input realisations the same prompt is sent Lambda times and the
Lambda generated vectors y_{psi,lambda} are compared.
"""

from __future__ import annotations

import numpy as np


def d_uq(samples: np.ndarray) -> float:
    """Eq. (13).  ``samples`` has shape (Psi, Lambda, ...).

    D_UQ = 1/Psi * sum_psi  [ 2 / (Lambda (Lambda-1)) * sum_{lambda<s} ||y_{psi,lambda} - y_{psi,s}||_2^2 ].
    """
    y = np.asarray(samples, dtype=float)
    if y.ndim < 2:
        raise ValueError("samples must have shape (Psi, Lambda, ...)")
    psi, lam = y.shape[:2]
    if lam < 2:
        raise ValueError("need at least Lambda = 2 samples per input")
    flat = y.reshape(psi, lam, -1)
    total = 0.0
    for p in range(psi):
        diff = flat[p][:, None, :] - flat[p][None, :, :]
        sq = np.sum(diff ** 2, axis=-1)  # (Lambda, Lambda)
        total += sq[np.triu_indices(lam, k=1)].sum() * 2.0 / (lam * (lam - 1))
    return float(total / psi)


def mad_uq(samples: np.ndarray) -> float:
    """Eq. (14).  ``samples`` has shape (Psi, Lambda, ...).

    MAD_UQ = 1/Psi * sum_psi [ 1/(Lambda*Upsilon) * sum_lambda sum_upsilon | y_{psi,lambda,upsilon} - mean_lambda y_{psi,upsilon} | ].
    """
    y = np.asarray(samples, dtype=float)
    psi, lam = y.shape[:2]
    flat = y.reshape(psi, lam, -1)
    dev = np.abs(flat - flat.mean(axis=1, keepdims=True))
    return float(dev.mean(axis=(1, 2)).mean())


def consistency_score(d: float) -> float:
    """Normalised consistency exp(-D_UQ) used for the binary user-association variable in the paper."""
    return float(np.exp(-d))
