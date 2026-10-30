"""Channel generation for the STAR-RIS MISO downlink (paper Sec. III-A and V-A.1).

Paper settings reproduced here:
  * BS with M antennas, STAR-RIS with N elements, U single-antenna users, direct links blocked;
  * BS is d_BR = 200 m from the STAR-RIS (centrally located), users uniform in a 50 m disc around it;
  * path loss L(d) = 35.6 + 22 log10(d) dB for all links, Rician factor 10, noise power -170 dBm.

The line-of-sight component is built from half-wavelength ULA (BS) / UPA (STAR-RIS) steering vectors.

Conventions: G in C^{N x M} (BS -> RIS), H in C^{N x U} (RIS -> users), both with a leading batch
dimension: G (B, N, M), H (B, N, U).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..config import SystemConfig


def _upa_shape(n: int) -> Tuple[int, int]:
    """Split N elements into an (Ny x Nz) panel with Ny the largest divisor <= sqrt(N)."""
    ny = int(np.floor(np.sqrt(n)))
    while n % ny != 0:
        ny -= 1
    return ny, n // ny


def _ris_steering(u: np.ndarray, n: int, spacing: float) -> np.ndarray:
    """RIS UPA (in the y-z plane) response to unit direction vectors ``u`` of shape (..., 3) -> (..., N)."""
    ny, nz = _upa_shape(n)
    p = np.repeat(np.arange(ny), nz)  # (N,)
    q = np.tile(np.arange(nz), ny)  # (N,)
    phase = 2.0 * np.pi * spacing * (p * u[..., 1:2] + q * u[..., 2:3])
    return np.exp(1j * phase)


def _bs_steering(u: np.ndarray, m: int, spacing: float) -> np.ndarray:
    """BS ULA (along y) response to unit direction vectors ``u`` of shape (..., 3) -> (..., M)."""
    idx = np.arange(m)
    return np.exp(1j * 2.0 * np.pi * spacing * idx * u[..., 1:2])


def _pathloss_gain(cfg: SystemConfig, d: np.ndarray) -> np.ndarray:
    pl_db = cfg.pathloss_intercept_db + cfg.pathloss_slope_db * np.log10(np.maximum(d, cfg.min_distance_m))
    return 10.0 ** (-pl_db / 10.0)


def _cn(rng: np.random.Generator, shape: Tuple[int, ...]) -> np.ndarray:
    """Unit-variance circularly-symmetric complex Gaussian."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def generate_channels(cfg: SystemConfig, n_instances: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``n_instances`` independent channel realisations.

    Returns (G, H) with shapes (B, N, M) and (B, N, U), complex128, *including* path loss.
    """
    rng = np.random.default_rng(seed)
    B, M, N, U = n_instances, cfg.M, cfg.N, cfg.U
    kappa = cfg.rician_factor

    # --- geometry -------------------------------------------------------------------
    bs = np.array([-cfg.d_bs_ris_m, 0.0, cfg.bs_height_m])
    ris = np.array([0.0, 0.0, cfg.ris_height_m])
    r = cfg.user_radius_m * np.sqrt(rng.uniform(0.0, 1.0, size=(B, U)))
    ang = rng.uniform(0.0, 2.0 * np.pi, size=(B, U))
    users = np.stack([r * np.cos(ang), r * np.sin(ang), np.full((B, U), cfg.user_height_m)], axis=-1)  # (B,U,3)

    # --- BS -> RIS ------------------------------------------------------------------
    v_rb = bs - ris
    d_rb = np.linalg.norm(v_rb)
    u_ris_to_bs = v_rb / d_rb
    u_bs_to_ris = -u_ris_to_bs
    a_ris = _ris_steering(u_ris_to_bs[None, :], N, cfg.antenna_spacing_wavelengths)[0]  # (N,)
    a_bs = _bs_steering(u_bs_to_ris[None, :], M, cfg.antenna_spacing_wavelengths)[0]  # (M,)
    G_los = np.outer(a_ris, a_bs.conj())  # (N, M)
    g_gain = float(_pathloss_gain(cfg, np.array(d_rb)))
    G_nlos = _cn(rng, (B, N, M))
    if cfg.fading == "rician":
        G = np.sqrt(g_gain) * (np.sqrt(kappa / (kappa + 1.0)) * G_los[None] + np.sqrt(1.0 / (kappa + 1.0)) * G_nlos)
    else:
        G = np.sqrt(g_gain) * G_nlos

    # --- RIS -> users ---------------------------------------------------------------
    v_ru = users - ris  # (B,U,3)
    d_ru = np.linalg.norm(v_ru, axis=-1)  # (B,U)
    u_ru = v_ru / d_ru[..., None]
    a_ru = _ris_steering(u_ru, N, cfg.antenna_spacing_wavelengths)  # (B,U,N)
    h_gain = _pathloss_gain(cfg, d_ru)  # (B,U)
    H_nlos = _cn(rng, (B, N, U))
    H_los = np.transpose(a_ru, (0, 2, 1))  # (B,N,U)
    if cfg.fading == "rician":
        H = np.sqrt(h_gain)[:, None, :] * (
            np.sqrt(kappa / (kappa + 1.0)) * H_los + np.sqrt(1.0 / (kappa + 1.0)) * H_nlos
        )
    else:
        H = np.sqrt(h_gain)[:, None, :] * H_nlos
    return G.astype(np.complex128), H.astype(np.complex128)


def perturb_csi(
    G: np.ndarray, H: np.ndarray, eta: float, noise_power_w: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Imperfect CSI of Sec. V-A.8: g_hat = g + dg with dg ~ CN(0, eta * sigma^2) for every entry."""
    if eta <= 0.0:
        return G.copy(), H.copy()
    rng = np.random.default_rng(seed)
    std = np.sqrt(eta * noise_power_w)
    return G + std * _cn(rng, G.shape), H + std * _cn(rng, H.shape)


def normalize_channels(G: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scale each instance by its mean entry magnitude so that the numbers shown to an LLM are O(1).

    Returns (G_n, H_n, scale_G, scale_H) with shapes (B,N,M), (B,N,U), (B,), (B,).
    The decision variables are invariant to this scaling (beamformers are Frobenius-normalised and the
    RIS only has unit-modulus phases / bounded amplitudes); rates are always computed with the true channels.
    """
    sg = np.mean(np.abs(G), axis=(-2, -1))
    sh = np.mean(np.abs(H), axis=(-2, -1))
    return G / sg[..., None, None], H / sh[..., None, None], sg, sh
