#!/usr/bin/env python3
"""Convert SESSA .spc outputs into a 401-point, noisy VAE dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import nnls


LABELS = ("CC_CH", "C_O", "C_eq_O", "O_C_eq_O", "pi_pi_star")
AXIS = np.linspace(280.0, 300.0, 401, dtype=np.float64)
AL_KA_ENERGY = 1486.6
GAUSSIAN_FWHM_FACTOR = 2.0 * np.sqrt(2.0 * np.log(2.0))


def zero_loss_peak_heights(
    zero_loss: np.ndarray,
    centers: np.ndarray,
    fwhm: np.ndarray,
    present: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit isolated Gaussian component heights to a SESSA zero-loss spectrum.

    Heights are in the input spectrum's intensity units. An absent component has
    height zero; the returned NRMSE checks how well the components explain it.
    """
    signal = np.asarray(zero_loss, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    fwhm = np.asarray(fwhm, dtype=np.float64)
    present = np.asarray(present, dtype=bool)
    if signal.shape != AXIS.shape or centers.shape != (len(LABELS),) or fwhm.shape != centers.shape:
        raise ValueError("Unexpected spectrum or component-label shape")
    if not present.any() or not np.isfinite(signal).all() or not np.isfinite(fwhm).all():
        raise ValueError("Peak-height fit requires finite data and a present component")
    if np.any(fwhm[present] <= 0):
        raise ValueError("Present components must have positive FWHM")
    maximum = float(signal.max())
    if maximum <= 0:
        raise ValueError("Zero-loss spectrum has no positive intensity")

    sigma = fwhm[present] / GAUSSIAN_FWHM_FACTOR
    basis = np.exp(-0.5 * ((AXIS[:, None] - centers[present]) / sigma) ** 2)
    relative_heights, _ = nnls(basis, signal / maximum)
    heights = np.zeros(len(LABELS), dtype=np.float32)
    heights[present] = (relative_heights * maximum).astype(np.float32)
    fitted = basis @ relative_heights
    fit_nrmse = float(np.sqrt(np.mean((fitted - signal / maximum) ** 2)))
    return heights, fit_nrmse


def find_spectrum(directory: Path, stem: str) -> Path:
    candidates = sorted(directory.glob(stem + "*.spc"))
    if not candidates:
        raise FileNotFoundError(f"No SESSA spectrum matching {stem}*.spc in {directory}")
    if len(candidates) > 1:
        region_one = [p for p in candidates if "reg1" in p.name.lower()]
        if len(region_one) == 1:
            return region_one[0]
        raise RuntimeError(f"Ambiguous SESSA outputs for {stem}: {candidates}")
    return candidates[0]


def read_spc(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.replace(",", " ").split()
        try:
            numbers = [float(x) for x in fields]
        except ValueError:
            continue
        if len(numbers) >= 2 and np.isfinite(numbers[:2]).all():
            rows.append(numbers[:2])
    if len(rows) < 10:
        raise ValueError(f"Could not read a two-column spectrum from {path}")
    data = np.asarray(rows, dtype=np.float64)
    order = np.argsort(data[:, 0])
    energy, intensity = data[order, 0], data[order, 1]
    energy, unique = np.unique(energy, return_index=True)
    return energy, intensity[unique]


def interpolate(path: Path) -> np.ndarray:
    energy, intensity = read_spc(path)
    if np.median(energy) > 1000.0:
        energy = AL_KA_ENERGY - energy
        order = np.argsort(energy)
        energy, intensity = energy[order], intensity[order]
    if energy.min() > AXIS[0] or energy.max() < AXIS[-1]:
        raise ValueError(f"{path} does not cover 280-300 eV: {energy.min()}-{energy.max()}")
    return np.interp(AXIS, energy, intensity).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("sessa_generated"))
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--output", type=Path, default=Path("atmospheric_c1s_60000.npz"))
    args = parser.parse_args()

    root = args.input.resolve()
    spectra_dir = root / "spectra"
    with (root / "physical_cases.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    n_cases, repeats = len(rows), args.replicates
    n = n_cases * repeats
    noisy = np.empty((n, AXIS.size), dtype=np.float32)
    expected_total = np.empty_like(noisy)
    clean = np.empty_like(noisy)
    background = np.empty_like(noisy)
    centers = np.empty((n, 5), dtype=np.float32)
    widths = np.empty((n, 5), dtype=np.float32)
    fractions = np.empty((n, 5), dtype=np.float32)
    heights = np.empty((n, 5), dtype=np.float32)
    height_fit_nrmse = np.empty(n, dtype=np.float32)
    case_ids = np.empty(n, dtype=np.int32)
    sweeps = np.empty(n, dtype=np.int16)
    count_scale = np.empty(n, dtype=np.float32)
    rng = np.random.default_rng(args.seed)

    out = 0
    for case_index, row in enumerate(rows):
        full_path = find_spectrum(spectra_dir, f"case_{case_index:05d}_full")
        zero_path = find_spectrum(spectra_dir, f"case_{case_index:05d}_zero")
        full = np.maximum(interpolate(full_path), 0.0)
        zero = np.minimum(np.maximum(interpolate(zero_path), 0.0), full)
        component_centers = np.array([float(row[f"center_{x}"]) for x in LABELS])
        component_widths = np.array([float(row[f"fwhm_{x}"]) for x in LABELS])
        component_fractions = np.array([float(row[f"fraction_{x}"]) for x in LABELS])
        raw_heights, fit_nrmse = zero_loss_peak_heights(
            zero, component_centers, component_widths, component_fractions > 0
        )

        for sweep in range(repeats):
            target_max = 10.0 ** rng.uniform(np.log10(200.0), np.log10(150_000.0))
            scale = target_max / max(float(full.max()), 1.0e-30)
            expectation = np.maximum(full * scale, 0.0)
            counts = rng.poisson(expectation).astype(np.float32)
            counts += rng.normal(0.0, rng.uniform(0.0, 1.5), counts.size).astype(np.float32)
            counts = np.maximum(counts, 0.0)

            noisy[out] = counts
            expected_total[out] = expectation
            clean[out] = zero * scale
            background[out] = np.maximum(full - zero, 0.0) * scale
            centers[out] = component_centers
            widths[out] = component_widths
            fractions[out] = component_fractions
            heights[out] = raw_heights * scale
            height_fit_nrmse[out] = fit_nrmse
            case_ids[out] = case_index
            sweeps[out] = sweep
            count_scale[out] = scale
            out += 1

        if (case_index + 1) % 100 == 0:
            print(f"Converted {case_index + 1}/{n_cases} physical cases")

    np.savez_compressed(
        args.output.resolve(),
        axis=AXIS.astype(np.float32),
        noisy=noisy,
        clean=clean,
        background=background,
        expected_total=expected_total,
        peak_center=centers,
        peak_width=widths,
        peak_fraction=fractions,
        peak_height=heights,
        peak_height_fit_nrmse=height_fit_nrmse,
        case_id=case_ids,
        sweep=sweeps,
        count_scale=count_scale,
        component_names=np.asarray(LABELS),
    )
    print(args.output.resolve())
    print(f"Saved {n:,} spectra ({n_cases:,} cases x {repeats} acquisitions).")


if __name__ == "__main__":
    main()
