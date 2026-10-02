#!/usr/bin/env python3
"""Evaluate the saved zero-loss vanilla VAE by physical case and failure regime.

The checkpoint is read with NumPy so this audit does not need a notebook kernel or
PyTorch. Only checkpoints produced by the local VanillaVAE architecture are accepted.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
from pathlib import Path
import pickle
import zipfile

import h5py
import numpy as np


class CheckpointReader(pickle.Unpickler):
    def __init__(self, data: bytes, archive: zipfile.ZipFile, prefix: str):
        super().__init__(io.BytesIO(data))
        self.archive = archive
        self.prefix = prefix

    def find_class(self, module: str, name: str):
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("torch", "FloatStorage"): "float32",
            ("torch._utils", "_rebuild_tensor_v2"): self.rebuild_tensor,
        }
        try:
            return allowed[(module, name)]
        except KeyError as error:
            raise ValueError(f"Unsupported checkpoint object: {module}.{name}") from error

    def persistent_load(self, identity):
        kind, dtype, key, device, count = identity
        if kind != "storage" or dtype != "float32" or device != "cpu":
            raise ValueError(f"Unsupported checkpoint storage: {identity}")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        result = np.frombuffer(raw, dtype="<f4")
        if result.size != count:
            raise ValueError(f"Storage {key} has {result.size} values, expected {count}")
        return result

    @staticmethod
    def rebuild_tensor(storage, offset, shape, strides, requires_grad, hooks):
        return np.lib.stride_tricks.as_strided(
            storage[offset:], shape=shape, strides=tuple(4 * step for step in strides)
        ).copy()


def load_checkpoint(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        checkpoint = CheckpointReader(archive.read(member), archive, member[:-9]).load()
    if checkpoint.get("target_key") != "clean":
        raise ValueError("This audit requires the zero-loss checkpoint (target_key='clean')")
    if int(checkpoint["latent_dim"]) != 12:
        raise ValueError("Unexpected latent dimension")
    return checkpoint


def linear(x: np.ndarray, state: dict, layer: str) -> np.ndarray:
    return x @ state[f"{layer}.weight"].T + state[f"{layer}.bias"]


def predict_counts(noisy: np.ndarray, checkpoint: dict, batch_size: int = 512) -> np.ndarray:
    state = checkpoint["model_state_dict"]
    scale = np.float32(checkpoint["transform_scale"])
    predictions = []
    for start in range(0, len(noisy), batch_size):
        counts = noisy[start : start + batch_size]
        x = np.clip(2.0 * np.sqrt(np.maximum(counts, 0.0) + 0.375) / scale, 0, 1)
        for layer in ("encoder.0", "encoder.2", "encoder.4"):
            x = np.maximum(linear(x, state, layer), 0)
        x = linear(x, state, "mu")
        for layer in ("decoder.0", "decoder.2", "decoder.4"):
            x = np.maximum(linear(x, state, layer), 0)
        x = linear(x, state, "decoder.6")
        transformed = np.clip(x, 0, 1) * scale
        predictions.append(np.maximum((transformed / 2) ** 2 - 0.375, 0))
    return np.concatenate(predictions)


def read_split(root: Path, split: str):
    with (root / "dataset_index.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    arrays = {key: [] for key in (
        "noisy", "clean", "case_id", "target_maximum_counts", "pi_fraction",
        "charge_shift_eV", "active_peaks", "family", "sweep",
        "median_fwhm_eV",
    )}
    axis = None
    for item in rows:
        with h5py.File(root / item["file"], "r") as handle:
            n = len(handle["spectra/noisy_counts"])
            arrays["noisy"].append(handle["spectra/noisy_counts"][:])
            arrays["clean"].append(handle["spectra/clean_zero_loss_counts"][:])
            arrays["case_id"].append(np.full(n, int(item["case_id"])))
            arrays["target_maximum_counts"].append(handle["acquisition_labels/target_maximum_counts"][:])
            fractions = handle["peaks/area_fraction"][:]
            arrays["pi_fraction"].append(np.full(n, float(fractions[4])))
            arrays["charge_shift_eV"].append(np.full(n, float(handle["sample_labels/charge_shift_eV"][()])))
            arrays["active_peaks"].append(np.full(n, int(handle["peaks/present"][:].sum())))
            present = handle["peaks/present"][:].astype(bool)
            arrays["median_fwhm_eV"].append(np.full(n, float(np.median(handle["peaks/fwhm_eV"][:][present]))))
            arrays["family"].append(np.full(n, item["atmospheric_family"]))
            arrays["sweep"].append(handle["acquisition_labels/noise_realization_id"][:])
            if axis is None:
                axis = handle["energy_eV"][:]
    if not rows:
        raise ValueError(f"No {split} cases found in {root}")
    return {key: np.concatenate(value) for key, value in arrays.items()}, axis


def group_summary(label: str, mask: np.ndarray, data: dict) -> dict:
    values = data["nrmse"][mask]
    case_ids = data["case_id"][mask]
    case_medians = [np.median(values[case_ids == case_id]) for case_id in np.unique(case_ids)]
    return {
        "group": label,
        "acquisitions": int(mask.sum()),
        "cases": int(np.unique(case_ids).size),
        "nrmse_p50": float(np.median(values)) if len(values) else float("nan"),
        "nrmse_p90": float(np.quantile(values, 0.9)) if len(values) else float("nan"),
        "nrmse_p95": float(np.quantile(values, 0.95)) if len(values) else float("nan"),
        "case_median_nrmse_p50": float(np.median(case_medians)) if case_medians else float("nan"),
        "height_error_p50_pct": float(100 * np.median(data["height_error"][mask])) if len(values) else float("nan"),
        "height_error_p90_pct": float(100 * np.quantile(data["height_error"][mask], 0.9)) if len(values) else float("nan"),
        "position_error_p50_eV": float(np.median(data["position_error"][mask])) if len(values) else float("nan"),
        "position_error_p90_eV": float(np.quantile(data["position_error"][mask], 0.9)) if len(values) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("atmospheric_c1s_h5"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/atmospheric_vanilla_vae/best_vanilla_vae_zero_loss.pt"))
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("outputs/atmospheric_vanilla_vae/error_slices.csv"))
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    data, axis = read_split(args.data_root, args.split)
    prediction = predict_counts(data["noisy"], checkpoint)
    target = data["clean"]
    data["nrmse"] = np.sqrt(np.mean((prediction - target) ** 2, axis=1)) / np.maximum(np.ptp(target, axis=1), 1)
    data["area_error"] = np.abs(prediction.sum(axis=1) - target.sum(axis=1)) / np.maximum(target.sum(axis=1), 1)
    data["height_error"] = np.abs(prediction.max(axis=1) - target.max(axis=1)) / np.maximum(target.max(axis=1), 1)
    data["position_error"] = np.abs(axis[np.argmax(prediction, axis=1)] - axis[np.argmax(target, axis=1)])

    counts = data["target_maximum_counts"]
    pi = data["pi_fraction"]
    shift = data["charge_shift_eV"]
    peaks = data["active_peaks"]
    widths = data["median_fwhm_eV"]
    masks = [
        ("all", np.ones(len(counts), dtype=bool)),
        ("original_cohort", data["case_id"] < 12000),
        ("continuation_cohort", data["case_id"] >= 12000),
        ("counts_200_1000", counts < 1000),
        ("counts_1000_10000", (counts >= 1000) & (counts < 10000)),
        ("counts_10000_plus", counts >= 10000),
        ("pi_absent_or_trace", pi < 0.03),
        ("pi_moderate", (pi >= 0.03) & (pi < 0.15)),
        ("pi_strong", pi >= 0.15),
        ("charge_negative_tail", shift < -0.5),
        ("charge_central", (shift >= -0.5) & (shift <= 1.5)),
        ("charge_positive_tail", shift > 1.5),
        ("weak_pi_absent_or_trace", (counts < 1000) & (pi < 0.03)),
        ("weak_pi_moderate", (counts < 1000) & (pi >= 0.03) & (pi < 0.15)),
        ("weak_and_pi_strong", (counts < 1000) & (pi >= 0.15)),
        ("weak_pi_strong_negative", (counts < 1000) & (pi >= 0.15) & (shift < -0.5)),
        ("weak_pi_strong_positive", (counts < 1000) & (pi >= 0.15) & (shift > 1.5)),
        ("weak_and_charge_positive", (counts < 1000) & (shift > 1.5)),
        ("weak_and_five_peaks", (counts < 1000) & (peaks == 5)),
        ("narrow_peaks_lt_1p1", widths < 1.1),
        ("wide_peaks_ge_1p4", widths >= 1.4),
        ("weak_and_narrow_peaks", (counts < 1000) & (widths < 1.1)),
        ("weak_pi_strong_narrow", (counts < 1000) & (pi >= 0.15) & (widths < 1.1)),
    ]
    masks.extend((f"active_peaks_{n}", peaks == n) for n in range(2, 6))
    summaries = [group_summary(name, mask, data) for name, mask in masks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    rows_path = args.output.with_name(args.output.stem + "_acquisitions.csv")
    columns = ("case_id", "sweep", "target_maximum_counts", "pi_fraction", "charge_shift_eV", "active_peaks", "median_fwhm_eV", "family", "nrmse", "area_error", "height_error", "position_error")
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*(data[key] for key in columns)))
    for row in summaries:
        print(f"{row['group']:<28} n={row['acquisitions']:>5} cases={row['cases']:>4} NRMSE p50={row['nrmse_p50']:.4f} p90={row['nrmse_p90']:.4f} height p50={row['height_error_p50_pct']:.1f}%")
    print(f"Saved {args.output} and {rows_path}")


if __name__ == "__main__":
    main()
