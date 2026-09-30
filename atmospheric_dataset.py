"""Load the self-contained atmospheric C 1s HDF5 dataset for VAE training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


SCHEMA_NAME = "atmospheric_xps_c1s_vae"
REQUIRED_SPLITS = {"train", "validation", "test"}


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def h5_manifest(data_root: Path) -> np.ndarray:
    """Return a cache key that changes when the index or case files change."""
    data_root = Path(data_root)
    paths = sorted(data_root.glob("atmospheric_c1s_case_*.h5"))
    index_path = data_root / "dataset_index.csv"
    records = []
    for path in ([index_path] if index_path.exists() else []) + paths:
        stat = path.stat()
        records.append(f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}")
    return np.asarray(records, dtype=np.str_)


def _repeat_rows(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    return np.repeat(values[None, ...], count, axis=0)


def load_atmospheric_c1s(
    data_root: Path,
    target_length: int = 401,
    paths: Iterable[Path] | None = None,
) -> dict[str, np.ndarray]:
    """Load acquisitions and labels, validating the common schema and energy grid."""
    data_root = Path(data_root)
    case_paths = sorted(paths or data_root.glob("atmospheric_c1s_case_*.h5"))
    if not case_paths:
        raise FileNotFoundError(f"No atmospheric C 1s HDF5 files found in {data_root}")

    parts: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "noisy", "clean", "background", "expected_total", "peak_center",
            "peak_width", "peak_fraction", "peak_present", "photon_energy",
            "case_id", "source", "sweep", "split", "atmospheric_family",
            "aging_index", "oxygen_to_carbon_ratio", "mass_density_g_cm3",
            "band_gap_eV", "charge_shift_eV", "count_scale",
            "target_maximum_counts", "read_noise_sigma_counts",
        )
    }
    reference_axis = None
    component_names = None

    for path in case_paths:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_name") != SCHEMA_NAME:
                raise ValueError(f"Unexpected schema in {path}")
            axis = handle["energy_eV"][:].astype(np.float32)
            if axis.shape != (target_length,):
                raise ValueError(f"Expected {target_length} energy points in {path}, got {axis.shape}")
            if reference_axis is None:
                reference_axis = axis
            elif not np.allclose(axis, reference_axis, rtol=0.0, atol=1e-6):
                raise ValueError(f"Energy axis differs in {path}")

            names = np.asarray([_text(value) for value in handle["peaks/machine_names"][:]])
            if component_names is None:
                component_names = names
            elif not np.array_equal(names, component_names):
                raise ValueError(f"Peak component order differs in {path}")

            noisy = handle["spectra/noisy_counts"][:].astype(np.float32)
            n_acquisitions = noisy.shape[0]
            if noisy.shape[1:] != (target_length,):
                raise ValueError(f"Unexpected noisy spectrum shape in {path}: {noisy.shape}")

            parts["noisy"].append(noisy)
            parts["clean"].append(handle["spectra/clean_zero_loss_counts"][:].astype(np.float32))
            parts["background"].append(handle["spectra/inelastic_background_counts"][:].astype(np.float32))
            parts["expected_total"].append(handle["spectra/expected_total_counts"][:].astype(np.float32))
            parts["peak_center"].append(_repeat_rows(handle["peaks/binding_energy_eV"][:], n_acquisitions))
            parts["peak_width"].append(_repeat_rows(handle["peaks/fwhm_eV"][:], n_acquisitions))
            parts["peak_fraction"].append(_repeat_rows(handle["peaks/area_fraction"][:], n_acquisitions))
            parts["peak_present"].append(_repeat_rows(handle["peaks/present"][:], n_acquisitions))

            case_id = int(handle.attrs["case_id"])
            split = _text(handle.attrs["split"])
            parts["photon_energy"].append(np.full(n_acquisitions, handle.attrs["photon_energy_eV"], np.float32))
            parts["case_id"].append(np.full(n_acquisitions, case_id, np.int32))
            parts["source"].append(np.full(n_acquisitions, path.stem))
            parts["sweep"].append(handle["acquisition_labels/noise_realization_id"][:].astype(np.int16))
            parts["split"].append(np.full(n_acquisitions, split))
            parts["atmospheric_family"].append(
                np.full(n_acquisitions, _text(handle.attrs["atmospheric_family"]))
            )

            for name in (
                "aging_index", "oxygen_to_carbon_ratio", "mass_density_g_cm3",
                "band_gap_eV", "charge_shift_eV",
            ):
                value = np.float32(handle[f"sample_labels/{name}"][()])
                parts[name].append(np.full(n_acquisitions, value, np.float32))
            for name in ("count_scale", "target_maximum_counts", "read_noise_sigma_counts"):
                parts[name].append(handle[f"acquisition_labels/{name}"][:].astype(np.float32))

    result = {name: np.concatenate(values) for name, values in parts.items()}
    result["axis"] = reference_axis
    result["component_names"] = component_names

    found_splits = set(np.unique(result["split"]).tolist())
    if found_splits != REQUIRED_SPLITS:
        raise ValueError(f"Expected splits {sorted(REQUIRED_SPLITS)}, found {sorted(found_splits)}")
    if not all(np.isfinite(result[name]).all() for name in ("noisy", "expected_total", "background")):
        raise ValueError("Spectra contain non-finite values")
    return result
