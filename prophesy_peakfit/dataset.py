"""Read the self-contained PROPHESY peak-fitting ZIP archive."""

from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


REGIONS = ("C1s", "S2p", "O1s")


def load_region(archive_path: Path, region: str) -> dict[str, np.ndarray]:
    if region not in REGIONS:
        raise ValueError(f"Region must be one of {REGIONS}, got {region!r}")
    with ZipFile(archive_path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        if metadata.get("schema") != "prophesy_peakfit_v1":
            raise ValueError("Unsupported PROPHESY archive schema")
        with np.load(io.BytesIO(archive.read(f"{region}.npz")), allow_pickle=False) as packed:
            result = {name: packed[name] for name in packed.files}

    n, width = result["noisy_counts"].shape
    peaks = result["peak_center_eV"].shape[1]
    if result["axis_eV"].shape != (width,):
        raise ValueError("Energy axis and spectra have different lengths")
    for name in ("clean_counts", "background_counts"):
        if result[name].shape != (n, width):
            raise ValueError(f"Bad {name} shape")
    for name in ("peak_sigma_eV", "peak_fraction", "peak_area_counts_eV"):
        if result[name].shape != (n, peaks):
            raise ValueError(f"Bad {name} shape")
    if set(result["split"].tolist()) != {"train", "validation", "test"}:
        raise ValueError("Missing a train, validation, or test split")
    if not np.allclose(result["peak_fraction"].sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Peak fractions must sum to one")
    for name in ("noisy_counts", "clean_counts", "peak_center_eV", "peak_sigma_eV", "peak_fraction"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Non-finite {name}")
    return result
