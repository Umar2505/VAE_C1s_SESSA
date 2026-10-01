#!/usr/bin/env python3
"""Package SESSA outputs as one self-contained HDF5 file per physical case."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np

from build_vae_dataset import AXIS, LABELS, find_spectrum, interpolate


DISPLAY_NAMES = np.asarray(["C-C/C-H", "C-O/C-N", "C=O/O-C-O", "O-C=O/COO-", "pi-pi*"], dtype=object)


def atmospheric_family(aging: float, fractions: np.ndarray) -> str:
    if fractions[4] >= 0.03:
        return "soot_or_aromatic"
    if aging < 0.33:
        return "fresh_carbonaceous"
    if aging > 0.66:
        return "aged_oxygenated_organic"
    return "mixed_organic_aerosol"


def write_case(
    destination: Path,
    row: dict[str, str],
    full: np.ndarray,
    zero: np.ndarray,
    replicates: int,
    seed: int,
    split: str,
) -> dict[str, object]:
    case_id = int(row["case_id"])
    centers = np.asarray([float(row[f"center_{name}"]) for name in LABELS], dtype=np.float32)
    widths = np.asarray([float(row[f"fwhm_{name}"]) for name in LABELS], dtype=np.float32)
    fractions = np.asarray([float(row[f"fraction_{name}"]) for name in LABELS], dtype=np.float32)
    present = fractions > 0.0
    aging = float(row["aging"])
    family = atmospheric_family(aging, fractions)
    rng = np.random.default_rng(seed + case_id * 1_000_003)

    noisy = np.empty((replicates, AXIS.size), dtype=np.float32)
    expected = np.empty_like(noisy)
    clean = np.empty_like(noisy)
    background = np.empty_like(noisy)
    count_scale = np.empty(replicates, dtype=np.float32)
    target_maximum = np.empty(replicates, dtype=np.float32)
    read_noise_sigma = np.empty(replicates, dtype=np.float32)

    for sweep in range(replicates):
        target_max = 10.0 ** rng.uniform(np.log10(200.0), np.log10(150_000.0))
        scale = target_max / max(float(full.max()), 1.0e-30)
        expectation = np.maximum(full * scale, 0.0)
        read_sigma = float(rng.uniform(0.0, 1.5))
        counts = rng.poisson(expectation).astype(np.float32)
        counts += rng.normal(0.0, read_sigma, counts.size).astype(np.float32)
        noisy[sweep] = np.maximum(counts, 0.0)
        expected[sweep] = expectation
        clean[sweep] = zero * scale
        background[sweep] = np.maximum(full - zero, 0.0) * scale
        count_scale[sweep] = scale
        target_maximum[sweep] = target_max
        read_noise_sigma[sweep] = read_sigma

    destination.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(destination, "w") as handle:
        handle.attrs.update({
            "schema_name": "atmospheric_xps_c1s_vae",
            "schema_version": "1.0.0",
            "case_id": case_id,
            "split": split,
            "atmospheric_family": family,
            "simulation_engine": "NIST SESSA",
            "simulation_version": "2.3.0",
            "excitation_source": "Al K-alpha",
            "photon_energy_eV": 1486.6,
            "energy_scale": "binding_energy",
            "spectral_region": "C 1s",
            "sample_morphology": "homogeneous_planar",
            "noise_model": "Poisson counting plus Gaussian read noise",
            "number_of_components": 5,
            "number_of_acquisitions": replicates,
            "random_seed": seed + case_id * 1_000_003,
        })

        energy = handle.create_dataset("energy_eV", data=AXIS.astype(np.float32))
        energy.attrs["units"] = "eV"
        energy.attrs["scale"] = "binding energy"
        energy.attrs["direction"] = "increasing"

        spectra = handle.create_group("spectra")
        compression = {"compression": "gzip", "compression_opts": 4, "shuffle": True}
        spectra.create_dataset("noisy_counts", data=noisy, **compression)
        spectra.create_dataset("expected_total_counts", data=expected, **compression)
        spectra.create_dataset("clean_zero_loss_counts", data=clean, **compression)
        spectra.create_dataset("inelastic_background_counts", data=background, **compression)
        spectra.create_dataset("raw_sessa_full", data=full.astype(np.float32), **compression)
        spectra.create_dataset("raw_sessa_zero_loss", data=zero.astype(np.float32), **compression)
        spectra["noisy_counts"].attrs["role"] = "VAE input"
        spectra["expected_total_counts"].attrs["role"] = "denoising reconstruction target"
        spectra["clean_zero_loss_counts"].attrs["role"] = "photoelectron peak signal"
        spectra["inelastic_background_counts"].attrs["role"] = "SESSA transport background"

        peaks = handle.create_group("peaks")
        peaks.create_dataset("machine_names", data=np.asarray(LABELS, dtype=object), dtype=string_dtype)
        peaks.create_dataset("display_names", data=DISPLAY_NAMES, dtype=string_dtype)
        peaks.create_dataset("binding_energy_eV", data=centers)
        peaks.create_dataset("fwhm_eV", data=widths)
        peaks.create_dataset("area_fraction", data=fractions)
        peaks.create_dataset("present", data=present)
        peaks.create_dataset("shape", data=np.asarray(["Gaussian"] * 5, dtype=object), dtype=string_dtype)

        sample = handle.create_group("sample_labels")
        sample.create_dataset("aging_index", data=np.float32(aging))
        sample.create_dataset("charge_shift_eV", data=np.float32(row["charge_shift"]))
        sample.create_dataset("oxygen_to_carbon_ratio", data=np.float32(row["o_to_c"]))
        sample.create_dataset("mass_density_g_cm3", data=np.float32(row["density"]))
        sample.create_dataset("band_gap_eV", data=np.float32(row["egap"]))

        acquisition = handle.create_group("acquisition_labels")
        acquisition.create_dataset("noise_realization_id", data=np.arange(replicates, dtype=np.int16))
        acquisition.create_dataset("count_scale", data=count_scale)
        acquisition.create_dataset("target_maximum_counts", data=target_maximum)
        acquisition.create_dataset("read_noise_sigma_counts", data=read_noise_sigma)

    return {
        "case_id": case_id,
        "file": destination.name,
        "split": split,
        "atmospheric_family": family,
        "aging_index": aging,
        "oxygen_to_carbon_ratio": float(row["o_to_c"]),
        "number_of_active_peaks": int(present.sum()),
        "number_of_acquisitions": replicates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("sessa_generated"))
    parser.add_argument("--output", type=Path, default=Path("atmospheric_c1s_h5"))
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    root = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (root / "physical_cases.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    split_rng = np.random.default_rng(args.seed)
    split_draws = split_rng.random(len(rows))
    index_rows = []
    for position, row in enumerate(rows):
        case_id = int(row["case_id"])
        draw = split_draws[position]
        split = "train" if draw < 0.70 else "validation" if draw < 0.85 else "test"
        spectra_dir = root / "spectra"
        full = np.maximum(interpolate(find_spectrum(spectra_dir, f"case_{case_id:05d}_full")), 0.0)
        zero = np.minimum(
            np.maximum(interpolate(find_spectrum(spectra_dir, f"case_{case_id:05d}_zero")), 0.0),
            full,
        )
        destination = output / f"atmospheric_c1s_case_{case_id:05d}.h5"
        index_rows.append(write_case(destination, row, full, zero, args.replicates, args.seed, split))
        if (position + 1) % 100 == 0:
            print(f"Packaged {position + 1:,}/{len(rows):,} HDF5 cases", flush=True)

    index_path = output / "dataset_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    print(output)
    print(f"Saved {len(rows):,} HDF5 files containing {len(rows) * args.replicates:,} acquisitions.")


if __name__ == "__main__":
    main()
