#!/usr/bin/env python3
"""Prepare and package new weak, rare C 1s cases without touching old spectra.

Run `prepare`, then run_sessa_shards.py with the continuation pattern, then
`package`. The latter appends only new HDF5 rows and leaves IDs 0-11999 intact.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from build_h5_dataset import write_case
from build_vae_dataset import find_spectrum, interpolate
from generate_sessa_batch import LABELS, commands_for_case, draw_case


SEED = 20261002
FIELDS = (["case_id", "aging", "charge_shift", "o_to_c", "density", "egap"]
          + [f"fraction_{name}" for name in LABELS]
          + [f"center_{name}" for name in LABELS]
          + [f"fwhm_{name}" for name in LABELS])
STRATA = ("strong_pi_negative", "strong_pi_central", "strong_pi_positive",
          "moderate_pi_positive", "nonaromatic_multiplet_positive")


def quota(cases: int) -> dict[str, int]:
    if cases % 20:
        raise ValueError("--cases must be a multiple of 20 for exact stratified splits")
    return {name: cases * share // 20 for name, share in zip(STRATA, (4, 4, 4, 4, 4))}


def fits(case: dict, stratum: str) -> bool:
    fractions = case["fractions"]
    pi = fractions[4]
    shift = case["charge_shift"]
    active = int((fractions > 0).sum())
    if stratum.startswith("strong_pi"):
        charge_group = stratum.removeprefix("strong_pi_")
        in_group = ((charge_group == "negative" and shift < -0.5) or
                    (charge_group == "central" and -0.5 <= shift <= 1.5) or
                    (charge_group == "positive" and shift > 1.5))
        return pi >= 0.15 and active >= 4 and in_group
    if stratum == "moderate_pi_positive":
        return 0.03 <= pi < 0.15 and active >= 4 and shift > 1.5
    return pi < 0.03 and active >= 4 and shift > 1.5


def split_names(n: int, rng: np.random.Generator) -> list[str]:
    order = rng.permutation(n)
    names = np.empty(n, dtype=object)
    names[order[: 7 * n // 10]] = "train"
    names[order[7 * n // 10 : 17 * n // 20]] = "validation"
    names[order[17 * n // 20 :]] = "test"
    return names.tolist()


def prepare(root: Path, start_id: int, cases: int, shards: int) -> Path:
    if shards < 1 or shards > cases:
        raise ValueError("--shards must be between 1 and --cases")
    existing = root / "physical_cases.csv"
    with existing.open(newline="", encoding="utf-8") as handle:
        old_ids = {int(row["case_id"]) for row in csv.DictReader(handle)}
    if start_id != max(old_ids) + 1:
        raise ValueError(f"Start ID must be {max(old_ids) + 1}; existing cases remain unchanged")
    if any((root / "spectra" / f"case_{case_id:05d}_fullreg1.spc").exists()
           for case_id in range(start_id, start_id + cases)):
        raise FileExistsError("New case IDs already have spectra; inspect before preparing again")
    manifest = root / f"physical_cases_continuation_{start_id}_{start_id + cases - 1}.csv"
    sessions = [root / f"atmospheric_c1s_continuation_{number:03d}.ses"
                for number in range(shards)]
    if manifest.exists() or any(path.exists() for path in sessions):
        raise FileExistsError("Continuation manifest or session already exists")

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    blocks: list[list[str]] = [[] for _ in sessions]
    for stratum, needed in quota(cases).items():
        splits = split_names(needed, rng)
        for split in splits:
            case = draw_case(rng)
            while not fits(case, stratum):
                case = draw_case(rng)
            case_id = start_id + len(rows)
            row = {"case_id": case_id, "aging": case["aging"],
                   "charge_shift": case["charge_shift"], "o_to_c": case["o_to_c"],
                   "density": case["density"], "egap": case["egap"],
                   "split": split, "continuation_stratum": stratum}
            row.update({f"fraction_{name}": value for name, value in zip(LABELS, case["fractions"])})
            row.update({f"center_{name}": value for name, value in zip(LABELS, case["centers"])})
            row.update({f"fwhm_{name}": value for name, value in zip(LABELS, case["fwhm"])})
            rows.append(row)
            blocks[(case_id - start_id) % shards].extend(commands_for_case(case_id, case, root / "spectra"))
    for path, commands in zip(sessions, blocks):
        path.write_text("\n".join(commands) + "\nQUIT\n", encoding="utf-8")
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + ["split", "continuation_stratum"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows):,} new physical cases ({start_id}-{start_id + cases - 1})")
    print(f"Split: train={sum(x['split']=='train' for x in rows)}, validation={sum(x['split']=='validation' for x in rows)}, test={sum(x['split']=='test' for x in rows)}")
    print(f"Manifest: {manifest}")
    return manifest


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def package(root: Path, h5_root: Path, start_id: int, cases: int) -> None:
    manifest = root / f"physical_cases_continuation_{start_id}_{start_id + cases - 1}.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        new_rows = list(csv.DictReader(handle))
    if len(new_rows) != cases:
        raise ValueError(f"Expected {cases} continuation cases, found {len(new_rows)}")
    h5_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for position, row in enumerate(new_rows, 1):
        case_id = int(row["case_id"])
        destination = h5_root / f"atmospheric_c1s_case_{case_id:05d}.h5"
        if not destination.exists():
            full = np.maximum(interpolate(find_spectrum(root / "spectra", f"case_{case_id:05d}_full")), 0)
            zero = np.minimum(np.maximum(interpolate(find_spectrum(root / "spectra", f"case_{case_id:05d}_zero")), 0), full)
            rng = np.random.default_rng(SEED + case_id * 1_000_003)
            maxima = np.concatenate((10 ** rng.uniform(np.log10(200), np.log10(1000), 4),
                                      10 ** rng.uniform(np.log10(1000), np.log10(10000), 1)))
            rng.shuffle(maxima)
            temporary = destination.with_suffix(".h5.tmp")
            write_case(temporary, row, full, zero, 5, SEED, row["split"],
                       target_maxima=maxima, continuation_stratum=row["continuation_stratum"])
            temporary.replace(destination)
        with __import__("h5py").File(destination, "r") as handle:
            if int(handle.attrs["case_id"]) != case_id or handle.attrs["split"] != row["split"]:
                raise ValueError(f"Existing file has mismatched case or split: {destination}")
            index_rows.append({"case_id": case_id, "file": destination.name,
                               "split": row["split"], "atmospheric_family": handle.attrs["atmospheric_family"],
                               "aging_index": row["aging"], "oxygen_to_carbon_ratio": row["o_to_c"],
                               "number_of_active_peaks": int(handle["peaks/present"][:].sum()),
                               "number_of_acquisitions": len(handle["spectra/noisy_counts"])})
        if position % 100 == 0:
            print(f"Packaged {position:,}/{cases:,}", flush=True)

    index_path = h5_root / "dataset_index.csv"
    with index_path.open(newline="", encoding="utf-8") as handle:
        old_index = list(csv.DictReader(handle))
    indexed = {int(row["case_id"]): row for row in old_index}
    missing_index = []
    for row in index_rows:
        case_id = int(row["case_id"])
        if case_id in indexed:
            if indexed[case_id]["file"] != row["file"] or indexed[case_id]["split"] != row["split"]:
                raise ValueError(f"Conflicting indexed case {case_id}")
        else:
            missing_index.append(row)
    if missing_index:
        atomic_csv(index_path, old_index + missing_index, list(old_index[0]))

    physical_path = root / "physical_cases.csv"
    with physical_path.open(newline="", encoding="utf-8") as handle:
        old_physical = list(csv.DictReader(handle))
    physical_ids = {int(row["case_id"]) for row in old_physical}
    missing_physical = [{field: row[field] for field in FIELDS}
                        for row in new_rows if int(row["case_id"]) not in physical_ids]
    if missing_physical:
        atomic_csv(physical_path, old_physical + missing_physical, FIELDS)
    total = len(old_index) + len(missing_index)
    print(f"Dataset now contains {total:,} physical cases and {5 * total:,} acquisitions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "package"))
    parser.add_argument("--input", type=Path, default=Path("sessa_generated"))
    parser.add_argument("--h5-root", type=Path, default=Path("atmospheric_c1s_h5"))
    parser.add_argument("--start-id", type=int, default=12000)
    parser.add_argument("--cases", type=int, default=2000)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.input.resolve(), args.start_id, args.cases, args.shards)
    else:
        package(args.input.resolve(), args.h5_root.resolve(), args.start_id, args.cases)


if __name__ == "__main__":
    main()
