#!/usr/bin/env python3
"""Generate a SESSA session containing atmospheric C 1s simulations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


LABELS = ("CC_CH", "C_O", "C_eq_O", "O_C_eq_O", "pi_pi_star")
BASE_CENTERS = np.array([284.8, 286.25, 287.75, 289.05, 291.2])
AL_KA_ENERGY = 1486.6


def draw_case(rng: np.random.Generator) -> dict:
    aging = float(rng.beta(2.0, 2.0))
    alpha = np.array(
        [3.5 * (1.0 - aging) + 0.35, 0.7 + 2.2 * aging,
         0.4 + 1.7 * aging, 0.2 + 1.8 * aging,
         0.15 + 0.45 * (1.0 - aging)]
    )
    active = np.array([True, True, True, True, rng.random() < 0.30])
    for index in (1, 2, 3):
        if rng.random() < 0.12:
            active[index] = False
    if active.sum() < 2:
        active[1] = True

    fractions = np.zeros(5)
    fractions[active] = rng.dirichlet(alpha[active])
    sessa_fractions = np.maximum(fractions, 1.0e-6)
    sessa_fractions /= sessa_fractions.sum()

    charge_shift = float(rng.uniform(-1.5, 2.5))
    centers = BASE_CENTERS + charge_shift + rng.normal(0.0, [0.10, 0.16, 0.18, 0.18, 0.25])
    common_fwhm = float(rng.uniform(0.90, 1.60))
    fwhm = np.clip(common_fwhm * rng.normal(1.0, 0.10, 5), 0.70, 2.00)
    sigma = fwhm / 2.354820045

    # Approximate O:C used by SESSA for transport/background, not as a fitted label.
    o_to_c = float(np.clip(
        0.03 + fractions[1] + fractions[2] + 2.0 * fractions[3], 0.03, 1.20
    ))
    density = float(rng.uniform(1.10, 1.75))
    egap = 0.0 if fractions[4] > 0.08 else float(rng.uniform(2.0, 7.0))
    return {
        "aging": aging,
        "fractions": fractions,
        "sessa_fractions": sessa_fractions,
        "centers": centers,
        "fwhm": fwhm,
        "sigma": sigma,
        "charge_shift": charge_shift,
        "o_to_c": o_to_c,
        "density": density,
        "egap": egap,
    }


def commands_for_case(case_id: int, case: dict, spectra_dir: Path) -> list[str]:
    f = case["sessa_fractions"]
    material = (
        f"/C[CC_CH]{f[0]:.9f}/C[C_O]{f[1]:.9f}/"
        f"C[C_eq_O]{f[2]:.9f}/C[O_C_eq_O]{f[3]:.9f}/"
        f"C[pi_pi_star]{f[4]:.9f}/O{case['o_to_c']:.9f}/"
    )
    prefix = spectra_dir / f"case_{case_id:05d}"
    lines = [
        "\\PROJECT RESET",
        "\\SOURCE SET ALKA",
        # SESSA 2.3.0 accepts the C 1s region reliably on its native kinetic scale.
        "\\PREFERENCES SET ENERGY_SCALE KINETIC",
        "\\PREFERENCES SET DENSITY_SCALE MASS",
        "\\PREFERENCES SET NDIIMFP 1",
        "\\SPECTROMETER SET RANGE 1186.6:1206.6 REGION 1",
        f"\\SAMPLE SET MATERIAL {material} LAYER 1",
        f"\\SAMPLE SET DENSITY {case['density']:.8f} LAYER 1",
        f"\\SAMPLE SET EGAP {case['egap']:.8f} LAYER 1",
    ]
    for peak, (center, sigma) in enumerate(zip(case["centers"], case["sigma"]), 1):
        kinetic_center = AL_KA_ENERGY - center
        lines.extend([
            f"\\SAMPLE PEAK SET TYPE GAUSS PEAK {peak} SUBPEAK 1",
            f"\\SAMPLE PEAK SET POSITION {kinetic_center:.8f} PEAK {peak} SUBPEAK 1",
            f"\\SAMPLE PEAK SET WIDTH {sigma:.8f} PEAK {peak} SUBPEAK 1",
            # SESSA uses relative subpeak height here. Detector-count heights are
            # recovered from its zero-loss output when packaging the dataset.
            f"\\SAMPLE PEAK SET HEIGHT 1.0 PEAK {peak} SUBPEAK 1",
        ])
    lines.extend([
        "\\MODEL SET SLA false",
        "\\MODEL SET TA false",
        "\\MODEL SET SE true",
        "\\MODEL SET QEAPLUS true",
        "\\MODEL SET CONVERGENCE 1.0e-2",
        "\\MODEL AUTO NCOL REGION 1",
        "\\MODEL AUTO NTRAJ REGION 1",
        "\\MODEL SIMULATE",
        f'\\MODEL SAVE SPECTRA "{prefix}_full"',
        "\\MODEL SET NCOL 0 REGION 1",
        "\\MODEL AUTO NTRAJ REGION 1",
        "\\MODEL SIMULATE",
        f'\\MODEL SAVE SPECTRA "{prefix}_zero"',
    ])
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=12_000)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", type=Path, default=Path("sessa_generated"))
    args = parser.parse_args()

    output = args.output.resolve()
    spectra_dir = output / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "physical_cases.csv"
    rng = np.random.default_rng(args.seed)

    if args.shards < 1 or args.shards > args.cases:
        parser.error("--shards must be between 1 and --cases")
    if args.shards == 1:
        session_paths = [output / "atmospheric_c1s_batch.ses"]
    else:
        session_paths = [
            output / f"atmospheric_c1s_batch_{index:03d}.ses"
            for index in range(args.shards)
        ]

    fields = ["case_id", "aging", "charge_shift", "o_to_c", "density", "egap"]
    fields += [f"fraction_{x}" for x in LABELS]
    fields += [f"center_{x}" for x in LABELS]
    fields += [f"fwhm_{x}" for x in LABELS]

    sessions = [path.open("w", encoding="utf-8", newline="\n") for path in session_paths]
    try:
        manifest = manifest_path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(manifest, fieldnames=fields)
        writer.writeheader()
        for case_id in range(args.cases):
            case = draw_case(rng)
            session = sessions[case_id % args.shards]
            session.write("\n".join(commands_for_case(case_id, case, spectra_dir)) + "\n")
            row = {
                "case_id": case_id,
                "aging": case["aging"],
                "charge_shift": case["charge_shift"],
                "o_to_c": case["o_to_c"],
                "density": case["density"],
                "egap": case["egap"],
            }
            row.update({f"fraction_{x}": v for x, v in zip(LABELS, case["fractions"])})
            row.update({f"center_{x}": v for x, v in zip(LABELS, case["centers"])})
            row.update({f"fwhm_{x}": v for x, v in zip(LABELS, case["fwhm"])})
            writer.writerow(row)
        manifest.close()
        for session in sessions:
            session.write("QUIT\n")
    finally:
        for session in sessions:
            session.close()

    for path in session_paths:
        print(path)
    print(manifest_path)
    print(f"Generated {args.cases} physical cases; load the .ses file in SESSA.")


if __name__ == "__main__":
    main()
