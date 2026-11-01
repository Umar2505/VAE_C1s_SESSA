#!/usr/bin/env python3
"""Generate labelled C 1s spectra with SESSA 2.3 and three Poisson count levels.

One physical SESSA simulation produces one clean spectrum. The script then
produces exactly three noisy observations of it, with expected peak counts of
50, 500, and 5000. SESSA's plotted data are exported with PLOT SAVE DATA;
the region .spc export is omitted because this installation has produced an
invalid .spc file in binding-energy mode.

Example:
    python generate_sessa_training_dataset.py --cases 20
    python generate_sessa_training_dataset.py --cases 10000 \
        --sessa-bin ~/Applications/Sessa-2.3.0-Linux/bin/sessa

Requires numpy, scipy, SESSA 2.3, and xvfb-run (or --use-display).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d


_SESSA_ON_MINT = Path.home() / "Applications/Sessa-2.3.0-Linux/bin/sessa"
_SESSA_BESIDE_PROJECT = Path(__file__).resolve().parent.parent / "Sessa-2.3.0-Linux/bin/sessa"
SESSA_BIN = (Path(os.environ["SESSA_BIN"]) if os.environ.get("SESSA_BIN") else
             _SESSA_ON_MINT if _SESSA_ON_MINT.is_file() else
             _SESSA_BESIDE_PROJECT)
COMPONENTS = ("CC", "CO", "CARBONYL", "OCO", "COO")
BASE_CENTERS_EV = {"CC": 285.0, "CO": 286.5, "CARBONYL": 287.7,
                   "OCO": 288.1, "COO": 289.1}
NOISE_PEAK_COUNTS = (50, 500, 5000)
ENERGY_GRID_EV = np.linspace(280.0, 300.0, 401)
AVOGADRO = 6.02214076e23
MASS = {"C": 12.011, "H": 1.008, "O": 15.999}


@dataclass(frozen=True)
class Template:
    name: str
    carbon: dict[str, int]
    hydrogen: int
    oxygen: int
    mass_density_g_cm3: float

    @property
    def molar_mass(self) -> float:
        return (sum(self.carbon.values()) * MASS["C"]
                + self.hydrogen * MASS["H"] + self.oxygen * MASS["O"])


# Repeat-unit compositions are deliberate carbon-state surrogates. SESSA uses
# the state IDs to distinguish emitters; it does not infer molecular bonding.
TEMPLATES = (
    Template("hydrocarbon", {"CC": 2}, 4, 0, 0.93),
    Template("pva_like", {"CC": 1, "CO": 1}, 4, 1, 1.20),
    Template("cellulose_like", {"CO": 5, "OCO": 1}, 10, 5, 1.50),
    Template("polyacrylic_acid_like", {"CC": 2, "COO": 1}, 4, 2, 1.22),
    Template("pmma_like", {"CC": 3, "CO": 1, "COO": 1}, 8, 2, 1.18),
    Template("methyl_ketone_like", {"CC": 3, "CARBONYL": 1}, 6, 1, 1.05),
)


def make_case(rng: np.random.Generator, case_index: int) -> dict:
    n_templates = int(rng.choice((1, 2, 3), p=(0.40, 0.45, 0.15)))
    chosen = rng.choice(len(TEMPLATES), size=n_templates, replace=False)
    copies = [int(x) for x in rng.integers(1, 5, size=n_templates)]
    carbon = {name: 0 for name in COMPONENTS}
    hydrogen = oxygen = 0
    total_mass = total_molar_volume = 0.0
    ingredients = []
    for template_index, count in zip(chosen, copies):
        template = TEMPLATES[int(template_index)]
        ingredients.append({"template": template.name, "copies": count})
        for state, number in template.carbon.items():
            carbon[state] += count * number
        hydrogen += count * template.hydrogen
        oxygen += count * template.oxygen
        total_mass += count * template.molar_mass
        total_molar_volume += count * template.molar_mass / template.mass_density_g_cm3

    mass_density = total_mass / total_molar_volume * float(rng.uniform(0.95, 1.05))
    n_atoms = sum(carbon.values()) + hydrogen + oxygen
    n_valence = 4 * sum(carbon.values()) + hydrogen + 6 * oxygen
    atomic_density = mass_density / total_mass * AVOGADRO * n_atoms
    material = "".join(f"/C[{state}]{carbon[state]}" for state in COMPONENTS
                       if carbon[state] > 0)
    if hydrogen:
        material += f"/H{hydrogen}"
    if oxygen:
        material += f"/O{oxygen}"
    material += "/"

    global_shift = float(rng.uniform(-0.20, 0.20))
    peaks = []
    for state in COMPONENTS:
        if carbon[state] == 0:
            continue
        center = BASE_CENTERS_EV[state] + global_shift + float(rng.uniform(-0.18, 0.18))
        shape = str(rng.choice(("gaussian", "pseudo_voigt", "lorentzian"),
                               p=(0.35, 0.55, 0.10)))
        peaks.append({
            "state": state,
            "center_ev": round(center, 5),
            "shape": shape,
            # Gaussian width is sigma in SESSA; Lorentz width is SESSA's
            # Lorentzian width parameter. The pseudo-Voigt is a height mix.
            "gaussian_sigma_ev": round(float(rng.uniform(0.15, 0.36)), 5),
            "lorentz_width_ev": round(float(rng.uniform(0.10, 0.30)), 5),
            "lorentz_height_fraction": round(float(rng.uniform(0.15, 0.75)), 5),
            "carbon_atoms": carbon[state],
            "carbon_fraction": carbon[state] / sum(carbon.values()),
        })

    return {
        "case_id": f"case_{case_index:04d}",
        "ingredients": ingredients,
        "material": material,
        "carbon_counts": carbon,
        "hydrogen": hydrogen,
        "oxygen": oxygen,
        "atoms_per_formula": n_atoms,
        "valence_electrons_per_formula": n_valence,
        "mass_density_g_cm3": mass_density,
        "atomic_density_atoms_cm3": atomic_density,
        "organic_thickness_angstrom": round(float(rng.uniform(20.0, 120.0)), 3),
        "band_gap_ev": round(float(rng.uniform(4.0, 7.0)), 3),
        "analyzer_fwhm_ev": float(rng.choice((0.5, 1.0))),
        "source": "Mg K-alpha, 1253.6 eV; SESSA includes its source spread",
        "global_shift_ev": global_shift,
        "peaks": peaks,
    }


def command_file(case: dict, case_dir: Path, rng_seed: int) -> str:
    lines = [
        "PROJECT RESET",
        "PREFERENCES SET ENERGY_SCALE BINDING",
        "PREFERENCES SET DENSITY_SCALE ATOMIC",
        "SOURCE SET MGKA",
        "SPECTROMETER SET RANGE 280:300 REGION 1",
        "SAMPLE MORPHOLOGY SET PLANAR",
        "SAMPLE SET MATERIAL /Au/ LAYER 1",
        f"SAMPLE ADD LAYER {case['material']} THICKNESS "
        f"{case['organic_thickness_angstrom']:.3f} ABOVE 0",
        f"SAMPLE SET DENSITY {case['atomic_density_atoms_cm3']:.8e} LAYER 1",
        f"SAMPLE SET EGAP {case['band_gap_ev']:.3f} LAYER 1",
        f"SAMPLE SET NVALENCE {case['valence_electrons_per_formula']} LAYER 1",
        f"SAMPLE SET NATOMS {case['atoms_per_formula']} LAYER 1",
        "SAMPLE PEAK DELETE XPS",
    ]
    for peak_number, peak in enumerate(case["peaks"], start=1):
        center = peak["center_ev"]
        state = peak["state"]
        lines.append(f"SAMPLE PEAK ADD XPSPEAK ENERGY {center:.5f} "
                     f"SUBSHELLID 1s ELEMENT C CHEMSTATID {state}")
        if peak["shape"] == "gaussian":
            lines.extend((
                f"SAMPLE PEAK SET TYPE GAUSS PEAK {peak_number} SUBPEAK 1",
                f"SAMPLE PEAK SET WIDTH {peak['gaussian_sigma_ev']:.5f} "
                f"PEAK {peak_number} SUBPEAK 1",
                f"SAMPLE PEAK SET HEIGHT 1 PEAK {peak_number} SUBPEAK 1",
            ))
        elif peak["shape"] == "lorentzian":
            lines.extend((
                f"SAMPLE PEAK SET TYPE LORENTZ PEAK {peak_number} SUBPEAK 1",
                f"SAMPLE PEAK SET WIDTH {peak['lorentz_width_ev']:.5f} "
                f"PEAK {peak_number} SUBPEAK 1",
                f"SAMPLE PEAK SET HEIGHT 1 PEAK {peak_number} SUBPEAK 1",
            ))
        else:
            fraction = peak["lorentz_height_fraction"]
            lines.extend((
                f"SAMPLE PEAK SET TYPE GAUSS PEAK {peak_number} SUBPEAK 1",
                f"SAMPLE PEAK SET WIDTH {peak['gaussian_sigma_ev']:.5f} "
                f"PEAK {peak_number} SUBPEAK 1",
                f"SAMPLE PEAK SET HEIGHT {1-fraction:.5f} "
                f"PEAK {peak_number} SUBPEAK 1",
                f"SAMPLE PEAK ADD SUBPEAK PEAK {peak_number} ENERGY {center:.5f}",
                f"SAMPLE PEAK SET TYPE LORENTZ PEAK {peak_number} SUBPEAK 2",
                f"SAMPLE PEAK SET WIDTH {peak['lorentz_width_ev']:.5f} "
                f"PEAK {peak_number} SUBPEAK 2",
                f"SAMPLE PEAK SET HEIGHT {fraction:.5f} "
                f"PEAK {peak_number} SUBPEAK 2",
            ))

    lines.extend((
        "MODEL SET NCHAN 2048",
        "MODEL SET CONVERGENCE 0.001",
        "MODEL AUTO NCOL REGION 1",
        "MODEL AUTO NTRAJ REGION 1",
        f"MODEL SET RNG_SEED {rng_seed}",
        "MODEL SIMULATE",
        "MODEL PLOT REGION 1",
        f'PLOT SAVE DATA "{case_dir / "total_plot.dat"}"',
    ))
    for peak_number, peak in enumerate(case["peaks"], start=1):
        lines.extend((
            f"SAMPLE PEAK PLOT SPEC PEAK {peak_number}",
            f'PLOT SAVE DATA "{case_dir / ("component_" + peak["state"] + ".dat")}"',
        ))
    return "\n".join("\\" + line for line in lines) + "\n"


def run_sessa(case_dir: Path, expected_plots: list[Path], sessa_bin: Path,
              use_display: bool, timeout_sec: int, max_attempts: int) -> None:
    session = case_dir / "run.ses"
    launch = [str(sessa_bin), "-c"]
    if not use_display:
        if shutil.which("xvfb-run") is None:
            raise RuntimeError("xvfb-run is missing; install it or pass --use-display")
        launch = ["xvfb-run", "-a", *launch]
    env = os.environ.copy()
    lib_dir = sessa_bin.parent.parent / "lib"
    env["LD_LIBRARY_PATH"] = str(lib_dir) + (
        ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    log_path = case_dir / "sessa.log"
    for attempt in range(1, max_attempts + 1):
        # A previous crash may have left only some plots. Never combine plots
        # from two simulation attempts into one training case.
        for path in expected_plots:
            path.unlink(missing_ok=True)
        try:
            result = subprocess.run(
                launch,
                input=f'PROJECT LOAD SESSION "{session}"\nquit\n',
                text=True,
                cwd=sessa_bin.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_sec,
                check=False,
            )
            message = result.stdout or ""
            successful = result.returncode == 0 and all(
                path.exists() for path in expected_plots
            )
            status = f"exit {result.returncode}"
            if result.returncode == 0 and not successful:
                status += ", missing plot exports"
        except subprocess.TimeoutExpired as exc:
            successful = False
            status = f"timed out after {timeout_sec} seconds"
            message = str(exc)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n=== Attempt {attempt}/{max_attempts}: {status} ===\n")
            handle.write(message)
            handle.write("\n")
        if successful:
            log_path.unlink(missing_ok=True)
            return
        if attempt < max_attempts:
            time.sleep(2)
    raise RuntimeError(
        f"SESSA failed for {case_dir.name} after {max_attempts} attempts; "
        f"see {log_path}"
    )


def read_plot(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise RuntimeError(f"SESSA did not export {path}; inspect the case's sessa.log")
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            rows.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue
    if len(rows) < 100:
        raise RuntimeError(f"{path} has too few numeric rows for a spectrum")
    data = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(data).all():
        raise RuntimeError(f"{path} has non-finite energy or intensity values")
    order = np.argsort(data[:, 0])
    energy, intensity = data[order, 0], data[order, 1]
    # SESSA sometimes prints duplicate rounded energy values. Average them.
    unique_energy, inverse = np.unique(energy, return_inverse=True)
    if len(unique_energy) != len(energy):
        sums = np.zeros(len(unique_energy))
        counts = np.zeros(len(unique_energy))
        np.add.at(sums, inverse, intensity)
        np.add.at(counts, inverse, 1)
        energy, intensity = unique_energy, sums / counts
    if energy[0] > 280.1 or energy[-1] < 299.9:
        raise RuntimeError(
            f"{path} spans {energy[0]:.2f}–{energy[-1]:.2f} eV, "
            "not the requested 280–300 eV. Do not add noise to this export."
        )
    if intensity.max() <= 0 or np.ptp(intensity) <= 0:
        raise RuntimeError(
            f"{path} has no varying positive spectrum. The earlier .spc export "
            "had this problem; inspect the plot export and SESSA log."
        )
    if intensity.min() < -0.01 * intensity.max():
        raise RuntimeError(f"{path} contains substantially negative intensities")
    return energy, np.clip(intensity, 0.0, None)


def broaden_and_resample(energy: np.ndarray, intensity: np.ndarray,
                         analyzer_fwhm_ev: float) -> np.ndarray:
    step = float(np.median(np.diff(energy)))
    sigma_ev = analyzer_fwhm_ev / 2.354820045
    blurred = gaussian_filter1d(intensity, sigma_ev / step, mode="nearest")
    return np.interp(ENERGY_GRID_EV, energy, blurred)


def save_csv(path: Path, intensity: np.ndarray, header: str) -> None:
    np.savetxt(path, np.column_stack((ENERGY_GRID_EV, intensity)),
               delimiter=",", header="binding_energy_eV," + header, comments="")


def prepare_and_simulate(case: dict, output: Path, sessa_bin: Path,
                         use_display: bool, timeout_sec: int,
                         max_attempts: int, rng_seed: int
                         ) -> tuple[Path, Path, list[Path]]:
    case_dir = output / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    case_config = case_dir / "metadata.json"
    if case_config.exists():
        old = json.loads(case_config.read_text(encoding="utf-8"))
        if old != case:
            raise RuntimeError(
                f"Existing {case_config} differs from this run. Choose a new "
                "--output directory or use the original seed and case count."
            )
    else:
        case_config.write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")
    session = case_dir / "run.ses"
    session.write_text(command_file(case, case_dir, rng_seed), encoding="utf-8")
    expected_plots = [case_dir / "total_plot.dat"] + [
        case_dir / f"component_{peak['state']}.dat" for peak in case["peaks"]
    ]
    if not all(path.exists() for path in expected_plots):
        run_sessa(case_dir, expected_plots, sessa_bin, use_display,
                  timeout_sec, max_attempts)
    session.unlink(missing_ok=True)
    return case_dir, case_config, expected_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=10_000,
                        help="number of distinct SESSA simulations (default: 10000)")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "datasets" / "sessa_c1s_v1")
    parser.add_argument("--sessa-bin", type=Path, default=SESSA_BIN)
    parser.add_argument("--use-display", action="store_true",
                        help="run on your existing X display instead of xvfb-run")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="retry transient SESSA crashes this many times (default: 3)")
    parser.add_argument("--workers", type=int, default=2,
                        help="simultaneous SESSA processes (default: 2)")
    args = parser.parse_args()
    if args.cases < 1:
        parser.error("--cases must be positive")
    if args.max_attempts < 1 or args.workers < 1:
        parser.error("--max-attempts and --workers must be positive")
    sessa_bin = args.sessa_bin.expanduser().resolve()
    if not sessa_bin.is_file():
        parser.error(f"SESSA executable not found: {sessa_bin}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    cases = [make_case(rng, i) for i in range(args.cases)]
    # Draw one split per case from a separate RNG. Earlier cases keep their
    # splits when --cases increases, so a pilot run can be resumed in place.
    split_rng = np.random.default_rng(args.seed + 19)
    split_names = ["train" if value < 0.8 else
                   "validation" if value < 0.9 else "test"
                   for value in split_rng.random(args.cases)]
    for index, case in enumerate(cases):
        case["split"] = split_names[index]

    clean_all, components_all, noisy_all = [], [], []
    centers_all, presence_all, areas_all, shapes_all = [], [], [], []
    manifest_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        next_index = 0

        def queue_next() -> None:
            nonlocal next_index
            if next_index < args.cases:
                index = next_index
                pending[index] = pool.submit(
                    prepare_and_simulate, cases[index], output, sessa_bin,
                    args.use_display, args.timeout_sec, args.max_attempts,
                    args.seed + index + 1,
                )
                next_index += 1

        for _ in range(min(args.workers, args.cases)):
            queue_next()
        for index, case in enumerate(cases):
            case_dir, case_config, expected_plots = pending.pop(index).result()
            queue_next()

            energy, intensity = read_plot(expected_plots[0])
            clean = broaden_and_resample(energy, intensity, case["analyzer_fwhm_ev"])
            raw_components = {}
            for peak, component_file in zip(case["peaks"], expected_plots[1:]):
                component_energy, component_intensity = read_plot(component_file)
                raw_components[peak["state"]] = broaden_and_resample(
                    component_energy, component_intensity, case["analyzer_fwhm_ev"]
                )
            component_sum = np.sum(list(raw_components.values()), axis=0)
            relative_mismatch = float(np.max(np.abs(component_sum - clean)) / clean.max())
            if relative_mismatch > 0.15:
                raise RuntimeError(
                    f"{case_dir.name}: individual peak spectra do not sum to the full "
                    f"spectrum (relative mismatch {relative_mismatch:.1%}). Inspect the "
                    "SESSA plot exports before training."
                )
            scale = float(clean.max())
            clean /= scale
            component_array = np.zeros((len(COMPONENTS), len(ENERGY_GRID_EV)))
            for component_index, state in enumerate(COMPONENTS):
                if state in raw_components:
                    component_array[component_index] = raw_components[state] / scale

            noise_rng = np.random.default_rng(args.seed + 1000003 * (index + 1))
            noisy = np.stack([noise_rng.poisson(clean * level) for level in NOISE_PEAK_COUNTS])
            save_csv(case_dir / "clean.csv", clean, "relative_intensity")
            for level, counts in zip(NOISE_PEAK_COUNTS, noisy):
                filename = f"poisson_{level}.csv"
                save_csv(case_dir / filename, counts, "counts")
                manifest_rows.append({
                    "case_id": case["case_id"], "split": case["split"],
                    "peak_counts": level,
                    "file": str((case_dir / filename).relative_to(output)),
                    "metadata": str(case_config.relative_to(output)),
                })

            centers = np.full(len(COMPONENTS), np.nan)
            presence = np.zeros(len(COMPONENTS), dtype=np.uint8)
            shape_codes = np.full(len(COMPONENTS), -1, dtype=np.int8)
            shape_map = {"gaussian": 0, "lorentzian": 1, "pseudo_voigt": 2}
            for peak in case["peaks"]:
                j = COMPONENTS.index(peak["state"])
                centers[j] = peak["center_ev"]
                presence[j] = 1
                shape_codes[j] = shape_map[peak["shape"]]
            areas = trapezoid(component_array, ENERGY_GRID_EV, axis=1)
            clean_all.append(clean)
            components_all.append(component_array)
            noisy_all.append(noisy)
            centers_all.append(centers)
            presence_all.append(presence)
            areas_all.append(areas)
            shapes_all.append(shape_codes)
            print(f"{index + 1}/{args.cases}: {case['case_id']} "
                  f"({len(case['peaks'])} peaks, {case['analyzer_fwhm_ev']} eV analyzer)",
                  flush=True)

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "split", "peak_counts", "file", "metadata"))
        writer.writeheader()
        writer.writerows(manifest_rows)
    np.savez_compressed(
        output / "dataset.npz",
        energy_ev=ENERGY_GRID_EV,
        components=np.asarray(COMPONENTS),
        case_id=np.asarray([case["case_id"] for case in cases]),
        split=np.asarray(split_names),
        peak_counts=np.asarray(NOISE_PEAK_COUNTS),
        clean_relative=np.stack(clean_all),
        component_relative=np.stack(components_all),
        poisson_counts=np.stack(noisy_all),
        presence=np.stack(presence_all),
        center_ev=np.stack(centers_all),
        shape_code=np.stack(shapes_all),
        component_area_window=np.stack(areas_all),
    )
    print(f"Dataset saved to {output / 'dataset.npz'}")


if __name__ == "__main__":
    main()
