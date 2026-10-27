"""Generate synthetic atmospheric C 1s spectra in one NeXus/HDF5 file.

This is a single-file generator. Its numerical priors are illustrative and should
be calibrated against measured spectra before drawing scientific conclusions.
The file uses generic NXentry/NXdata/NXprocess groups. It does not claim NXxps
compliance because synthetic scans have no real instrument/sample metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from lmfit.models import DoniachModel, GaussianModel, PseudoVoigtModel, VoigtModel
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve


COMPONENTS = (
    "CC_CH", "C_O", "C_eq_O", "O_C_eq_O", "aromatic_graphitic_sp2", "pi_pi_star",
)
NOMINAL_CENTERS_EV = np.array((284.8, 286.3, 287.8, 289.0, 284.4, 290.6))
FAMILIES = ("hydrocarbon_rich", "oxidized_organic", "soot_rich")
FAMILY_WEIGHTS = (0.35, 0.40, 0.25)
BACKGROUND_KINDS = ("linear", "shirley", "tougaard")
BACKGROUND_WEIGHTS = (0.35, 0.40, 0.25)
SHAPE_NAMES = ("absent", "voigt", "pseudo_voigt", "doniach_gaussian", "gaussian")
ABSENT, VOIGT, PSEUDO_VOIGT, DONIACH_GAUSSIAN, GAUSSIAN = range(5)
ORDINARY_SHAPES = (GAUSSIAN, VOIGT, PSEUDO_VOIGT)
ORDINARY_SHAPE_WEIGHTS = (0.30, 0.35, 0.35)
SATELLITE_PARENT = 4
ENERGY_EV = np.linspace(280.0, 300.0, 401, dtype=np.float64)
ENERGY_STEP_EV = float(ENERGY_EV[1] - ENERGY_EV[0])
NATIVE_STEPS_EV = (0.05, 0.10, 0.20)
FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "datasets" / "c1s_nexus" /
    "c1s_atmospheric_v2.nxs"
)
MODELS = {
    GAUSSIAN: GaussianModel(), VOIGT: VoigtModel(),
    PSEUDO_VOIGT: PseudoVoigtModel(),
    DONIACH_GAUSSIAN: DoniachModel(),
}

# Conditional synthetic priors; record these values when reporting the dataset.
FAMILY_PRIORS = {
    "hydrocarbon_rich": {
        "presence": (1.0, 0.55, 0.18, 0.12, 0.35),
        "dirichlet": (6.0, 1.6, 0.6, 0.5, 1.7),
        "satellite_probability": 0.40,
    },
    "oxidized_organic": {
        "presence": (1.0, 0.95, 0.72, 0.65, 0.12),
        "dirichlet": (2.8, 3.0, 1.6, 1.5, 0.7),
        "satellite_probability": 0.35,
    },
    "soot_rich": {
        "presence": (0.80, 0.58, 0.28, 0.20, 1.0),
        "dirichlet": (1.7, 1.1, 0.7, 0.6, 6.0),
        "satellite_probability": 0.80,
    },
}


def _fwhm(x: np.ndarray, y: np.ndarray) -> float:
    """Numerical FWHM, including for an asymmetric peak."""
    maximum = int(np.argmax(y))
    half = float(y[maximum]) / 2.0
    left = np.flatnonzero(y[:maximum] <= half)
    right = np.flatnonzero(y[maximum + 1:] <= half)
    if not len(left) or not len(right):
        raise ValueError("Peak FWHM falls outside the energy window")
    lo = int(left[-1])
    hi = int(maximum + 1 + right[0])
    x_left = x[lo] + (half - y[lo]) * (x[lo + 1] - x[lo]) / (y[lo + 1] - y[lo])
    x_right = x[hi - 1] + (half - y[hi - 1]) * (
        x[hi] - x[hi - 1]
    ) / (y[hi] - y[hi - 1])
    return float(x_right - x_left)


def _unit_profile(
    shape: int, center: float, sigma: float, *, lorentz_gamma: float = np.nan,
    pseudo_fraction: float = np.nan, asymmetry_gamma: float = np.nan,
    instrument_fwhm: float = 0.0,
) -> np.ndarray:
    """Evaluate lmfit and normalize the area within the 280-300 eV window."""
    model = MODELS[shape]
    if shape == GAUSSIAN:
        params = model.make_params(amplitude=1.0, center=center, sigma=sigma)
    elif shape == VOIGT:
        params = model.make_params(amplitude=1.0, center=center, sigma=sigma)
        # lmfit's VoigtModel otherwise constrains gamma to sigma.
        params["gamma"].set(value=lorentz_gamma, expr="")
    elif shape == PSEUDO_VOIGT:
        params = model.make_params(
            amplitude=1.0, center=center, sigma=sigma, fraction=pseudo_fraction,
        )
    else:
        params = model.make_params(
            amplitude=1.0, center=center, sigma=sigma, gamma=asymmetry_gamma,
        )
    profile = np.asarray(model.eval(params, x=ENERGY_EV), dtype=np.float64)
    if shape == DONIACH_GAUSSIAN:
        sigma_points = instrument_fwhm / FWHM_TO_SIGMA / ENERGY_STEP_EV
        profile = gaussian_filter1d(profile, sigma=sigma_points, mode="constant")
    profile = np.maximum(profile, 0.0)
    area = float(trapezoid(profile, ENERGY_EV))
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Invalid component area")
    return profile / area


def _sample_background(
    rng: np.random.Generator, peak_sum: np.ndarray,
) -> tuple[np.ndarray, str, np.ndarray]:
    """Forward-generate linear, Shirley, or Tougaard-form backgrounds.

    This uses the known clean peak envelope, rather than estimating a baseline
    from noisy measured counts as a background-subtraction algorithm would.
    """
    kind = str(rng.choice(BACKGROUND_KINDS, p=BACKGROUND_WEIGHTS))
    peak_max = float(peak_sum.max())
    low_fraction = float(rng.uniform(0.005, 0.050))
    high_fraction = float(rng.uniform(0.005, 0.050))
    position = (ENERGY_EV - ENERGY_EV[0]) / np.ptp(ENERGY_EV)
    background = peak_max * (
        low_fraction + (high_fraction - low_fraction) * position
    )
    shirley_step_fraction = np.nan
    tougaard_b_ev2 = np.nan
    tougaard_c_ev2 = np.nan
    if kind == "shirley":
        # With the clean zero-loss signal known, the Shirley integral is direct.
        shirley_step_fraction = float(rng.uniform(0.04, 0.22))
        cumulative = cumulative_trapezoid(peak_sum, ENERGY_EV, initial=0.0)
        background += peak_max * shirley_step_fraction * cumulative / cumulative[-1]
    elif kind == "tougaard":
        # The ATMOS notebook's universal B,T,C loss kernel. B is varied modestly;
        # C remains at its reference value (eV^2). Losses run to higher BE.
        tougaard_b_ev2 = float(2866.0 * rng.uniform(0.8, 1.2))
        tougaard_c_ev2 = 1643.0
        loss_ev = ENERGY_EV - ENERGY_EV[0]
        kernel = tougaard_b_ev2 * loss_ev / (tougaard_c_ev2 + loss_ev**2)**2
        loss_signal = np.zeros_like(peak_sum)
        # The Tougaard subtraction relation uses the total signal. Solve the
        # corresponding forward relation J = zero_loss + K * J iteratively.
        for _ in range(20):
            updated = fftconvolve(
                peak_sum + loss_signal, kernel, mode="full"
            )[:ENERGY_EV.size] * ENERGY_STEP_EV
            updated = np.maximum(updated, 0.0)
            if np.max(np.abs(updated - loss_signal)) < 1e-8 * peak_max:
                loss_signal = updated
                break
            loss_signal = updated
        else:
            raise RuntimeError("Tougaard forward background failed to converge")
        background += loss_signal
    parameters = np.array((
        low_fraction, high_fraction, shirley_step_fraction,
        tougaard_b_ev2, tougaard_c_ev2,
    ))
    return background, kind, parameters


def _sample_case(rng: np.random.Generator) -> dict[str, object]:
    family = str(rng.choice(FAMILIES, p=FAMILY_WEIGHTS))
    prior = FAMILY_PRIORS[family]
    present = np.zeros(len(COMPONENTS), dtype=bool)
    present[:5] = rng.random(5) < np.asarray(prior["presence"])
    present[5] = bool(
        present[SATELLITE_PARENT]
        and rng.random() < prior["satellite_probability"]
    )

    # All components of a case share charging shift and instrument resolution.
    charge_shift = float(np.clip(rng.normal(0.0, 0.13), -0.35, 0.35))
    centers = np.full(len(COMPONENTS), np.nan, dtype=np.float64)
    centers[:5] = NOMINAL_CENTERS_EV[:5] + charge_shift + rng.normal(0, 0.08, 5)
    if present[5]:
        centers[5] = centers[SATELLITE_PARENT] + rng.uniform(5.8, 6.5)
    centers[~present] = np.nan

    areas = np.zeros(len(COMPONENTS), dtype=np.float64)
    main_indices = np.flatnonzero(present[:5])
    areas[main_indices] = rng.dirichlet(
        np.asarray(prior["dirichlet"])[main_indices]
    )
    satellite_ratio = np.nan
    if present[5]:
        satellite_ratio = float(rng.uniform(0.04, 0.20))
        areas[5] = areas[SATELLITE_PARENT] * satellite_ratio
    areas /= areas.sum()

    instrument_fwhm = float(rng.uniform(0.35, 0.80))
    chemical_fwhm = float(rng.uniform(0.65, 1.30))
    ordinary_shape = int(rng.choice(ORDINARY_SHAPES, p=ORDINARY_SHAPE_WEIGHTS))
    shared_lorentz_gamma = float(rng.uniform(0.09, 0.23))
    shared_pseudo_fraction = float(rng.uniform(0.12, 0.48))
    shape_code = np.zeros(len(COMPONENTS), dtype=np.int8)
    model_sigma = np.full(len(COMPONENTS), np.nan)
    lorentz_gamma = np.full(len(COMPONENTS), np.nan)
    pseudo_fraction = np.full(len(COMPONENTS), np.nan)
    asymmetry_gamma = np.full(len(COMPONENTS), np.nan)
    fwhm = np.full(len(COMPONENTS), np.nan)
    components = np.zeros((len(COMPONENTS), ENERGY_EV.size), dtype=np.float64)

    for index in np.flatnonzero(present):
        local_chemical_fwhm = float(np.clip(
            chemical_fwhm + rng.normal(0.0, 0.09), 0.55, 1.50,
        ))
        gaussian_fwhm = float(np.hypot(local_chemical_fwhm, instrument_fwhm))
        if index == 5:
            shape = PSEUDO_VOIGT
            gaussian_fwhm *= float(rng.uniform(1.35, 1.80))
        elif family == "soot_rich" and index == SATELLITE_PARENT:
            shape = DONIACH_GAUSSIAN
        else:
            shape = ordinary_shape

        shape_code[index] = shape
        if shape == GAUSSIAN:
            # Chemical disorder and instrument Gaussian widths combine in quadrature.
            model_sigma[index] = gaussian_fwhm / FWHM_TO_SIGMA
        elif shape == VOIGT:
            model_sigma[index] = gaussian_fwhm / FWHM_TO_SIGMA
            lorentz_gamma[index] = shared_lorentz_gamma * rng.uniform(0.85, 1.15)
        elif shape == PSEUDO_VOIGT:
            # Approximate Voigt FWHM before using lmfit's pseudo-Voigt model.
            lorentz_fwhm = 2.0 * shared_lorentz_gamma
            target_fwhm = 0.5346 * lorentz_fwhm + np.sqrt(
                0.2166 * lorentz_fwhm ** 2 + gaussian_fwhm ** 2
            )
            model_sigma[index] = target_fwhm / 2.0
            pseudo_fraction[index] = float(np.clip(
                shared_pseudo_fraction + rng.normal(0.0, 0.05), 0.05, 0.65,
            ))
        else:
            # Doniach sigma is a line-shape scale, not Gaussian sigma.
            model_sigma[index] = local_chemical_fwhm / 2.0
            asymmetry_gamma[index] = float(rng.uniform(0.04, 0.14))

        unit = _unit_profile(
            shape, float(centers[index]), float(model_sigma[index]),
            lorentz_gamma=float(lorentz_gamma[index]),
            pseudo_fraction=float(pseudo_fraction[index]),
            asymmetry_gamma=float(asymmetry_gamma[index]),
            instrument_fwhm=instrument_fwhm,
        )
        components[index] = unit * areas[index]
        fwhm[index] = _fwhm(ENERGY_EV, unit)

    peak_sum = components.sum(axis=0)
    background, background_kind, background_parameters = _sample_background(
        rng, peak_sum
    )
    return {
        "family": family, "present": present, "centers": centers,
        "shape_code": shape_code, "model_sigma": model_sigma,
        "lorentz_gamma": lorentz_gamma, "pseudo_fraction": pseudo_fraction,
        "asymmetry_gamma": asymmetry_gamma, "fwhm": fwhm,
        "components": components, "background": background,
        "background_kind": background_kind,
        "background_parameters": background_parameters,
        "instrument_fwhm": instrument_fwhm,
        "charge_shift": charge_shift, "satellite_ratio": satellite_ratio,
    }


def _compressed(group: h5py.Group, name: str, **kwargs: object) -> h5py.Dataset:
    return group.create_dataset(
        name, compression="gzip", compression_opts=4, shuffle=True, **kwargs
    )


def generate(output: Path, n_cases: int, replicates: int = 5, seed: int = 42) -> None:
    """Write a new .nxs file. Refuse to overwrite an existing dataset."""
    if n_cases < 10 or replicates < 1:
        raise ValueError("Use at least 10 cases and one acquisition per case")
    if output.suffix.lower() != ".nxs":
        raise ValueError("The output file must have a .nxs extension")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")

    rng = np.random.default_rng(seed)
    n_scans = n_cases * replicates
    n_points = ENERGY_EV.size
    n_components = len(COMPONENTS)
    case_split = np.full(n_cases, b"test", dtype="S10")
    shuffled = rng.permutation(n_cases)
    n_train = max(1, int(0.8 * n_cases))
    n_validation = max(1, int(0.1 * n_cases))
    case_split[shuffled[:n_train]] = b"train"
    case_split[shuffled[n_train:n_train + n_validation]] = b"validation"

    labels = {
        "peak_present": np.empty((n_scans, n_components), dtype=bool),
        "peak_center_ev": np.empty((n_scans, n_components), dtype=np.float32),
        "peak_fwhm_ev": np.empty((n_scans, n_components), dtype=np.float32),
        "peak_shape_code": np.empty((n_scans, n_components), dtype=np.int8),
        "peak_model_sigma_ev": np.empty((n_scans, n_components), dtype=np.float32),
        "peak_lorentz_gamma_ev": np.empty((n_scans, n_components), dtype=np.float32),
        "peak_pseudo_voigt_fraction": np.empty((n_scans, n_components), dtype=np.float32),
        "peak_asymmetry_gamma": np.empty((n_scans, n_components), dtype=np.float32),
        "peak_height_counts": np.empty((n_scans, n_components), dtype=np.float32),
        "peak_area_counts_ev": np.empty((n_scans, n_components), dtype=np.float32),
        "peak_area_fraction": np.empty((n_scans, n_components), dtype=np.float32),
        "family": np.empty(n_scans, dtype="S24"),
        "background_kind": np.empty(n_scans, dtype="S24"),
        "split": np.empty(n_scans, dtype="S10"),
        "case_id": np.repeat(np.arange(n_cases, dtype=np.int32), replicates),
        "sweep": np.tile(np.arange(replicates, dtype=np.int16), n_cases),
        "native_step_ev": np.empty(n_scans, dtype=np.float32),
        "native_point_count": np.empty(n_scans, dtype=np.int16),
        "instrument_gaussian_fwhm_ev": np.empty(n_scans, dtype=np.float32),
        "charge_shift_ev": np.empty(n_scans, dtype=np.float32),
        "satellite_to_parent_area_ratio": np.empty(n_scans, dtype=np.float32),
        "target_maximum_counts": np.empty(n_scans, dtype=np.float32),
        "scale_counts_per_unit_profile": np.empty(n_scans, dtype=np.float32),
        "background_parameters": np.empty((n_scans, 5), dtype=np.float32),
    }

    # Keep a partially written file hidden until generation finishes.
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".partial", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as nexus:
            nexus.attrs["default"] = "entry"
            nexus.attrs["file_name"] = output.name
            nexus.attrs["synthetic"] = True
            entry = nexus.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            entry.attrs["default"] = "data"
            entry.create_dataset("title", data="Synthetic atmospheric C 1s XPS spectra")
            entry.create_dataset("start_time", data=datetime.now(timezone.utc).isoformat())

            data = entry.create_group("data")
            data.attrs["NX_class"] = "NXdata"
            data.attrs["signal"] = "counts"
            data.attrs["axes"] = np.asarray(
                ("scan_index", "binding_energy"), dtype=h5py.string_dtype("utf-8")
            )
            data.attrs["scan_index_indices"] = 0
            data.attrs["binding_energy_indices"] = 1
            data.attrs["auxiliary_signals"] = np.asarray(
                ("expected_counts", "background_counts"),
                dtype=h5py.string_dtype("utf-8"),
            )
            data.create_dataset("scan_index", data=np.arange(n_scans, dtype=np.int32))
            energy = data.create_dataset("binding_energy", data=ENERGY_EV.astype("f4"))
            energy.attrs["units"] = "eV"
            scan_chunk = (min(max(replicates, 8), n_scans), n_points)
            counts = _compressed(data, "counts", shape=(n_scans, n_points),
                                 dtype="f4", chunks=scan_chunk)
            expected = _compressed(data, "expected_counts", shape=(n_scans, n_points),
                                   dtype="f4", chunks=scan_chunk)
            backgrounds = _compressed(data, "background_counts", shape=(n_scans, n_points),
                                      dtype="f4", chunks=scan_chunk)
            for dataset in (counts, expected, backgrounds):
                dataset.attrs["units"] = "counts"
            counts.attrs["description"] = "Poisson native scans interpolated to shared energy grid"

            acquisition = entry.create_group("acquisition")
            acquisition.attrs["NX_class"] = "NXcollection"
            native_counts = _compressed(
                acquisition, "native_counts", shape=(n_scans, n_points),
                dtype="i4", chunks=scan_chunk, fillvalue=-1,
            )
            native_energy = _compressed(
                acquisition, "native_binding_energy_ev", shape=(n_scans, n_points),
                dtype="f4", chunks=scan_chunk, fillvalue=np.nan,
            )
            native_counts.attrs["units"] = "counts"
            native_counts.attrs["padding_value"] = -1
            native_energy.attrs["units"] = "eV"
            acquisition.attrs["description"] = (
                "Use labels/native_point_count to remove padding from each native scan"
            )

            simulation = entry.create_group("simulation")
            simulation.attrs["NX_class"] = "NXprocess"
            simulation.create_dataset("program", data=Path(__file__).name)
            simulation.create_dataset("version", data="2")
            simulation.create_dataset("date", data=datetime.now(timezone.utc).isoformat())
            simulation.create_dataset("script_sha256", data=hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest())
            parameters = simulation.create_group("parameters")
            parameters.attrs["NX_class"] = "NXparameters"
            for name, value in (("seed", seed), ("n_cases", n_cases),
                                ("replicates", replicates)):
                parameters.create_dataset(name, data=value)
            parameters.create_dataset("family_weights", data=FAMILY_WEIGHTS)
            parameters.create_dataset("background_weights", data=BACKGROUND_WEIGHTS)
            parameters.create_dataset("ordinary_shape_codes", data=ORDINARY_SHAPES)
            parameters.create_dataset("ordinary_shape_weights", data=ORDINARY_SHAPE_WEIGHTS)
            parameters.create_dataset("native_steps_ev", data=NATIVE_STEPS_EV)
            parameters.create_dataset("nominal_centers_ev", data=NOMINAL_CENTERS_EV)
            parameters.create_dataset("family_names", data=np.asarray(
                FAMILIES, dtype=h5py.string_dtype("utf-8")
            ))
            parameters.create_dataset("component_presence_probabilities", data=np.asarray(
                [FAMILY_PRIORS[name]["presence"] for name in FAMILIES]
            ))
            parameters.create_dataset("component_dirichlet_concentrations", data=np.asarray(
                [FAMILY_PRIORS[name]["dirichlet"] for name in FAMILIES]
            ))
            parameters.create_dataset("satellite_probabilities", data=np.asarray(
                [FAMILY_PRIORS[name]["satellite_probability"] for name in FAMILIES]
            ))
            parameters.create_dataset("sampling_rules_json", data=json.dumps({
                "charge_shift_ev": "clip(normal(0, 0.13), -0.35, 0.35)",
                "state_center_jitter_ev": "normal(0, 0.08)",
                "satellite_offset_ev": "uniform(5.8, 6.5) from sp2 parent",
                "satellite_parent_area_ratio": "uniform(0.04, 0.20)",
                "instrument_gaussian_fwhm_ev": "uniform(0.35, 0.80)",
                "shared_chemical_fwhm_ev": "uniform(0.65, 1.30)",
                "component_chemical_fwhm_ev": "clip(shared + normal(0, 0.09), 0.55, 1.50)",
                "voigt_lorentz_hwhm_ev": "uniform(0.09, 0.23) times uniform(0.85, 1.15)",
                "pseudo_voigt_lorentz_fraction": "clip(uniform(0.12, 0.48) + normal(0, 0.05), 0.05, 0.65)",
                "graphitic_doniach_gamma": "uniform(0.04, 0.14)",
                "satellite_width_multiplier": "uniform(1.35, 1.80)",
                "target_maximum_counts": "log_uniform(300, 100000)",
                "background_families": BACKGROUND_KINDS,
                "linear_endpoint_fraction_of_peak_max": "independent uniform(0.005, 0.050) at each end",
                "shirley_step_fraction_of_peak_max": "uniform(0.04, 0.22)",
                "tougaard_B_ev2": "2866 times uniform(0.8, 1.2)",
                "tougaard_C_ev2": 1643.0,
            }))
            simulation.attrs["area_convention"] = (
                "Component areas are normalized over the finite 280-300 eV window"
            )
            simulation.attrs["background_convention"] = (
                "Forward models of linear, Shirley, and Tougaard backgrounds on "
                "known clean peaks; Tougaard uses the universal kernel only "
                "within this 20 eV window, not a full measured-spectrum fit"
            )
            simulation.attrs["background_reference"] = (
                "ATMOS XPS-background-subtraction notebook; "
                "Tougaard, J. Vac. Sci. Technol. A 39, 011201 (2021), "
                "doi:10.1116/6.0000661"
            )
            case_components = _compressed(
                simulation, "case_component_profiles", shape=(n_cases, n_components, n_points),
                dtype="f4", chunks=(1, n_components, n_points),
            )
            case_backgrounds = _compressed(
                simulation, "case_background_profiles", shape=(n_cases, n_points),
                dtype="f4", chunks=(1, n_points),
            )
            case_components.attrs["units"] = "fraction / eV"
            case_backgrounds.attrs["units"] = "fraction / eV"

            for case_id in range(n_cases):
                sampled = _sample_case(rng)
                components = sampled["components"]
                background = sampled["background"]
                profile = components.sum(axis=0) + background
                case_components[case_id] = components.astype("f4")
                case_backgrounds[case_id] = background.astype("f4")

                for sweep in range(replicates):
                    row = case_id * replicates + sweep
                    step = float(rng.choice(NATIVE_STEPS_EV))
                    native_axis = np.linspace(
                        ENERGY_EV[0], ENERGY_EV[-1], int(round(20.0 / step)) + 1
                    )
                    native_profile = np.interp(native_axis, ENERGY_EV, profile)
                    native_background = np.interp(native_axis, ENERGY_EV, background)
                    target_max = float(10 ** rng.uniform(np.log10(300), np.log10(100_000)))
                    scale = target_max / native_profile.max()
                    observed_native = rng.poisson(native_profile * scale).astype("i4")
                    counts[row] = np.interp(
                        ENERGY_EV, native_axis, observed_native
                    ).astype("f4")
                    expected[row] = np.interp(
                        ENERGY_EV, native_axis, native_profile * scale
                    ).astype("f4")
                    backgrounds[row] = np.interp(
                        ENERGY_EV, native_axis, native_background * scale
                    ).astype("f4")
                    native_counts[row, :native_axis.size] = observed_native
                    native_energy[row, :native_axis.size] = native_axis.astype("f4")

                    labels["peak_present"][row] = sampled["present"]
                    labels["peak_center_ev"][row] = sampled["centers"]
                    labels["peak_fwhm_ev"][row] = sampled["fwhm"]
                    labels["peak_shape_code"][row] = sampled["shape_code"]
                    labels["peak_model_sigma_ev"][row] = sampled["model_sigma"]
                    labels["peak_lorentz_gamma_ev"][row] = sampled["lorentz_gamma"]
                    labels["peak_pseudo_voigt_fraction"][row] = sampled["pseudo_fraction"]
                    labels["peak_asymmetry_gamma"][row] = sampled["asymmetry_gamma"]
                    areas = trapezoid(components, ENERGY_EV, axis=1) * scale
                    labels["peak_area_counts_ev"][row] = areas
                    labels["peak_area_fraction"][row] = areas / areas.sum()
                    labels["peak_height_counts"][row] = components.max(axis=1) * scale
                    labels["family"][row] = sampled["family"].encode("ascii")
                    labels["background_kind"][row] = sampled["background_kind"].encode("ascii")
                    labels["background_parameters"][row] = sampled["background_parameters"]
                    labels["split"][row] = case_split[case_id]
                    labels["native_step_ev"][row] = step
                    labels["native_point_count"][row] = native_axis.size
                    labels["instrument_gaussian_fwhm_ev"][row] = sampled["instrument_fwhm"]
                    labels["charge_shift_ev"][row] = sampled["charge_shift"]
                    labels["satellite_to_parent_area_ratio"][row] = sampled["satellite_ratio"]
                    labels["target_maximum_counts"][row] = target_max
                    labels["scale_counts_per_unit_profile"][row] = scale

            ground_truth = entry.create_group("labels")
            ground_truth.attrs["NX_class"] = "NXcollection"
            ground_truth.attrs["satellite_parent_component_index"] = SATELLITE_PARENT
            ground_truth.attrs["missing_numeric_label"] = "NaN for absent or inapplicable values"
            string_type = h5py.string_dtype("utf-8")
            ground_truth.create_dataset(
                "component_names", data=np.asarray(COMPONENTS, dtype=string_type)
            )
            ground_truth.create_dataset(
                "shape_names", data=np.asarray(SHAPE_NAMES, dtype=string_type)
            )
            ground_truth.create_dataset("background_parameter_names", data=np.asarray(
                ("linear_low_fraction", "linear_high_fraction",
                 "shirley_step_fraction", "tougaard_B_eV2", "tougaard_C_eV2"),
                dtype=string_type,
            ))
            for name, values in labels.items():
                dataset = _compressed(ground_truth, name, data=values)
                if name.endswith("_ev"):
                    dataset.attrs["units"] = "eV"
                if name.endswith("_counts"):
                    dataset.attrs["units"] = "counts"
            ground_truth["peak_area_counts_ev"].attrs["units"] = "counts * eV"
            ground_truth["peak_lorentz_gamma_ev"].attrs["description"] = (
                "Voigt Lorentzian half-width; NaN for other profile families"
            )
            ground_truth["peak_pseudo_voigt_fraction"].attrs["description"] = (
                "Lorentzian weight in lmfit PseudoVoigtModel; NaN otherwise"
            )
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=int, default=10_000)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output, args.cases, args.replicates, args.seed)
    print(f"Saved {args.cases * args.replicates:,} C 1s scans to {args.output}")


if __name__ == "__main__":
    main()
