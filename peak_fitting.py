"""Build lmfit peak models directly from predicted peak records.

The adapter accepts any number of peaks with center and width estimates. Peak
names are labels, not chemical rules. Missing profile types are selected from a
small built-in catalogue by fit quality; background choice is automatic too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from lmfit import Parameters, minimize
from lmfit.models import DoniachModel, GaussianModel, PseudoVoigtModel, VoigtModel
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve


PROFILE_MODELS = {
    "gaussian": GaussianModel,
    "voigt": VoigtModel,
    "pseudo_voigt": PseudoVoigtModel,
    "doniach_gaussian": DoniachModel,
}
BACKGROUND_KINDS = ("linear", "shirley", "tougaard")
FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))


@dataclass
class PeakFit:
    energy_ev: np.ndarray
    observed_counts: np.ndarray
    initial_counts: np.ndarray
    fitted_counts: np.ndarray
    background_counts: np.ndarray
    component_counts: dict[str, np.ndarray]
    peaks: list[dict[str, Any]]
    background_kind: str
    bic: float
    success: bool
    fit_result: Any


def _number(record: dict, names: tuple[str, ...], default: float | None = None) -> float | None:
    for name in names:
        value = record.get(name)
        if value is not None:
            number = float(value)
            if np.isfinite(number):
                return number
    return default


def _normalize_predictions(predictions: Sequence[dict], x: np.ndarray, y: np.ndarray) -> list[dict]:
    active = []
    names = set()
    span = float(x[-1] - x[0])
    step = float(np.median(np.diff(x)))
    for index, record in enumerate(predictions):
        if not record.get("present", True):
            continue
        name = str(record.get("component", record.get("name", f"peak_{index}")))
        if name in names:
            raise ValueError(f"Duplicate peak name: {name}")
        names.add(name)
        center = _number(record, ("center_ev", "center"))
        width = _number(record, ("fwhm_ev", "fwhm", "width_ev", "width"))
        sigma = _number(record, ("model_sigma_ev", "sigma_ev", "sigma"))
        if center is None or not x[0] <= center <= x[-1]:
            raise ValueError(f"{name}: center must lie inside the energy axis")
        if width is None and sigma is not None:
            width = sigma * FWHM_FACTOR
        if width is None or width <= 0:
            raise ValueError(f"{name}: provide a positive FWHM or sigma")
        profile = str(record.get("profile") or "auto").lower().replace("-", "_")
        if profile != "auto" and profile not in PROFILE_MODELS:
            raise ValueError(f"{name}: unsupported profile {profile!r}")
        area = _number(record, ("area_counts_ev", "area"))
        height = _number(record, ("height_counts", "height"))
        if area is None or area <= 0:
            if height is None or height <= 0:
                height = float(np.interp(center, x, y))
            area = max(height, 1.0) * max(width, step)
        active.append({
            "name": name, "center": center, "width": float(np.clip(width, step, span)),
            "sigma": sigma, "area": area, "profile": profile,
            "lorentz_gamma": _number(record, ("lorentz_gamma_ev", "lorentz_gamma")),
            "fraction": _number(record, ("pseudo_voigt_fraction", "fraction")),
            "asymmetry": _number(record, ("asymmetry_gamma",)),
            "instrument_fwhm": _number(record, ("instrument_fwhm_ev",)),
        })
    if not active:
        raise ValueError("No present peaks were supplied")
    return active


def fit_predicted_peaks(
    energy_ev: Sequence[float], counts: Sequence[float], predictions: Sequence[dict],
    *, backgrounds: Sequence[str] = BACKGROUND_KINDS,
    refine_profiles: bool = False, max_nfev: int = 1200,
) -> PeakFit:
    """Fit predicted peaks and choose supported backgrounds by BIC.

    Each prediction needs a center and FWHM (or sigma); area, profile and shape
    parameters are optional. Unknown profiles start Gaussian and are selected by
    a greedy BIC comparison. Set ``refine_profiles`` to compare alternative
    profiles even when the predictor supplies one.
    """
    x = np.asarray(energy_ev, dtype=float)
    y = np.asarray(counts, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 20:
        raise ValueError("Energy and counts must be matching one-dimensional arrays of at least 20 points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(y < 0):
        raise ValueError("Energy and counts must be finite; counts must be nonnegative")
    if not np.all(np.diff(x) > 0):
        raise ValueError("Energy must be strictly increasing")
    backgrounds = tuple(dict.fromkeys(backgrounds))
    if not backgrounds or any(kind not in BACKGROUND_KINDS for kind in backgrounds):
        raise ValueError(f"Choose one or more of {BACKGROUND_KINDS}")
    guesses = _normalize_predictions(predictions, x, y)
    step = float(np.median(np.diff(x)))
    uniform = np.allclose(np.diff(x), step, rtol=0.02, atol=1e-9)
    span = float(x[-1] - x[0])
    convolution_x = x if uniform else np.linspace(x[0], x[-1], len(x))
    convolution_step = float(convolution_x[1] - convolution_x[0])

    def broaden(curve: np.ndarray, fwhm: float) -> np.ndarray:
        sampled = curve if uniform else np.interp(convolution_x, x, curve)
        smoothed = gaussian_filter1d(
            sampled, fwhm / (FWHM_FACTOR * convolution_step), mode="constant"
        )
        return smoothed if uniform else np.interp(x, convolution_x, smoothed)

    edge = max(5, min(15, len(y) // 20))
    edge_low = max(float(np.median(y[:edge])), 0.0)
    edge_high = max(float(np.median(y[-edge:])), 0.0)

    def run(profiles: tuple[str, ...], background_kind: str):
        params = Parameters()
        models = []
        if "doniach_gaussian" in profiles:
            instrument = next((g["instrument_fwhm"] for g in guesses
                               if g["instrument_fwhm"] is not None), None)
            start = instrument if instrument is not None else max(2 * step, min(g["width"] for g in guesses) * 0.5)
            params.add("instrument_fwhm_ev", value=float(np.clip(start, step, span / 3)),
                       min=step * 0.5, max=span / 2)
        for slot, (guess, profile) in enumerate(zip(guesses, profiles)):
            prefix = f"p{slot}_"
            model = PROFILE_MODELS[profile](prefix=prefix)
            sigma = guess["sigma"]
            if sigma is None or guess["profile"] != profile:
                sigma = guess["width"] / (2 if profile == "pseudo_voigt" else FWHM_FACTOR)
            sigma = float(np.clip(sigma, step * 0.5, span / 3))
            local = model.make_params(amplitude=1.0, center=guess["center"], sigma=sigma)
            if profile == "voigt":
                gamma = guess["lorentz_gamma"] or sigma * 0.5
                local[prefix + "gamma"].set(value=float(np.clip(gamma, step * 0.1, span / 4)),
                                            min=step * 0.05, max=span / 3, expr="")
            elif profile == "pseudo_voigt":
                local[prefix + "fraction"].set(value=float(np.clip(guess["fraction"] if guess["fraction"] is not None else 0.3, 0.0, 1.0)), min=0, max=1)
            elif profile == "doniach_gaussian":
                local[prefix + "gamma"].set(value=float(np.clip(guess["asymmetry"] if guess["asymmetry"] is not None else 0.08, 0.001, 0.35)), min=0.001, max=0.35)
            unit = np.asarray(model.eval(params=local, x=x), dtype=float)
            if profile == "doniach_gaussian":
                unit = broaden(unit, params["instrument_fwhm_ev"].value)
            unit_area = float(trapezoid(unit, x))
            if not np.isfinite(unit_area) or unit_area <= 0:
                raise ValueError(f"Invalid initial profile for {guess['name']}")
            amplitude = max(guess["area"] / unit_area, 1e-9)
            local[prefix + "amplitude"].set(value=amplitude, min=0, max=max(amplitude * 50, float(y.max()) * span * 10, 1))
            movement = max(guess["width"] * 1.5, 3 * step)
            local[prefix + "center"].set(value=guess["center"],
                                          min=max(x[0], guess["center"] - movement),
                                          max=min(x[-1], guess["center"] + movement))
            local[prefix + "sigma"].set(value=sigma, min=step * 0.25, max=span / 2)
            params.update(local)
            models.append((guess["name"], profile, prefix, model))
        ceiling = max(float(y.max()) * 2, 1)
        params.add("background_low", value=min(edge_low, ceiling), min=0, max=ceiling)
        params.add("background_high", value=min(edge_high, ceiling), min=0, max=ceiling)
        if background_kind == "shirley":
            params.add("shirley_fraction", value=0.12, min=0, max=0.8)
        if background_kind == "tougaard":
            params.add("tougaard_B_eV2", value=2866, min=100, max=10000)

        def evaluate(fit_params):
            parts = {}
            for name, profile, _, model in models:
                curve = np.asarray(model.eval(params=fit_params, x=x), dtype=float)
                if profile == "doniach_gaussian":
                    curve = broaden(curve, fit_params["instrument_fwhm_ev"].value)
                parts[name] = curve
            peak_sum = np.sum(list(parts.values()), axis=0)
            position = (x - x[0]) / span
            background = fit_params["background_low"].value + (fit_params["background_high"].value - fit_params["background_low"].value) * position
            if background_kind == "shirley":
                cumulative = cumulative_trapezoid(peak_sum, x, initial=0)
                background += fit_params["shirley_fraction"].value * peak_sum.max() * cumulative / max(float(cumulative[-1]), 1e-12)
            elif background_kind == "tougaard":
                loss = convolution_x - convolution_x[0]
                kernel = fit_params["tougaard_B_eV2"].value * loss / (1643.0 + loss**2)**2
                peak_on_grid = peak_sum if uniform else np.interp(convolution_x, x, peak_sum)
                signal = np.zeros_like(peak_on_grid)
                for _ in range(20):
                    updated = fftconvolve(peak_on_grid + signal, kernel, mode="full")[:len(x)] * convolution_step
                    if np.max(np.abs(updated - signal)) < 1e-8 * max(float(peak_sum.max()), 1):
                        signal = updated
                        break
                    signal = updated
                background += signal if uniform else np.interp(x, convolution_x, signal)
            return peak_sum + background, parts, background

        initial, _, _ = evaluate(params)
        def residual(fit_params):
            fitted, _, _ = evaluate(fit_params)
            return (fitted - y) / np.sqrt(np.maximum(y, 1))
        result = minimize(residual, params, method="least_squares", max_nfev=max_nfev)
        if not np.isfinite(result.bic):
            raise RuntimeError("lmfit returned a non-finite BIC")
        return float(result.bic), bool(result.success), result, initial, evaluate, models, background_kind

    def best_for(profiles, kinds):
        candidates = []
        for kind in kinds:
            try:
                candidates.append(run(profiles, kind))
            except (RuntimeError, ValueError, FloatingPointError):
                continue
        if not candidates:
            raise RuntimeError("No peak/background model yielded a finite fit")
        converged = [candidate for candidate in candidates if candidate[1]]
        return min(converged or candidates, key=lambda candidate: candidate[0])

    profiles = tuple(g["profile"] if g["profile"] != "auto" else "gaussian" for g in guesses)
    best = best_for(profiles, backgrounds)
    if refine_profiles or any(g["profile"] == "auto" for g in guesses):
        for slot, guess in enumerate(guesses):
            if guess["profile"] != "auto" and not refine_profiles:
                continue
            alternatives = []
            for profile in PROFILE_MODELS:
                candidate_profiles = list(profiles)
                candidate_profiles[slot] = profile
                try:
                    alternatives.append((best_for(tuple(candidate_profiles), (best[6],)), tuple(candidate_profiles)))
                except RuntimeError:
                    pass
            if alternatives:
                selected, profiles = min(alternatives, key=lambda item: item[0][0])
                best = selected
        best = best_for(profiles, backgrounds)

    bic, success, result, initial, evaluate, models, background_kind = best
    fitted, parts, background = evaluate(result.params)
    peak_data = []
    for name, profile, prefix, _ in models:
        curve = parts[name]
        maximum = int(np.argmax(curve))
        half = curve[maximum] / 2
        left = np.flatnonzero(curve[:maximum] <= half)
        right = np.flatnonzero(curve[maximum + 1:] <= half)
        fwhm = None
        if len(left) and len(right):
            lo, hi = int(left[-1]), int(maximum + 1 + right[0])
            x_left = float(np.interp(half, curve[lo:lo + 2], x[lo:lo + 2]))
            x_right = float(np.interp(half, curve[hi - 1:hi + 1][::-1], x[hi - 1:hi + 1][::-1]))
            fwhm = x_right - x_left
        peak = {"component": name, "profile": profile,
                "center_ev": float(result.params[prefix + "center"].value),
                "model_sigma_ev": float(result.params[prefix + "sigma"].value),
                "height_counts": float(curve.max()),
                "area_counts_ev": float(trapezoid(curve, x)), "fwhm_ev": fwhm}
        if profile == "voigt":
            peak["lorentz_gamma_ev"] = float(result.params[prefix + "gamma"].value)
        elif profile == "pseudo_voigt":
            peak["pseudo_voigt_fraction"] = float(result.params[prefix + "fraction"].value)
        elif profile == "doniach_gaussian":
            peak["asymmetry_gamma"] = float(result.params[prefix + "gamma"].value)
            peak["instrument_fwhm_ev"] = float(result.params["instrument_fwhm_ev"].value)
        peak_data.append(peak)
    area_sum = sum(peak["area_counts_ev"] for peak in peak_data)
    for peak in peak_data:
        peak["area_fraction"] = peak["area_counts_ev"] / area_sum if area_sum > 0 else 0.0
    return PeakFit(x, y, initial, fitted, background, parts, peak_data,
                   background_kind, bic, success, result)
