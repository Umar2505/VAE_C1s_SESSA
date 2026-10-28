"""Shared format-5 C 1s training, rendering, and evaluation code.

The three notebooks deliberately share this data contract and reporting code so
architecture comparisons do not silently change preprocessing or metrics.
Running/importing this module never starts training.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / "c1s_nexus" / "c1s_peak_training_v1"
OUTPUT_ROOT = ROOT / "outputs" / "c1s_peak_models"
COMPONENT_NAMES = (
    "CC_CH", "C_O", "C_eq_O", "O_C_eq_O", "aromatic_graphitic_sp2", "pi_pi_star"
)
SHAPE_NAMES = ("absent", "voigt", "pseudo_voigt", "doniach_gaussian", "gaussian")
BACKGROUND_NAMES = ("linear", "shirley", "tougaard")
PEAK_FIELDS = (
    "peak_present", "peak_shape_code", "peak_center_ev", "peak_model_sigma_ev",
    "peak_lorentz_gamma_ev", "peak_pseudo_voigt_fraction",
    "peak_asymmetry_gamma", "peak_area_counts_ev",
)
LABEL_FIELDS = PEAK_FIELDS + (
    "instrument_gaussian_fwhm_ev", "background_kind_code", "background_parameters",
    "case_id", "split",
)


@dataclass
class RunConfig:
    model_name: str
    dataset_dir: str = str(DATA_DIR)
    output_dir: str = ""
    wait_hours: float = 24.0
    poll_seconds: int = 30
    seed: int = 42
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 18
    workers: int = 0
    resume: bool = True
    # These are starting weights, recorded alongside every experiment.
    spectrum_weight: float = 2.0
    background_curve_weight: float = 0.3
    detection_weight: float = 1.0
    shape_weight: float = 0.5
    peak_parameter_weight: float = 0.7
    background_parameter_weight: float = 0.2
    instrument_weight: float = 0.03
    vae_beta_max: float = 0.01
    vae_warmup_epochs: int = 25
    teacher_forcing_epochs: int = 20

    def __post_init__(self):
        if self.model_name not in {"cnn", "vae", "transformer"}:
            raise ValueError("model_name must be cnn, vae, or transformer")
        if not self.output_dir:
            self.output_dir = str(OUTPUT_ROOT / self.model_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wait_for_dataset(directory: str | Path, hours: float = 24.0,
                     poll_seconds: int = 30) -> tuple[dict, str]:
    """Wait for the generator's atomic completion manifest, then verify all shards."""
    directory = Path(directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    deadline = time.monotonic() + hours * 3600
    last_message = 0.0
    while True:
        if manifest_path.is_file():
            raw = manifest_path.read_bytes()
            manifest = json.loads(raw)
            if manifest.get("format_version") != 5 or manifest.get("schema") != "independent_peak_parameters_and_clean_spectrum":
                raise ValueError(f"Expected compact format-5 dataset: {manifest_path}")
            shards = manifest.get("shards", [])
            if not shards or sum(int(s["scans"]) for s in shards) != manifest.get("total_scans"):
                raise ValueError("Dataset manifest has missing or inconsistent shards")
            missing = [s["file"] for s in shards if not (directory / s["file"]).is_file()]
            if not missing:
                return manifest, hashlib.sha256(raw).hexdigest()
        else:
            missing = ["manifest.json"]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Dataset did not complete within {hours:g} hours at {directory}; missing: {missing[:5]}")
        if time.monotonic() - last_message >= 300 or last_message == 0:
            print(f"Waiting for dataset generation in {directory} ({len(missing)} files pending)...", flush=True)
            last_message = time.monotonic()
        time.sleep(max(1, poll_seconds))


def _strings(values):
    return tuple(v.decode() if isinstance(v, bytes) else str(v) for v in values)


def load_dataset(directory: str | Path, manifest: dict) -> dict:
    """Load all small shards, retaining source coordinates for failure review."""
    directory = Path(directory).expanduser().resolve()
    blocks = {name: [] for name in (
        "counts", "expected_counts", "background_counts", "native_step_ev", *LABEL_FIELDS
    )}
    source_files, source_rows, families = [], [], []
    axis = None
    for shard in manifest["shards"]:
        path = directory / shard["file"]
        with h5py.File(path, "r") as h5:
            if h5.attrs.get("format_version") != 5:
                raise ValueError(f"Wrong dataset version in {path}")
            data, labels = h5["entry/data"], h5["entry/labels"]
            local_axis = np.asarray(data["binding_energy"], dtype=np.float32)
            if axis is None:
                axis = local_axis
                if _strings(labels["component_names"][:]) != COMPONENT_NAMES:
                    raise ValueError("Component names do not match the training schema")
                if _strings(labels["shape_names"][:]) != SHAPE_NAMES:
                    raise ValueError("Shape codes do not match the training schema")
                if _strings(labels["background_kind_names"][:]) != BACKGROUND_NAMES:
                    raise ValueError("Background codes do not match the training schema")
            elif not np.array_equal(axis, local_axis):
                raise ValueError(f"Energy axis changed in {path}")
            n = int(data["counts"].shape[0])
            if n != int(shard["scans"]):
                raise ValueError(f"Scan count differs from manifest in {path}")
            for name in ("counts", "expected_counts", "background_counts", "native_step_ev"):
                blocks[name].append(np.asarray(data[name]))
            for name in LABEL_FIELDS:
                blocks[name].append(np.asarray(labels[name]))
            source_files.extend([shard["file"]] * n)
            source_rows.extend(range(n))
            families.extend([shard["family"]] * n)
    result = {name: np.concatenate(parts, axis=0) for name, parts in blocks.items()}
    result.update(energy=np.asarray(axis), source_file=np.asarray(source_files),
                  source_row=np.asarray(source_rows), family=np.asarray(families),
                  split=np.asarray(_strings(result["split"])))
    n = len(result["counts"])
    if n != manifest["total_scans"] or result["counts"].shape[1] != 401:
        raise ValueError("Dataset size or energy grid is inconsistent")
    if not np.all(np.isfinite(result["counts"])) or np.any(result["counts"] < 0):
        raise ValueError("Input counts must be finite and nonnegative")
    if not np.all(np.isfinite(result["expected_counts"])):
        raise ValueError("Clean targets contain non-finite values")
    native_steps = np.asarray(result["native_step_ev"], dtype=np.float32)
    allowed_steps = np.array([0.05, 0.10, 0.20], dtype=np.float32)
    if not np.all(np.min(np.abs(native_steps[:, None] - allowed_steps), axis=1) < 0.025):
        raise ValueError("Unsupported native sampling step")
    case_splits = {}
    for case_id, split in zip(result["case_id"], result["split"]):
        old = case_splits.setdefault(int(case_id), split)
        if old != split:
            raise ValueError(f"Case {case_id} occurs in multiple splits")
    for split in ("train", "validation", "test"):
        if not np.any(result["split"] == split):
            raise ValueError(f"Dataset has no {split} cases")
    return result


class SpectrumDataset(Dataset):
    def __init__(self, data: dict, indices: np.ndarray):
        self.data = data
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_index):
        i = int(self.indices[local_index])
        labels = {}
        for name in PEAK_FIELDS + ("instrument_gaussian_fwhm_ev", "background_kind_code", "background_parameters"):
            value = np.asarray(self.data[name][i])
            if name in {"peak_present", "peak_shape_code", "background_kind_code"}:
                labels[name] = torch.as_tensor(value.astype(np.int64))
            else:
                labels[name] = torch.as_tensor(value.astype(np.float32))
        return {
            "counts": torch.as_tensor(self.data["counts"][i], dtype=torch.float32),
            "clean": torch.as_tensor(self.data["expected_counts"][i], dtype=torch.float32),
            "background": torch.as_tensor(self.data["background_counts"][i], dtype=torch.float32),
            "native_step": torch.as_tensor(self.data["native_step_ev"][i], dtype=torch.float32),
            "labels": labels,
            "index": torch.tensor(i, dtype=torch.long),
        }


def make_loaders(data: dict, cfg: RunConfig) -> dict:
    result = {}
    for split in ("train", "validation", "test"):
        idx = np.flatnonzero(data["split"] == split)
        result[split] = DataLoader(
            SpectrumDataset(data, idx), batch_size=cfg.batch_size,
            shuffle=(split == "train"), num_workers=cfg.workers,
            pin_memory=torch.cuda.is_available(),
        )
    return result


def input_features(counts: torch.Tensor, native_step: torch.Tensor):
    """Two local channels plus overall count level and known sampling step."""
    peak = counts.amax(dim=-1, keepdim=True).clamp_min(1.0)
    log_shape = torch.log1p(counts) / torch.log1p(peak)
    sqrt_shape = torch.sqrt(counts + 0.375) / torch.sqrt(peak + 0.375)
    local = torch.stack((log_shape, sqrt_shape), dim=1)
    meta = torch.stack((torch.log1p(peak[:, 0]) / 12.0,
                        native_step / 0.2), dim=-1)
    return local, meta, peak[:, 0]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 5, stride, 2)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 5, 1, 2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels and stride == 1
                     else nn.Conv1d(in_channels, out_channels, 1, stride))

    def forward(self, x):
        return F.silu(self.norm2(self.conv2(F.silu(self.norm1(self.conv1(x)))))
                      + self.skip(x))


class SpectrumCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3), nn.GroupNorm(8, 32), nn.SiLU(),
            ResidualBlock(32, 32), ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64), ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128), nn.AdaptiveAvgPool1d(16), nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.SiLU(),
        )

    def forward(self, local):
        return self.net(local)


def physical_heads(peak_raw: torch.Tensor, global_raw: torch.Tensor,
                   count_scale: torch.Tensor, component_logits=None):
    """Decode unconstrained outputs into physically valid parameter ranges."""
    b, k, _ = peak_raw.shape
    result = {
        "presence_logits": peak_raw[..., 0],
        "shape_logits": peak_raw[..., 1:5],  # shape codes 1..4
        "center": 280.0 + 20.0 * torch.sigmoid(peak_raw[..., 5]),
        "sigma": 0.05 + 2.45 * torch.sigmoid(peak_raw[..., 6]),
        "lorentz_gamma": 0.01 + 0.99 * torch.sigmoid(peak_raw[..., 7]),
        "pseudo_fraction": torch.sigmoid(peak_raw[..., 8]),
        "asymmetry_gamma": 0.30 * torch.sigmoid(peak_raw[..., 9]),
        "area": F.softplus(peak_raw[..., 10]) * count_scale[:, None],
        "background_logits": global_raw[..., :3],
        "background_parameters": torch.stack((
            0.2 * torch.sigmoid(global_raw[..., 3]),
            0.2 * torch.sigmoid(global_raw[..., 4]),
            0.5 * torch.sigmoid(global_raw[..., 5]),
            1000.0 + 4000.0 * torch.sigmoid(global_raw[..., 6]),
        ), dim=-1),
        "instrument_fwhm": 0.1 + 1.2 * torch.sigmoid(global_raw[..., 7]),
    }
    if component_logits is not None:
        result["component_logits"] = component_logits
        # A set query is absent when its seventh class is "no peak".
        result["presence_logits"] = (
            torch.logsumexp(component_logits[..., :6], dim=-1)
            - component_logits[..., 6]
        )
    return result


class PeakCNN(nn.Module):
    """Deterministic six-chemical-slot 1D residual CNN."""
    def __init__(self):
        super().__init__()
        self.encoder = SpectrumCNNEncoder()
        self.shared = nn.Sequential(nn.Linear(258, 256), nn.SiLU(), nn.Dropout(0.1))
        self.peaks = nn.Linear(256, 6 * 11)
        self.global_head = nn.Linear(256, 8)

    def forward(self, counts, native_step, sample=False):
        local, meta, count_scale = input_features(counts, native_step)
        hidden = self.shared(torch.cat((self.encoder(local), meta), dim=-1))
        return physical_heads(self.peaks(hidden).reshape(-1, 6, 11),
                              self.global_head(hidden), count_scale)


class PeakVAE(nn.Module):
    """Supervised VAE: latent -> physical peaks -> rendered clean spectrum."""
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = SpectrumCNNEncoder()
        self.posterior = nn.Sequential(nn.Linear(258, 256), nn.SiLU())
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
        )
        self.peaks = nn.Linear(256, 6 * 11)
        self.global_head = nn.Linear(256, 8)

    def forward(self, counts, native_step, sample=True):
        local, meta, count_scale = input_features(counts, native_step)
        encoded = self.posterior(torch.cat((self.encoder(local), meta), -1))
        mu = self.mu(encoded)
        logvar = self.logvar(encoded).clamp(-10.0, 5.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        hidden = self.decoder(torch.cat((z, meta), -1))
        output = physical_heads(self.peaks(hidden).reshape(-1, 6, 11),
                                self.global_head(hidden), count_scale)
        output.update(mu=mu, logvar=logvar)
        return output


class PeakSetTransformer(nn.Module):
    """Eight unordered peak queries; six chemical classes plus no-peak class."""
    def __init__(self, query_count: int = 8, width: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 64, 7, 2, 3), nn.GroupNorm(8, 64), nn.SiLU(),
            ResidualBlock(64, 64), nn.Conv1d(64, width, 5, 2, 2),
            nn.GroupNorm(8, width), nn.SiLU(),
        )
        self.position = nn.Parameter(torch.zeros(1, 101, width))
        self.meta_projection = nn.Linear(2, width)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, 4, 256, 0.1, batch_first=True,
                                       activation="gelu", norm_first=True), 2,
            enable_nested_tensor=False,
        )
        self.queries = nn.Parameter(torch.randn(query_count, width) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(width, 4, 256, 0.1,
                                       batch_first=True, activation="gelu", norm_first=True), 2
        )
        self.peaks = nn.Linear(width, 11)
        self.components = nn.Linear(width, 7)
        self.global_head = nn.Sequential(nn.Linear(width + 2, width), nn.SiLU(),
                                         nn.Linear(width, 8))
        # TransformerEncoder/Decoder clone their layers. Give each clone
        # independent initial weights rather than identical copies.
        for stack in (self.encoder.layers, self.decoder.layers):
            for layer in stack:
                for parameter in layer.parameters():
                    if parameter.ndim > 1:
                        nn.init.xavier_uniform_(parameter)

    def forward(self, counts, native_step, sample=False):
        local, meta, count_scale = input_features(counts, native_step)
        tokens = self.stem(local).transpose(1, 2)
        if tokens.shape[1] != self.position.shape[1]:
            raise ValueError("Transformer expects the format-5 401-point energy grid")
        memory = self.encoder(tokens + self.position + self.meta_projection(meta)[:, None, :])
        queries = self.queries[None].expand(len(counts), -1, -1)
        decoded = self.decoder(queries, memory)
        global_raw = self.global_head(torch.cat((memory.mean(1), meta), -1))
        return physical_heads(self.peaks(decoded), global_raw, count_scale,
                              component_logits=self.components(decoded))


def build_model(name: str) -> nn.Module:
    return {"cnn": PeakCNN, "vae": PeakVAE,
            "transformer": PeakSetTransformer}[name]()


class C1sRenderer(nn.Module):
    """Differentiable version of the generator's finite-window peak synthesis.

    Voigt is evaluated as a discrete Gaussian/Lorentzian convolution. On this
    0.05 eV grid its normalized error against scipy's Faddeeva Voigt is tiny
    for the generator's width range. No lmfit optimizer is called here.
    """
    def __init__(self, energy: np.ndarray):
        super().__init__()
        energy = np.asarray(energy, dtype=np.float32)
        if (energy.ndim != 1 or len(energy) != 401
                or not np.isclose(energy[0], 280.0)
                or not np.isclose(energy[-1], 300.0)
                or not np.allclose(np.diff(energy), 0.05, atol=5e-5)):
            raise ValueError("Renderer expects the 280-300 eV format-5 grid")
        self.register_buffer("energy", torch.as_tensor(energy))
        self.dx = float((energy[-1] - energy[0]) / (len(energy) - 1))
        self.n = len(energy)
        self.fft_size = 1 << (3 * self.n - 2).bit_length()
        self.causal_fft_size = 1 << (2 * self.n - 1).bit_length()

    def _centered_convolve(self, signal, kernel):
        product = torch.fft.rfft(signal, n=self.fft_size) * torch.fft.rfft(kernel, n=self.fft_size)
        full = torch.fft.irfft(product, n=self.fft_size)
        return full[..., self.n - 1:2 * self.n - 1] * self.dx

    def _unit_profiles(self, pred):
        x = self.energy[None, None, :]
        center = pred["center"][..., None]
        sigma = pred["sigma"][..., None].clamp_min(0.02)
        gamma = pred["lorentz_gamma"][..., None].clamp_min(0.005)
        fraction = pred["pseudo_fraction"][..., None]
        asymmetry = pred["asymmetry_gamma"][..., None]
        diff = x - center
        gaussian = torch.exp(-0.5 * (diff / sigma).square()) / (sigma * math.sqrt(2 * math.pi))

        lag = (torch.arange(-(self.n - 1), self.n, device=x.device,
                            dtype=x.dtype) * self.dx)[None, None, :]
        lorentz_kernel = gamma / (math.pi * (lag.square() + gamma.square()))
        voigt = self._centered_convolve(gaussian, lorentz_kernel)

        # lmfit PseudoVoigtModel's sigma is HWHM, not Gaussian sigma.
        pseudo_gaussian_sigma = sigma / math.sqrt(2.0 * math.log(2.0))
        pseudo_g = torch.exp(-0.5 * (diff / pseudo_gaussian_sigma).square()) / (
            pseudo_gaussian_sigma * math.sqrt(2 * math.pi))
        pseudo_l = sigma / (math.pi * (diff.square() + sigma.square()))
        pseudo = (1.0 - fraction) * pseudo_g + fraction * pseudo_l

        arg = diff / sigma
        one_minus_gamma = 1.0 - asymmetry
        doniach = (
            torch.cos(math.pi * asymmetry / 2.0 + one_minus_gamma * torch.atan(arg))
            / sigma.pow(one_minus_gamma)
            / (1.0 + arg.square()).pow(one_minus_gamma / 2.0)
        )
        instrument_sigma = (pred["instrument_fwhm"][:, None, None]
                            / (2.0 * math.sqrt(2.0 * math.log(2.0)))).clamp_min(0.01)
        instrument_kernel = torch.exp(-0.5 * (lag / instrument_sigma).square())
        instrument_kernel = instrument_kernel / (
            instrument_kernel.sum(dim=-1, keepdim=True) * self.dx
        )
        doniach = self._centered_convolve(doniach, instrument_kernel)

        profiles = torch.stack((voigt, pseudo, doniach, gaussian), dim=-2).clamp_min(0.0)
        areas = torch.trapz(profiles, dx=self.dx, dim=-1).clamp_min(1e-8)
        return profiles / areas[..., None]

    def _backgrounds(self, peak_sum, pred):
        params = pred["background_parameters"]
        peak_max = peak_sum.amax(dim=-1, keepdim=True)
        position = (self.energy - self.energy[0]) / (self.energy[-1] - self.energy[0])
        baseline = peak_max * (params[:, 0, None] * (1.0 - position)
                               + params[:, 1, None] * position)
        increments = 0.5 * (peak_sum[:, 1:] + peak_sum[:, :-1]) * self.dx
        cumulative = F.pad(torch.cumsum(increments, dim=-1), (1, 0))
        shirley = baseline + peak_max * params[:, 2, None] * (
            cumulative / cumulative[:, -1:].clamp_min(1e-8)
        )
        loss_ev = self.energy - self.energy[0]
        kernel = params[:, 3, None] * loss_ev / (1643.0 + loss_ev.square()).square()
        # Generator solves J = peak_sum + K * J. The causal FFT quotient
        # performs that linear solve; samples after the window cannot affect J.
        numerator = torch.fft.rfft(peak_sum, n=self.causal_fft_size)
        denominator = 1.0 - self.dx * torch.fft.rfft(kernel, n=self.causal_fft_size)
        total = torch.fft.irfft(numerator / denominator, n=self.causal_fft_size)[:, :self.n]
        tougaard = baseline + (total - peak_sum).clamp_min(0.0)
        return torch.stack((baseline, shirley, tougaard), dim=1)

    def _resample(self, values, native_step):
        # The generator samples the common 0.05 eV curve at 0.05/0.10/0.20 eV
        # and linearly interpolates it back. Both grids share their endpoints.
        original_shape = values.shape
        flat = values.reshape(values.shape[0], -1, self.n)
        result = flat
        for ratio, step in ((2, 0.10), (4, 0.20)):
            sampled = F.interpolate(flat[..., ::ratio], size=self.n,
                                    mode="linear", align_corners=True)
            mask = (native_step - step).abs() < 0.025
            result = torch.where(mask[:, None, None], sampled, result)
        return result.reshape(original_shape)

    def forward(self, pred, native_step, presence_override=None,
                shape_override=None, background_override=None,
                hard=False, threshold=0.5):
        if hard:
            presence = (torch.sigmoid(pred["presence_logits"]) >= threshold).float()
            shape = F.one_hot(pred["shape_logits"].argmax(-1), 4).float()
            bg_weights = F.one_hot(pred["background_logits"].argmax(-1), 3).float()
        else:
            presence = torch.sigmoid(pred["presence_logits"])
            shape = torch.softmax(pred["shape_logits"], dim=-1)
            bg_weights = torch.softmax(pred["background_logits"], dim=-1)
        if presence_override is not None:
            presence = presence_override
        if shape_override is not None:
            shape = shape_override
        if background_override is not None:
            bg_weights = background_override
        unit = self._unit_profiles(pred)
        selected = (unit * shape[..., None]).sum(dim=-2)
        components = selected * pred["area"][..., None] * presence[..., None]
        peak_sum = components.sum(dim=1)
        backgrounds = self._backgrounds(peak_sum, pred)
        background = (backgrounds * bg_weights[..., None]).sum(dim=1)
        return {
            "clean": self._resample(peak_sum + background, native_step),
            "background": self._resample(background, native_step),
            "components": self._resample(components, native_step),
        }


def truth_prediction(labels):
    """Put ground-truth parameters through the same renderer for an audit."""
    p = labels["background_parameters"]
    return {
        "presence_logits": torch.where(labels["peak_present"].bool(), 30.0, -30.0),
        "shape_logits": F.one_hot((labels["peak_shape_code"] - 1).clamp(0, 3), 4).float() * 30.0,
        "center": torch.nan_to_num(labels["peak_center_ev"], nan=284.8),
        "sigma": torch.nan_to_num(labels["peak_model_sigma_ev"], nan=0.5),
        "lorentz_gamma": torch.nan_to_num(labels["peak_lorentz_gamma_ev"], nan=0.15),
        "pseudo_fraction": torch.nan_to_num(labels["peak_pseudo_voigt_fraction"], nan=0.3),
        "asymmetry_gamma": torch.nan_to_num(labels["peak_asymmetry_gamma"], nan=0.08),
        "area": labels["peak_area_counts_ev"],
        "background_logits": F.one_hot(labels["background_kind_code"], 3).float() * 30.0,
        "background_parameters": torch.stack((
            p[:, 0], p[:, 1], torch.nan_to_num(p[:, 2], nan=0.1),
            torch.nan_to_num(p[:, 3], nan=2866.0),
        ), dim=-1),
        "instrument_fwhm": labels["instrument_gaussian_fwhm_ev"],
    }


@torch.no_grad()
def audit_renderer(loader, renderer, device, max_batches=2):
    errors, bg_errors = [], []
    for batch_number, batch in enumerate(loader):
        if batch_number >= max_batches:
            break
        labels = {k: v.to(device) for k, v in batch["labels"].items()}
        step = batch["native_step"].to(device)
        curves = renderer(truth_prediction(labels), step, hard=True)
        clean = batch["clean"].to(device)
        background = batch["background"].to(device)
        scale = clean.amax(-1).clamp_min(1.0)
        errors.extend((((curves["clean"] - clean).square().mean(-1).sqrt()
                        / scale).cpu().tolist()))
        bg_errors.extend((((curves["background"] - background).square().mean(-1).sqrt()
                           / scale).cpu().tolist()))
    if not errors:
        raise ValueError("Empty renderer audit")
    report = {"median_clean_nrmse": float(np.median(errors)),
              "max_clean_nrmse": float(np.max(errors)),
              "max_background_nrmse": float(np.max(bg_errors))}
    if report["max_clean_nrmse"] > 0.005 or report["max_background_nrmse"] > 0.005:
        raise ValueError(f"Renderer disagrees with generated labels: {report}")
    return report


def _target_assignment(pred, labels):
    """Align true chemical peaks to output slots or unordered set queries."""
    b, k = pred["center"].shape
    if "component_logits" not in pred:
        if k != 6:
            raise ValueError("Named-slot models require six peak outputs")
        return torch.arange(6, device=pred["center"].device)[None].expand(b, -1)
    with torch.no_grad():
        component_logp = F.log_softmax(pred["component_logits"], -1)
        shape_logp = F.log_softmax(pred["shape_logits"], -1)
        shape_ids = (labels["peak_shape_code"] - 1).clamp(0, 3)
        shape_cost = shape_logp.gather(2, shape_ids[:, None, :].expand(-1, k, -1))
        center = torch.nan_to_num(labels["peak_center_ev"], nan=0.0)
        area = torch.nan_to_num(labels["peak_area_counts_ev"], nan=0.0)
        cost = (
            -component_logp[:, :, :6]
            - 0.5 * shape_cost
            + 2.0 * (pred["center"][:, :, None] - center[:, None, :]).abs() / 0.5
            + 0.4 * (torch.log1p(pred["area"][:, :, None])
                     - torch.log1p(area[:, None, :])).abs()
        )
        # Transfer one cost table for the whole batch. The old row-by-row
        # transfers synchronized the GPU once for every spectrum.
        cost_cpu = cost.cpu().numpy()
        present_cpu = labels["peak_present"].cpu().numpy().astype(bool)
        assignments = np.full((b, k), -1, dtype=np.int64)
        for row in range(b):
            true_slots = np.flatnonzero(present_cpu[row])
            if len(true_slots):
                output_rows, target_cols = linear_sum_assignment(cost_cpu[row][:, true_slots])
                assignments[row, output_rows] = true_slots[target_cols]
    return torch.as_tensor(assignments, device=pred["center"].device)


def _masked_huber(predicted, target, mask, scale):
    target = torch.nan_to_num(target, nan=0.0)
    error = F.smooth_l1_loss((predicted - target) / scale, torch.zeros_like(predicted),
                            reduction="none", beta=1.0)
    weights = mask.to(error.dtype)
    return (error * weights).sum() / weights.sum().clamp_min(1.0)


def _aligned_targets(labels, assignments):
    safe = assignments.clamp_min(0)
    gathered = {}
    for name in PEAK_FIELDS:
        gathered[name] = labels[name].gather(1, safe)
    gathered["peak_present"] = gathered["peak_present"].bool() & (assignments >= 0)
    return gathered


def compute_train_weights(data):
    train = data["split"] == "train"
    present = data["peak_present"][train].astype(bool)
    frequency = present.mean(axis=0)
    positive = np.clip((1.0 - frequency) / np.maximum(frequency, 1e-5), 1.0, 6.0)
    shapes = data["peak_shape_code"][train][present]
    shape_frequency = np.bincount(shapes, minlength=5)[1:].astype(float)
    shape_weight = np.clip(shape_frequency.sum() /
                           (4.0 * np.maximum(shape_frequency, 1)), 0.5, 3.0)
    return {"positive": torch.tensor(positive, dtype=torch.float32),
            "shape": torch.tensor(shape_weight, dtype=torch.float32)}


def loss_terms(pred, batch, renderer, cfg: RunConfig, class_weights,
               teacher_ratio=0.0, beta=0.0):
    labels = batch["labels"]
    assignments = _target_assignment(pred, labels)
    target = _aligned_targets(labels, assignments)
    present = target["peak_present"]
    shape_ids = (target["peak_shape_code"] - 1).clamp(0, 3)
    if "component_logits" in pred:
        component_target = torch.where(present, assignments, 6)
        class_weight = torch.tensor([1.0] * 6 + [0.35],
                                    device=pred["center"].device)
        detection = F.cross_entropy(pred["component_logits"].reshape(-1, 7),
                                    component_target.reshape(-1), weight=class_weight)
    else:
        detection = F.binary_cross_entropy_with_logits(
            pred["presence_logits"], present.float(),
            pos_weight=class_weights["positive"].to(pred["center"].device),
        )
    shape_weights = class_weights["shape"].to(pred["center"].device)
    shape_losses = F.cross_entropy(pred["shape_logits"].reshape(-1, 4),
                                   shape_ids.reshape(-1), weight=shape_weights,
                                   reduction="none").reshape_as(shape_ids)
    shape = ((shape_losses * present).sum()
             / (shape_weights[shape_ids] * present).sum().clamp_min(1e-12))

    scale = batch["counts"].amax(-1).clamp_min(1.0)[:, None]
    area_log = torch.log1p(pred["area"] / scale)
    target_area_log = torch.log1p(target["peak_area_counts_ev"] / scale)
    parameter = {
        "center": _masked_huber(pred["center"], target["peak_center_ev"], present, 0.2),
        "sigma": _masked_huber(pred["sigma"], target["peak_model_sigma_ev"], present, 0.2),
        "area": _masked_huber(area_log, target_area_log, present, 0.25),
    }
    code = target["peak_shape_code"]
    parameter["lorentz_gamma"] = _masked_huber(
        pred["lorentz_gamma"], target["peak_lorentz_gamma_ev"], present & (code == 1), 0.1)
    parameter["pseudo_fraction"] = _masked_huber(
        pred["pseudo_fraction"], target["peak_pseudo_voigt_fraction"], present & (code == 2), 0.2)
    parameter["asymmetry_gamma"] = _masked_huber(
        pred["asymmetry_gamma"], target["peak_asymmetry_gamma"], present & (code == 3), 0.05)
    peak_parameters = sum(parameter.values())

    bg_kind = labels["background_kind_code"]
    background_class = F.cross_entropy(pred["background_logits"], bg_kind)
    bg_true = labels["background_parameters"]
    background_params = (
        _masked_huber(pred["background_parameters"][:, :2], bg_true[:, :2],
                      torch.ones_like(bg_true[:, :2], dtype=torch.bool), 0.05)
        + _masked_huber(pred["background_parameters"][:, 2], bg_true[:, 2],
                        bg_kind == 1, 0.1)
        + _masked_huber(pred["background_parameters"][:, 3], bg_true[:, 3],
                        bg_kind == 2, 1000.0)
    )
    instrument = _masked_huber(pred["instrument_fwhm"],
                               labels["instrument_gaussian_fwhm_ev"],
                               torch.ones_like(bg_kind, dtype=torch.bool), 0.25)

    if teacher_ratio:
        presence = teacher_ratio * present.float() + (1.0 - teacher_ratio) * torch.sigmoid(pred["presence_logits"])
        shape_mix = (teacher_ratio * F.one_hot(shape_ids, 4).float()
                     + (1.0 - teacher_ratio) * torch.softmax(pred["shape_logits"], -1))
        background_mix = (
            teacher_ratio * F.one_hot(bg_kind, 3).float()
            + (1.0 - teacher_ratio) * torch.softmax(pred["background_logits"], -1)
        )
    else:
        presence = shape_mix = background_mix = None
    rendered = renderer(pred, batch["native_step"], presence, shape_mix, background_mix)
    target_clean = batch["clean"]
    curve_scale = target_clean.amax(-1, keepdim=True).clamp_min(1.0)
    normalized_error = (rendered["clean"] - target_clean) / curve_scale
    spectrum = F.smooth_l1_loss(normalized_error / 0.05,
                                torch.zeros_like(normalized_error), beta=1.0)
    # First-difference error emphasizes peak shoulders and widths.
    slope_error = (normalized_error[:, 1:] - normalized_error[:, :-1]) / 0.025
    spectrum = spectrum + 0.25 * F.smooth_l1_loss(
        slope_error, torch.zeros_like(slope_error), beta=1.0)
    background_error = (rendered["background"] - batch["background"]) / curve_scale
    background_curve = F.smooth_l1_loss(background_error / 0.05,
                                        torch.zeros_like(background_error), beta=1.0)
    if "mu" in pred:
        kl_by_dim = -0.5 * (1 + pred["logvar"] - pred["mu"].square()
                              - pred["logvar"].exp()).mean(0)
        kl = kl_by_dim.mean()
        kl_penalty = kl_by_dim.clamp_min(0.02).mean()  # free bits
    else:
        kl = spectrum.new_zeros(())
        kl_penalty = kl
    total = (
        cfg.spectrum_weight * spectrum
        + cfg.background_curve_weight * background_curve
        + cfg.detection_weight * detection
        + cfg.shape_weight * shape
        + cfg.peak_parameter_weight * peak_parameters
        + cfg.background_parameter_weight * (background_class + background_params)
        + cfg.instrument_weight * instrument
        + beta * kl_penalty
    )
    return {
        "total": total, "spectrum": spectrum, "background_curve": background_curve,
        "detection": detection, "shape": shape,
        "background_class": background_class, "background_parameters": background_params,
        "instrument": instrument, "kl": kl, **parameter,
    }


def _move_batch(batch, device):
    return {
        "counts": batch["counts"].to(device, non_blocking=True),
        "clean": batch["clean"].to(device, non_blocking=True),
        "background": batch["background"].to(device, non_blocking=True),
        "native_step": batch["native_step"].to(device, non_blocking=True),
        "index": batch["index"],
        "labels": {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()},
    }


def _epoch(model, renderer, loader, cfg, weights, optimizer, device,
           teacher_ratio=0.0, beta=0.0):
    training = optimizer is not None
    model.train(training)
    names, sums, scans = None, None, 0
    for source_batch in loader:
        batch = _move_batch(source_batch, device)
        with torch.set_grad_enabled(training):
            pred = model(batch["counts"], batch["native_step"], sample=training)
            terms = loss_terms(pred, batch, renderer, cfg, weights,
                               teacher_ratio=teacher_ratio, beta=beta)
            if training:
                optimizer.zero_grad(set_to_none=True)
                terms["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        n = len(batch["counts"])
        scans += n
        if names is None:
            names = tuple(terms)
            sums = torch.zeros(len(names), device=device, dtype=torch.float32)
        # Keep metrics on the device until the epoch ends. Converting each
        # scalar to float used to synchronize CUDA once per loss term/batch.
        sums += torch.stack([terms[name].detach() for name in names]) * n
    if not scans:
        raise ValueError("Empty data split")
    return dict(zip(names, (sums / scans).cpu().tolist()))


def _save_checkpoint(payload, path):
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_model(model, renderer, loaders, data, cfg: RunConfig,
                manifest_sha: str, device):
    """Run only when a notebook explicitly calls this function."""
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights = compute_train_weights(data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    best, bad_epochs, start_epoch = float("inf"), 0, 0
    history = []
    latest = output / "latest.pt"
    if not cfg.resume and (latest.is_file() or (output / "best.pt").is_file()):
        raise FileExistsError("This output directory already has checkpoints; choose a new output_dir for a new experiment")
    if cfg.resume and latest.is_file():
        checkpoint = torch.load(latest, map_location=device, weights_only=True)
        if checkpoint["manifest_sha"] != manifest_sha or checkpoint["model_name"] != cfg.model_name:
            raise ValueError("Existing checkpoint belongs to another dataset or model")
        allowed_changes = {"epochs", "patience", "resume", "wait_hours", "poll_seconds"}
        changed = [key for key, value in asdict(cfg).items()
                   if key not in allowed_changes and checkpoint["config"].get(key) != value]
        if changed:
            raise ValueError(f"Training settings changed since checkpoint ({changed}); choose a new output_dir")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best = float(checkpoint["best"])
        bad_epochs = int(checkpoint["bad_epochs"])
        start_epoch = int(checkpoint["epoch"]) + 1
        history = checkpoint["history"]
        print(f"Resuming {cfg.model_name} at epoch {start_epoch + 1}")
    for epoch in range(start_epoch, cfg.epochs):
        teacher_ratio = max(0.0, 1.0 - epoch / max(1, cfg.teacher_forcing_epochs))
        beta = (cfg.vae_beta_max * min(1.0, (epoch + 1) / max(1, cfg.vae_warmup_epochs))
                if cfg.model_name == "vae" else 0.0)
        train_started = time.perf_counter()
        train = _epoch(model, renderer, loaders["train"], cfg, weights, optimizer,
                       device, teacher_ratio, beta)
        train_seconds = time.perf_counter() - train_started
        validation_started = time.perf_counter()
        validation = _epoch(model, renderer, loaders["validation"], cfg, weights,
                            None, device, 0.0, cfg.vae_beta_max if cfg.model_name == "vae" else 0.0)
        validation_seconds = time.perf_counter() - validation_started
        scheduler.step(validation["total"])
        row = {"epoch": epoch + 1, "teacher_ratio": teacher_ratio,
               "vae_beta": beta, "learning_rate": optimizer.param_groups[0]["lr"],
               "train_seconds": train_seconds, "validation_seconds": validation_seconds}
        row.update({f"train_{k}": v for k, v in train.items()})
        row.update({f"validation_{k}": v for k, v in validation.items()})
        history.append(row)
        checkpoint_started = time.perf_counter()
        improved = validation["total"] < best - 1e-5
        if improved:
            best = validation["total"]
            bad_epochs = 0
            _save_checkpoint({"model": model.state_dict(), "model_name": cfg.model_name,
                              "manifest_sha": manifest_sha, "epoch": epoch,
                              "validation_total": best, "config": asdict(cfg)},
                             output / "best.pt")
        else:
            bad_epochs += 1
        _save_checkpoint({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "model_name": cfg.model_name,
                          "manifest_sha": manifest_sha, "epoch": epoch,
                          "best": best, "bad_epochs": bad_epochs, "history": history,
                          "config": asdict(cfg)}, latest)
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        print(f"{cfg.model_name} epoch {epoch + 1}/{cfg.epochs}: "
              f"train={train['total']:.4f}, val={validation['total']:.4f}, "
              f"peak detection={validation['detection']:.4f}, "
              f"shape={validation['shape']:.4f}, "
              f"curve={validation['spectrum']:.4f}, best={best:.4f}, "
              f"time: train={train_seconds:.1f}s, validation={validation_seconds:.1f}s, "
              f"checkpoint={checkpoint_seconds:.1f}s", flush=True)
        if bad_epochs >= cfg.patience:
            print(f"Early stopping after {bad_epochs} epochs without improvement")
            break
    if not (output / "best.pt").is_file():
        raise RuntimeError("No best checkpoint was saved")
    best_checkpoint = torch.load(output / "best.pt", map_location=device,
                                 weights_only=True)
    model.load_state_dict(best_checkpoint["model"])
    return pd.DataFrame(history), best_checkpoint


def _fwhm(energy, curve):
    if not np.any(curve > 0):
        return float("nan")
    maximum = int(np.argmax(curve))
    half = curve[maximum] / 2.0
    left = np.flatnonzero(curve[:maximum] <= half)
    right = np.flatnonzero(curve[maximum + 1:] <= half)
    if not len(left) or not len(right):
        return float("nan")
    lo, hi = int(left[-1]), int(maximum + 1 + right[0])
    x_left = np.interp(half, curve[lo:lo + 2], energy[lo:lo + 2])
    x_right = np.interp(half, curve[hi - 1:hi + 1][::-1],
                        energy[hi - 1:hi + 1][::-1])
    return float(x_right - x_left)


def _decode_batch(pred):
    """Detach just the small parameter tables used for metrics and plots."""
    fields = ("center", "sigma", "lorentz_gamma", "pseudo_fraction",
              "asymmetry_gamma", "area", "background_parameters", "instrument_fwhm")
    result = {key: pred[key].detach().cpu().numpy() for key in fields}
    result["presence_prob"] = torch.sigmoid(pred["presence_logits"]).detach().cpu().numpy()
    result["shape_code"] = (pred["shape_logits"].argmax(-1) + 1).detach().cpu().numpy()
    result["background_code"] = pred["background_logits"].argmax(-1).detach().cpu().numpy()
    if "component_logits" in pred:
        result["component_code"] = pred["component_logits"][..., :6].argmax(-1).detach().cpu().numpy()
    else:
        result["component_code"] = np.broadcast_to(np.arange(6), result["center"].shape)
    return result


@torch.no_grad()
def choose_presence_threshold(model, loader, device):
    """Select threshold on validation only, using chemical-identity F1."""
    model.eval()
    candidates = np.round(np.arange(0.20, 0.81, 0.05), 2)
    totals = {float(t): np.zeros(3, dtype=int) for t in candidates}
    for source_batch in loader:
        batch = _move_batch(source_batch, device)
        decoded = _decode_batch(model(batch["counts"], batch["native_step"], sample=False))
        truth = batch["labels"]["peak_present"].cpu().numpy().astype(bool)
        for row in range(len(truth)):
            true_classes = set(np.flatnonzero(truth[row]).tolist())
            for threshold in candidates:
                chosen = decoded["presence_prob"][row] >= threshold
                predicted = decoded["component_code"][row][chosen].tolist()
                # Each true chemical slot occurs at most once. Duplicates count as FP.
                tp = sum(1 for component in true_classes if component in predicted)
                fp = len(predicted) - tp
                fn = len(true_classes) - tp
                totals[float(threshold)] += (tp, fp, fn)
    scores = []
    for threshold in candidates:
        tp, fp, fn = totals[float(threshold)]
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        scores.append({"threshold": float(threshold), "f1": float(f1),
                       "tp": int(tp), "fp": int(fp), "fn": int(fn)})
    table = pd.DataFrame(scores)
    # Prefer 0.5 on a tie; only validation is involved in threshold selection.
    table["distance_from_half"] = (table["threshold"] - 0.5).abs()
    table = table.sort_values(["f1", "distance_from_half"], ascending=[False, True])
    return float(table.iloc[0]["threshold"]), table.drop(columns="distance_from_half")


def _relative_error(predicted, target):
    return abs(float(predicted) - float(target)) / max(abs(float(target)), 1e-9)


def _value_or_nan(value):
    return float(value) if np.isfinite(value) else float("nan")


@torch.no_grad()
def evaluate_model(model, renderer, loader, data, cfg: RunConfig,
                   threshold: float, device, split="test", manifest_sha=""):
    """Save scan, true-peak, predicted-peak, subgroup and gallery reports."""
    model.eval()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scan_rows, true_rows, predicted_rows = [], [], []
    energy = data["energy"]
    train_max = data["counts"][data["split"] == "train"].max(axis=1)
    count_edges = np.quantile(train_max, [0.25, 0.5, 0.75])
    for source_batch in loader:
        batch = _move_batch(source_batch, device)
        pred = model(batch["counts"], batch["native_step"], sample=False)
        decoded = _decode_batch(pred)
        curves = renderer(pred, batch["native_step"], hard=True, threshold=threshold)
        truth_curves = renderer(truth_prediction(batch["labels"]),
                                batch["native_step"], hard=True)
        predicted_clean = curves["clean"].cpu().numpy()
        predicted_bg = curves["background"].cpu().numpy()
        predicted_components = curves["components"].cpu().numpy()
        true_components = truth_curves["components"].cpu().numpy()
        for local, global_index in enumerate(batch["index"].numpy()):
            index = int(global_index)
            true_slots = np.flatnonzero(data["peak_present"][index]).tolist()
            chosen = np.flatnonzero(decoded["presence_prob"][local] >= threshold).tolist()
            matched = {}
            used_queries = set()
            for slot in true_slots:
                candidates = [q for q in chosen if q not in used_queries
                              and int(decoded["component_code"][local, q]) == slot]
                if candidates:
                    query = min(candidates, key=lambda q: abs(
                        float(decoded["center"][local, q])
                        - float(data["peak_center_ev"][index, slot])))
                    matched[slot] = query
                    used_queries.add(query)
            fp = len(chosen) - len(matched)
            fn = len(true_slots) - len(matched)
            background_code = int(data["background_kind_code"][index])
            true_background_parameters = data["background_parameters"][index]
            predicted_background_parameters = decoded["background_parameters"][local]
            scale = max(float(data["expected_counts"][index].max()), 1.0)
            clean_nrmse = float(np.sqrt(np.mean((predicted_clean[local]
                                                - data["expected_counts"][index]) ** 2)) / scale)
            bg_nrmse = float(np.sqrt(np.mean((predicted_bg[local]
                                             - data["background_counts"][index]) ** 2)) / scale)
            count_bin = int(np.searchsorted(count_edges, data["counts"][index].max())) + 1
            identity_shape_errors = 0
            center_failures = 0
            for slot in true_slots:
                query = matched.get(slot)
                present_match = query is not None
                true_shape = int(data["peak_shape_code"][index, slot])
                true_center = float(data["peak_center_ev"][index, slot])
                true_sigma = float(data["peak_model_sigma_ev"][index, slot])
                true_area = float(data["peak_area_counts_ev"][index, slot])
                true_curve = true_components[local, slot]
                true_height = float(true_curve.max())
                true_width = _fwhm(energy, true_curve)
                neighbors = [abs(true_center - float(data["peak_center_ev"][index, other]))
                             for other in true_slots if other != slot]
                nearest = min(neighbors) if neighbors else float("nan")
                total_area = float(data["peak_area_counts_ev"][index].sum())
                area_fraction = true_area / max(total_area, 1e-9)
                record = {
                    "index": index, "case_id": int(data["case_id"][index]),
                    "source_file": str(data["source_file"][index]),
                    "source_row": int(data["source_row"][index]),
                    "family": str(data["family"][index]),
                    "background": BACKGROUND_NAMES[background_code],
                    "n_true_peaks": len(true_slots), "count_quartile": count_bin,
                    "component": COMPONENT_NAMES[slot], "true_shape": SHAPE_NAMES[true_shape],
                    "true_center_ev": true_center, "true_sigma_ev": true_sigma,
                    "true_area_counts_ev": true_area,
                    "true_height_counts": true_height, "true_fwhm_ev": true_width,
                    "area_fraction": area_fraction, "nearest_peak_ev": nearest,
                    "area_group": ("under_5_percent" if area_fraction < 0.05 else
                                   "5_to_20_percent" if area_fraction < 0.20 else "20_percent_or_more"),
                    "overlap_group": ("single_peak" if not neighbors else
                                      "under_0.5_eV" if nearest < 0.5 else "0.5_eV_or_more"),
                    "detected": present_match,
                }
                if present_match:
                    prediction_curve = predicted_components[local, query]
                    pred_center = float(decoded["center"][local, query])
                    pred_shape = int(decoded["shape_code"][local, query])
                    pred_sigma = float(decoded["sigma"][local, query])
                    pred_area = float(decoded["area"][local, query])
                    predicted_total_area = sum(float(decoded["area"][local, q])
                                               for q in chosen)
                    predicted_area_fraction = pred_area / max(predicted_total_area, 1e-9)
                    pred_height = float(prediction_curve.max())
                    pred_width = _fwhm(energy, prediction_curve)
                    record.update(
                        predicted_shape=SHAPE_NAMES[pred_shape],
                        predicted_center_ev=pred_center, predicted_sigma_ev=pred_sigma,
                        predicted_area_counts_ev=pred_area,
                        predicted_area_fraction=predicted_area_fraction,
                        predicted_height_counts=pred_height, predicted_fwhm_ev=pred_width,
                        presence_probability=float(decoded["presence_prob"][local, query]),
                        center_error_ev=abs(pred_center - true_center),
                        sigma_error_ev=abs(pred_sigma - true_sigma),
                        lorentz_gamma_error_ev=(abs(float(decoded["lorentz_gamma"][local, query])
                                                    - float(data["peak_lorentz_gamma_ev"][index, slot]))
                                                 if true_shape == 1 else np.nan),
                        pseudo_fraction_error=(abs(float(decoded["pseudo_fraction"][local, query])
                                                   - float(data["peak_pseudo_voigt_fraction"][index, slot]))
                                               if true_shape == 2 else np.nan),
                        asymmetry_gamma_error=(abs(float(decoded["asymmetry_gamma"][local, query])
                                                   - float(data["peak_asymmetry_gamma"][index, slot]))
                                               if true_shape == 3 else np.nan),
                        area_relative_error=_relative_error(pred_area, true_area),
                        area_fraction_absolute_error=abs(predicted_area_fraction - area_fraction),
                        height_relative_error=_relative_error(pred_height, true_height),
                        fwhm_error_ev=abs(pred_width - true_width),
                        shape_correct=pred_shape == true_shape,
                        sigma_within_0p2_ev=abs(pred_sigma - true_sigma) <= 0.2,
                        lorentz_gamma_within_0p05_ev=(abs(float(decoded["lorentz_gamma"][local, query])
                                                         - float(data["peak_lorentz_gamma_ev"][index, slot])) <= 0.05
                                                      if true_shape == 1 else np.nan),
                        pseudo_fraction_within_0p1=(abs(float(decoded["pseudo_fraction"][local, query])
                                                       - float(data["peak_pseudo_voigt_fraction"][index, slot])) <= 0.1
                                                    if true_shape == 2 else np.nan),
                        asymmetry_gamma_within_0p03=(abs(float(decoded["asymmetry_gamma"][local, query])
                                                       - float(data["peak_asymmetry_gamma"][index, slot])) <= 0.03
                                                    if true_shape == 3 else np.nan),
                        center_within_0p2_ev=abs(pred_center - true_center) <= 0.2,
                        fwhm_within_0p2_ev=abs(pred_width - true_width) <= 0.2,
                        height_within_20_percent=_relative_error(pred_height, true_height) <= 0.2,
                        area_within_20_percent=_relative_error(pred_area, true_area) <= 0.2,
                    )
                    identity_shape_errors += int(pred_shape != true_shape)
                    center_failures += int(abs(pred_center - true_center) > 0.2)
                else:
                    record.update(predicted_shape="missed", predicted_center_ev=np.nan,
                                  predicted_sigma_ev=np.nan, predicted_area_counts_ev=np.nan,
                                  predicted_area_fraction=np.nan, area_fraction_absolute_error=np.nan,
                                  predicted_height_counts=np.nan, predicted_fwhm_ev=np.nan,
                                  presence_probability=np.nan, center_error_ev=np.nan,
                                  sigma_error_ev=np.nan, lorentz_gamma_error_ev=np.nan,
                                  pseudo_fraction_error=np.nan, asymmetry_gamma_error=np.nan,
                                  area_relative_error=np.nan,
                                  height_relative_error=np.nan, fwhm_error_ev=np.nan,
                                  shape_correct=False, center_within_0p2_ev=False,
                                  sigma_within_0p2_ev=False,
                                  lorentz_gamma_within_0p05_ev=(False if true_shape == 1 else np.nan),
                                  pseudo_fraction_within_0p1=(False if true_shape == 2 else np.nan),
                                  asymmetry_gamma_within_0p03=(False if true_shape == 3 else np.nan),
                                  fwhm_within_0p2_ev=False,
                                  height_within_20_percent=False,
                                  area_within_20_percent=False)
                true_rows.append(record)
            for query in chosen:
                slot = int(decoded["component_code"][local, query])
                linked = next((s for s, q in matched.items() if q == query), None)
                predicted_rows.append({
                    "index": index, "case_id": int(data["case_id"][index]),
                    "source_file": str(data["source_file"][index]),
                    "source_row": int(data["source_row"][index]),
                    "family": str(data["family"][index]),
                    "query": query, "component": COMPONENT_NAMES[slot],
                    "matched_true_component": COMPONENT_NAMES[linked] if linked is not None else "",
                    "false_positive": linked is None,
                    "presence_probability": float(decoded["presence_prob"][local, query]),
                    "shape": SHAPE_NAMES[int(decoded["shape_code"][local, query])],
                    "center_ev": float(decoded["center"][local, query]),
                    "sigma_ev": float(decoded["sigma"][local, query]),
                    "area_counts_ev": float(decoded["area"][local, query]),
                    "lorentz_gamma_ev": float(decoded["lorentz_gamma"][local, query]),
                    "pseudo_fraction": float(decoded["pseudo_fraction"][local, query]),
                    "asymmetry_gamma": float(decoded["asymmetry_gamma"][local, query]),
                })
            score = (2.0 * fn + 2.0 * fp + identity_shape_errors
                     + center_failures + 10.0 * clean_nrmse)
            scan_rows.append({
                "index": index, "case_id": int(data["case_id"][index]),
                "source_file": str(data["source_file"][index]),
                "source_row": int(data["source_row"][index]),
                "family": str(data["family"][index]),
                "background": BACKGROUND_NAMES[background_code],
                "predicted_background": BACKGROUND_NAMES[int(decoded["background_code"][local])],
                "background_correct": int(decoded["background_code"][local]) == background_code,
                "n_true_peaks": len(true_slots), "n_predicted_peaks": len(chosen),
                "true_positive": len(matched), "false_positive": fp, "false_negative": fn,
                "clean_nrmse": clean_nrmse, "background_nrmse": bg_nrmse,
                "count_quartile": count_bin, "difficulty_score": score,
                "background_low_fraction_error": abs(float(predicted_background_parameters[0])
                                                     - float(true_background_parameters[0])),
                "background_high_fraction_error": abs(float(predicted_background_parameters[1])
                                                      - float(true_background_parameters[1])),
                "background_shirley_fraction_error": (
                    abs(float(predicted_background_parameters[2]) - float(true_background_parameters[2]))
                    if background_code == 1 else np.nan),
                "background_tougaard_B_error_ev2": (
                    abs(float(predicted_background_parameters[3]) - float(true_background_parameters[3]))
                    if background_code == 2 else np.nan),
                "instrument_error_ev": abs(float(decoded["instrument_fwhm"][local])
                                           - float(data["instrument_gaussian_fwhm_ev"][index])),
            })
    scans = pd.DataFrame(scan_rows)
    true_peaks = pd.DataFrame(true_rows)
    predicted_peaks = pd.DataFrame(predicted_rows)
    scans.to_csv(output / f"{split}_scans.csv", index=False)
    true_peaks.to_csv(output / f"{split}_true_peaks.csv", index=False)
    predicted_peaks.to_csv(output / f"{split}_predicted_peaks.csv", index=False)
    summary = _summarize(scans, true_peaks)
    summary.update(model=cfg.model_name, split=split, threshold=threshold,
                   manifest_sha=manifest_sha,
                   scans=int(len(scans)), true_peaks=int(len(true_peaks)))
    (output / f"{split}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    breakdown = _breakdown(scans, true_peaks)
    breakdown.to_csv(output / f"{split}_breakdown.csv", index=False)
    cases = scans.groupby("case_id", as_index=False).agg(
        family=("family", "first"), scans=("index", "size"),
        mean_difficulty=("difficulty_score", "mean"),
        mean_clean_nrmse=("clean_nrmse", "mean"),
        missed_peaks=("false_negative", "sum"),
        extra_peaks=("false_positive", "sum"),
    ).sort_values("mean_difficulty")
    cases.to_csv(output / f"{split}_cases_ranked.csv", index=False)
    if split == "test":
        _make_figures(model, renderer, data, scans, true_peaks, cases,
                      summary, threshold, device, output)
    return summary, scans, true_peaks, cases


def _summarize(scans, peaks):
    tp = int(scans["true_positive"].sum())
    fp = int(scans["false_positive"].sum())
    fn = int(scans["false_negative"].sum())
    def finite(name):
        return float(peaks.loc[peaks["detected"], name].median())

    def shape_success(code, name):
        return float(peaks.loc[peaks["true_shape"] == code, name].mean())
    return {
        "presence_precision": tp / max(tp + fp, 1),
        "presence_recall": tp / max(tp + fn, 1),
        "presence_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "shape_accuracy_all_true": float(peaks["shape_correct"].mean()),
        "shape_accuracy_detected": float(peaks.loc[peaks["detected"], "shape_correct"].mean()),
        "center_within_0p2_ev_all_true": float(peaks["center_within_0p2_ev"].mean()),
        "sigma_within_0p2_ev_all_true": float(peaks["sigma_within_0p2_ev"].mean()),
        "fwhm_within_0p2_ev_all_true": float(peaks["fwhm_within_0p2_ev"].mean()),
        "height_within_20_percent_all_true": float(peaks["height_within_20_percent"].mean()),
        "area_within_20_percent_all_true": float(peaks["area_within_20_percent"].mean()),
        "voigt_gamma_within_0p05_ev_all_true": shape_success("voigt", "lorentz_gamma_within_0p05_ev"),
        "pseudo_fraction_within_0p1_all_true": shape_success("pseudo_voigt", "pseudo_fraction_within_0p1"),
        "doniach_asymmetry_within_0p03_all_true": shape_success("doniach_gaussian", "asymmetry_gamma_within_0p03"),
        "median_center_error_ev_detected": finite("center_error_ev"),
        "median_sigma_error_ev_detected": finite("sigma_error_ev"),
        "median_lorentz_gamma_error_ev_detected": finite("lorentz_gamma_error_ev"),
        "median_pseudo_fraction_error_detected": finite("pseudo_fraction_error"),
        "median_asymmetry_gamma_error_detected": finite("asymmetry_gamma_error"),
        "median_fwhm_error_ev_detected": finite("fwhm_error_ev"),
        "median_height_relative_error_detected": finite("height_relative_error"),
        "median_area_relative_error_detected": finite("area_relative_error"),
        "median_clean_nrmse": float(scans["clean_nrmse"].median()),
        "mean_clean_nrmse": float(scans["clean_nrmse"].mean()),
        "median_background_nrmse": float(scans["background_nrmse"].median()),
        "background_kind_accuracy": float(scans["background_correct"].mean()),
        "median_background_low_fraction_error": float(scans["background_low_fraction_error"].median()),
        "median_background_high_fraction_error": float(scans["background_high_fraction_error"].median()),
        "median_shirley_fraction_error": float(scans["background_shirley_fraction_error"].median()),
        "median_tougaard_B_error_ev2": float(scans["background_tougaard_B_error_ev2"].median()),
        "median_instrument_error_ev": float(scans["instrument_error_ev"].median()),
    }


def _breakdown(scans, peaks):
    rows = []
    for name, table, groups in (
        ("scan", scans, ("family", "background", "n_true_peaks", "count_quartile")),
        ("peak", peaks, ("family", "background", "n_true_peaks", "count_quartile",
                         "component", "true_shape", "area_group", "overlap_group")),
    ):
        for group in groups:
            for value, part in table.groupby(group, dropna=False):
                row = {"level": name, "group": group, "value": str(value),
                       "n": len(part)}
                if name == "scan":
                    row.update(presence_recall=float(part["true_positive"].sum()
                                                     / max((part["true_positive"] + part["false_negative"]).sum(), 1)),
                               false_positives=int(part["false_positive"].sum()),
                               median_clean_nrmse=float(part["clean_nrmse"].median()),
                               background_accuracy=float(part["background_correct"].mean()))
                else:
                    row.update(presence_recall=float(part["detected"].mean()),
                               shape_accuracy_all_true=float(part["shape_correct"].mean()),
                               center_within_0p2_ev_all_true=float(part["center_within_0p2_ev"].mean()),
                               fwhm_within_0p2_ev_all_true=float(part["fwhm_within_0p2_ev"].mean()),
                               height_within_20_percent_all_true=float(part["height_within_20_percent"].mean()),
                               area_within_20_percent_all_true=float(part["area_within_20_percent"].mean()),
                               median_area_relative_error_detected=float(part["area_relative_error"].median()))
                rows.append(row)
    return pd.DataFrame(rows)


def plot_learning_history(history: pd.DataFrame, output_dir: str | Path):
    path = Path(output_dir) / "learning_curves.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, name, title in zip(
        axes.flat,
        ("total", "spectrum", "detection", "area"),
        ("Combined loss", "Clean-spectrum loss", "Peak detection loss", "Peak-area loss"),
    ):
        for split in ("train", "validation"):
            key = f"{split}_{name}"
            if key in history:
                ax.plot(history["epoch"], history[key], label=split)
        ax.set(title=title, xlabel="Epoch", ylabel="Loss")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def _make_figures(model, renderer, data, scans, peaks, cases,
                  summary, threshold, device, output):
    names = ["Peak recall", "Correct shape", "Center ±0.2 eV",
             "FWHM ±0.2 eV", "Height ±20%", "Area ±20%"]
    keys = ["presence_recall", "shape_accuracy_all_true",
            "center_within_0p2_ev_all_true", "fwhm_within_0p2_ev_all_true",
            "height_within_20_percent_all_true", "area_within_20_percent_all_true"]
    fig, ax = plt.subplots(figsize=(11, 4.7))
    values = [summary[k] for k in keys]
    bars = ax.bar(names, values, color=["#315b7d", "#5886a5", "#65a9a2",
                                       "#78b7a8", "#c99b55", "#b87b52"])
    ax.set(ylim=(0, 1), ylabel="Fraction correct", title="Held-out peak tasks (missed peaks count as failures)")
    ax.tick_params(axis="x", rotation=22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02,
                f"{value:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "task_overview.png", dpi=170)
    plt.close(fig)

    confusion = pd.crosstab(pd.Categorical(peaks["true_shape"], categories=SHAPE_NAMES[1:]),
                            pd.Categorical(peaks["predicted_shape"],
                                           categories=SHAPE_NAMES[1:] + ("missed",)),
                            dropna=False)
    confusion.to_csv(output / "shape_confusion.csv")
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    matrix = confusion.to_numpy()
    ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(matrix.shape[1]), yticks=range(matrix.shape[0]),
           xticklabels=confusion.columns, yticklabels=confusion.index,
           xlabel="Predicted shape", ylabel="True shape", title="Shape confusion and missed peaks")
    ax.tick_params(axis="x", rotation=30)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "shape_confusion.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, group, title in zip(axes.flat,
                                ("family", "n_true_peaks", "area_group", "overlap_group"),
                                ("Chemical family", "Number of true peaks", "Peak strength", "Nearest-peak separation")):
        grouped = peaks.groupby(group).agg(
            detection=("detected", "mean"),
            shape=("shape_correct", "mean"),
            area=("area_within_20_percent", "mean"), n=("index", "size"))
        positions = np.arange(len(grouped))
        width = 0.25
        for offset, column, label in ((-1, "detection", "Detected"),
                                      (0, "shape", "Shape correct"),
                                      (1, "area", "Area ±20%")):
            ax.bar(positions + offset * width, grouped[column], width,
                   label=label)
        ax.set(xticks=positions, xticklabels=[f"{v}\n(n={n})" for v, n in zip(grouped.index, grouped["n"])],
               ylim=(0, 1), title=title, ylabel="Fraction of true peaks")
        ax.tick_params(axis="x", rotation=18)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "failure_subgroups.png", dpi=170)
    plt.close(fig)

    best_cases = cases.head(3)["case_id"].tolist()
    worst_cases = cases.tail(3).iloc[::-1]["case_id"].tolist()
    selections = []
    for label, case_ids in (("best", best_cases), ("worst", worst_cases)):
        for case_id in case_ids:
            part = scans[scans["case_id"] == case_id]
            row = (part.nsmallest(1, "difficulty_score") if label == "best"
                   else part.nlargest(1, "difficulty_score")).iloc[0]
            selections.append({"category": label, "case_id": int(case_id),
                               "index": int(row["index"]),
                               "difficulty_score": float(row["difficulty_score"]),
                               "clean_nrmse": float(row["clean_nrmse"]),
                               "false_negative": int(row["false_negative"]),
                               "false_positive": int(row["false_positive"]),
                               "source_file": row["source_file"],
                               "source_row": int(row["source_row"])})
    pd.DataFrame(selections).to_csv(output / "best_worst_cases.csv", index=False)
    _plot_case_gallery(model, renderer, data, selections, threshold,
                       device, output / "best_worst_cases.png")


@torch.no_grad()
def _plot_case_gallery(model, renderer, data, selections, threshold, device, path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for ax, selected in zip(axes.flat, selections):
        index = selected["index"]
        row = SpectrumDataset(data, [index])[0]
        counts = row["counts"][None].to(device)
        step = row["native_step"][None].to(device)
        pred = model(counts, step, sample=False)
        curves = renderer(pred, step, hard=True, threshold=threshold)
        truth = renderer(truth_prediction({k: v[None].to(device)
                                           for k, v in row["labels"].items()}), step, hard=True)
        clean = curves["clean"][0].cpu().numpy()
        background = curves["background"][0].cpu().numpy()
        components = curves["components"][0].cpu().numpy()
        true_components = truth["components"][0].cpu().numpy()
        energy = data["energy"]
        ax.plot(energy, data["counts"][index], color="0.7", lw=0.9, label="Measured")
        ax.plot(energy, data["expected_counts"][index], color="#174a72", lw=1.8,
                label="True clean")
        ax.plot(energy, clean, color="#b23535", lw=1.6, label="Model clean")
        ax.plot(energy, background, color="#777", lw=1.1, ls=":", label="Model background")
        for component in components:
            if component.max() > 0:
                ax.plot(energy, component, lw=0.8, alpha=0.65)
        for component in true_components:
            if component.max() > 0:
                ax.plot(energy, component, lw=0.65, alpha=0.35, color="#174a72", ls="--")
        ax.set(title=(f"{selected['category'].title()} case {selected['case_id']} · "
                      f"NRMSE {selected['clean_nrmse']:.3f} · "
                      f"missed {selected['false_negative']}, extra {selected['false_positive']}"),
               xlabel="Binding energy (eV)", ylabel="Counts")
        ax.grid(alpha=0.18)
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def compare_available_models(output_root: str | Path = OUTPUT_ROOT):
    """Each completed notebook refreshes this comparison with available runs."""
    output_root = Path(output_root)
    rows = []
    dataset_hashes = set()
    for name in ("cnn", "vae", "transformer"):
        path = output_root / name / "test_summary.json"
        if path.is_file():
            item = json.loads(path.read_text())
            dataset_hashes.add(item.get("manifest_sha", ""))
            rows.append({
                "model": name, "test_scans": item["scans"],
                "peak_recall": item["presence_recall"],
                "peak_precision": item["presence_precision"],
                "shape_all_true": item["shape_accuracy_all_true"],
                "center_within_0p2_ev": item["center_within_0p2_ev_all_true"],
                "fwhm_within_0p2_ev": item["fwhm_within_0p2_ev_all_true"],
                "height_within_20_percent": item["height_within_20_percent_all_true"],
                "area_within_20_percent": item["area_within_20_percent_all_true"],
                "median_clean_nrmse": item["median_clean_nrmse"],
            })
    if len(dataset_hashes) > 1:
        raise ValueError("Model reports use different dataset manifests; comparison would be invalid")
    table = pd.DataFrame(rows)
    if len(table):
        output_root.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_root / "model_comparison.csv", index=False)
        tasks = ("peak_recall", "shape_all_true", "center_within_0p2_ev",
                 "fwhm_within_0p2_ev", "height_within_20_percent",
                 "area_within_20_percent")
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        positions = np.arange(len(tasks))
        bar_width = 0.75 / len(table)
        for j, row in table.reset_index(drop=True).iterrows():
            axes[0].bar(positions + (j - (len(table) - 1) / 2) * bar_width,
                        [row[task] for task in tasks], bar_width, label=row["model"])
        axes[0].set(xticks=positions,
                    xticklabels=("Recall", "Shape", "Center", "FWHM", "Height", "Area"),
                    ylim=(0, 1), ylabel="Fraction correct", title="Same held-out cases")
        axes[0].legend()
        axes[1].bar(table["model"], table["median_clean_nrmse"], color="#557d91")
        axes[1].set(ylabel="Median normalized RMSE (lower is better)",
                    title="Clean-spectrum recreation")
        fig.tight_layout()
        fig.savefig(output_root / "model_comparison.png", dpi=170)
        plt.close(fig)
        case_tables = []
        for name in table["model"]:
            case_path = output_root / name / "test_cases_ranked.csv"
            if case_path.is_file():
                cases = pd.read_csv(case_path)[[
                    "case_id", "mean_clean_nrmse", "missed_peaks", "extra_peaks"
                ]].rename(columns={
                    "mean_clean_nrmse": f"{name}_clean_nrmse",
                    "missed_peaks": f"{name}_missed_peaks",
                    "extra_peaks": f"{name}_extra_peaks",
                })
                case_tables.append(cases)
        if len(case_tables) == len(table):
            combined = case_tables[0]
            for cases in case_tables[1:]:
                combined = combined.merge(cases, on="case_id", validate="one_to_one")
            combined.to_csv(output_root / "case_comparison.csv", index=False)
    return table


@torch.no_grad()
def vae_uncertainty_report(model, data, cfg: RunConfig, device,
                           n_scans=200, draws=16):
    """Exploratory latent-sampling spread; this is not calibrated uncertainty."""
    if cfg.model_name != "vae":
        raise ValueError("Latent uncertainty report requires the VAE")
    test_indices = np.flatnonzero(data["split"] == "test")
    selected = np.random.default_rng(cfg.seed).choice(
        test_indices, size=min(n_scans, len(test_indices)), replace=False)
    loader = DataLoader(SpectrumDataset(data, selected), batch_size=16,
                        shuffle=False, num_workers=0)
    rows = []
    model.eval()
    for source_batch in loader:
        batch = _move_batch(source_batch, device)
        b = len(batch["counts"])
        counts = batch["counts"].repeat_interleave(draws, 0)
        step = batch["native_step"].repeat_interleave(draws, 0)
        pred = model(counts, step, sample=True)
        center = pred["center"].reshape(b, draws, 6).cpu().numpy()
        area = pred["area"].reshape(b, draws, 6).cpu().numpy()
        presence = torch.sigmoid(pred["presence_logits"]).reshape(b, draws, 6).cpu().numpy()
        for local, index in enumerate(batch["index"].numpy()):
            for slot in np.flatnonzero(data["peak_present"][index]):
                rows.append({
                    "index": int(index), "case_id": int(data["case_id"][index]),
                    "component": COMPONENT_NAMES[slot],
                    "true_center_ev": float(data["peak_center_ev"][index, slot]),
                    "mean_center_ev": float(center[local, :, slot].mean()),
                    "center_sample_std_ev": float(center[local, :, slot].std()),
                    "center_abs_error_ev": abs(float(center[local, :, slot].mean())
                                           - float(data["peak_center_ev"][index, slot])),
                    "true_area_counts_ev": float(data["peak_area_counts_ev"][index, slot]),
                    "mean_area_counts_ev": float(area[local, :, slot].mean()),
                    "area_sample_std_counts_ev": float(area[local, :, slot].std()),
                    "area_relative_error": _relative_error(
                        area[local, :, slot].mean(), data["peak_area_counts_ev"][index, slot]),
                    "mean_presence_probability": float(presence[local, :, slot].mean()),
                })
    table = pd.DataFrame(rows)
    output = Path(cfg.output_dir)
    table.to_csv(output / "test_latent_sampling.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(table["center_sample_std_ev"], table["center_abs_error_ev"],
                    s=8, alpha=0.35)
    axes[0].set(xlabel="Sample spread in center (eV)",
                ylabel="Actual center error (eV)", title="Does VAE spread track center error?")
    axes[1].scatter(table["area_sample_std_counts_ev"] /
                    table["true_area_counts_ev"].clip(lower=1),
                    table["area_relative_error"], s=8, alpha=0.35)
    axes[1].set(xlabel="Sample spread / true area", ylabel="Actual relative area error",
                title="Does VAE spread track area error?")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "test_latent_sampling.png", dpi=170)
    plt.close(fig)
    return table
