"""Train a vanilla VAE with a supervised peak-parameter head on PROPHESY data.

The encoder and decoder match the fully connected VanillaVAE in the atmospheric
notebook. A small head reads the latent mean to predict peak centers, Gaussian
sigmas, and area fractions. Each spectral region is trained separately because
its energy grid and peak definition differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from prophesy_peakfit.dataset import REGIONS, load_region


class VanillaPeakVAE(nn.Module):
    def __init__(self, input_dim: int, n_peaks: int, axis_min: float, axis_max: float, latent_dim: int = 12):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.mu = nn.Linear(64, latent_dim)
        self.logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim),
        )
        self.peak_head = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 3 * n_peaks))
        self.n_peaks = n_peaks
        self.axis_min = float(axis_min)
        self.axis_max = float(axis_max)
        self.latent_dim = latent_dim

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.mu(features), self.logvar(features)

    def peak_parameters(self, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.peak_head(mu).reshape(-1, self.n_peaks, 3)
        center = self.axis_min + (self.axis_max - self.axis_min) * torch.sigmoid(raw[:, :, 0])
        sigma = 0.01 + F.softplus(raw[:, :, 1])
        fraction = torch.softmax(raw[:, :, 2], dim=1)
        return center, sigma, fraction

    def forward(self, x: torch.Tensor, sample: bool = True):
        mu, logvar = self.encode(x)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        return self.decoder(z), self.peak_parameters(mu), mu, logvar


def normalize(noisy: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.maximum(noisy.max(axis=1), 1.0).astype(np.float32)
    divisor = np.log1p(scale)[:, None]
    return (np.log1p(np.maximum(noisy, 0)) / divisor).astype(np.float32), (
        np.log1p(np.maximum(clean, 0)) / divisor
    ).astype(np.float32), scale


def _tensor_loader(data: dict, indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    noisy = data["noisy_counts"][indices]
    clean = data["clean_counts"][indices]
    centers = data["peak_center_eV"][indices]
    sigmas = data["peak_sigma_eV"][indices]
    fractions = data["peak_fraction"][indices]
    x, target, _ = normalize(noisy, clean)
    tensors = TensorDataset(*(torch.from_numpy(a.astype(np.float32)) for a in
                             (x, target, centers, sigmas, fractions)))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def loss_parts(model: VanillaPeakVAE, output, target, true_center, true_sigma, true_fraction,
               beta: float, peak_weight: float) -> tuple[torch.Tensor, ...]:
    reconstruction, (center, sigma, fraction), mu, logvar = output
    recon_loss = (reconstruction - target).square().mean()
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1)).mean() / model.latent_dim
    center_scale = (model.axis_max - model.axis_min) / 4
    peak_loss = (((center - true_center) / center_scale).square().mean()
                 + (torch.log(sigma) - torch.log(true_sigma)).square().mean()
                 + (fraction - true_fraction).square().mean())
    return recon_loss + beta * kl + peak_weight * peak_loss, recon_loss, kl, peak_loss


@torch.no_grad()
def evaluate(model: VanillaPeakVAE, loader: DataLoader, device: torch.device,
             beta: float, peak_weight: float) -> dict[str, float]:
    model.eval()
    totals = np.zeros(6, dtype=np.float64)
    count = 0
    for batch in loader:
        x, target, center, sigma, fraction = (a.to(device) for a in batch)
        output = model(x, sample=False)
        losses = loss_parts(model, output, target, center, sigma, fraction, beta, peak_weight)
        predicted_center, predicted_sigma, predicted_fraction = output[1]
        metrics = [v.item() for v in losses] + [
            (predicted_center - center).abs().mean().item(),
            (predicted_sigma - sigma).abs().mean().item(),
        ]
        totals += len(x) * np.asarray(metrics)
        count += len(x)
    return dict(zip(("total", "reconstruction", "kl", "peak", "center_mae_eV", "sigma_mae_eV"),
                    (totals / count).tolist()))


@torch.no_grad()
def write_predictions(model: VanillaPeakVAE, data: dict, output_path: Path, device: torch.device) -> None:
    indices = np.flatnonzero(data["split"] == "test")
    noisy = data["noisy_counts"][indices]
    clean = data["clean_counts"][indices]
    x, _, scale = normalize(noisy, clean)
    axis = data["axis_eV"]
    rows = []
    model.eval()
    for start in range(0, len(indices), 256):
        chunk = torch.from_numpy(x[start:start + 256]).to(device)
        reconstruction, (center, sigma, fraction), _, _ = model(chunk, sample=False)
        reconstructed = np.expm1(reconstruction.cpu().numpy() * np.log1p(scale[start:start + 256, None]))
        total_area = np.trapezoid(np.maximum(reconstructed, 0), axis, axis=1)
        for j, original_index in enumerate(indices[start:start + 256]):
            for peak in range(model.n_peaks):
                rows.append({
                    "sample_id": data["sample_id"][original_index], "peak_index": peak + 1,
                    "true_center_eV": float(data["peak_center_eV"][original_index, peak]),
                    "predicted_center_eV": float(center[j, peak]),
                    "true_sigma_eV": float(data["peak_sigma_eV"][original_index, peak]),
                    "predicted_sigma_eV": float(sigma[j, peak]),
                    "true_fraction": float(data["peak_fraction"][original_index, peak]),
                    "predicted_fraction": float(fraction[j, peak]),
                    "true_area_counts_eV": float(data["peak_area_counts_eV"][original_index, peak]),
                    "predicted_area_counts_eV": float(fraction[j, peak]) * float(total_area[j]),
                })
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_spectrum(checkpoint_path: Path, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Fit one spectrum already sampled on the checkpoint's region grid."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = VanillaPeakVAE(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    values = np.asarray(counts, dtype=np.float32)
    if values.shape != (checkpoint["model_config"]["input_dim"],):
        raise ValueError("Spectrum must be sampled on checkpoint axis_eV")
    x, _, scale = normalize(values[None, :], values[None, :])
    with torch.no_grad():
        reconstruction, (center, sigma, fraction), _, _ = model(torch.from_numpy(x), sample=False)
    clean = np.maximum(np.expm1(reconstruction.numpy()[0] * np.log1p(scale[0])), 0)
    area = float(np.trapezoid(clean, checkpoint["axis_eV"])) * fraction.numpy()[0]
    return {"axis_eV": checkpoint["axis_eV"], "clean_counts": clean,
            "center_eV": center.numpy()[0], "sigma_eV": sigma.numpy()[0],
            "fraction": fraction.numpy()[0], "area_counts_eV": area}


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data = load_region(args.archive, args.region)
    indices = {name: np.flatnonzero(data["split"] == name) for name in ("train", "validation", "test")}
    if args.max_samples:
        indices = {name: values[:args.max_samples] for name, values in indices.items()}
    train_loader = _tensor_loader(data, indices["train"], args.batch_size, True)
    validation_loader = _tensor_loader(data, indices["validation"], args.batch_size, False)
    test_loader = _tensor_loader(data, indices["test"], args.batch_size, False)
    axis = data["axis_eV"]
    config = {"input_dim": len(axis), "n_peaks": data["peak_center_eV"].shape[1],
              "axis_min": float(axis[0]), "axis_max": float(axis[-1]), "latent_dim": args.latent_dim}
    model = VanillaPeakVAE(**config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    output_dir = args.output / args.region
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            x, target, center, sigma, fraction = (a.to(device) for a in batch)
            optimizer.zero_grad(set_to_none=True)
            losses = loss_parts(model, model(x), target, center, sigma, fraction,
                                args.beta, args.peak_weight)
            losses[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        validation = evaluate(model, validation_loader, device, args.beta, args.peak_weight)
        score = validation["total"]
        print(f"{args.region} epoch {epoch:03d}: validation={score:.5f}, "
              f"center MAE={validation['center_mae_eV']:.3f} eV")
        if score < best_score - args.min_delta:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "model_config": config,
                        "axis_eV": axis, "region": args.region, "archive": str(args.archive)},
                       output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, test_loader, device, args.beta, args.peak_weight)
    write_predictions(model, data, output_dir / "test_peak_fits.csv", device)
    summary = {"region": args.region, "device": str(device),
               "spectra": len(data["noisy_counts"]),
               "distinct_conditions": len(np.unique(data["condition_id"])),
               "train_rows": len(indices["train"]), "validation_rows": len(indices["validation"]),
               "test_rows": len(indices["test"]),
               "best_epoch": best_epoch, "best_validation": best_score, "test_metrics": test}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("datasets/prophesy_peakfit.zip"))
    parser.add_argument("--region", choices=REGIONS, default="C1s")
    parser.add_argument("--output", type=Path, default=Path("outputs/prophesy_vanilla_vae"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--latent-dim", type=int, default=12)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--peak-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--max-samples", type=int, default=None, help="Limit base samples per split for a quick run")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
