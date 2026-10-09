# PROPHESY peak fitting with the vanilla VAE

The [archive](../datasets/prophesy_peakfit.zip) contains 52,500 simulated PROPHESY spectra:
1,500 primary scans with explicit generating peak parameters and 51,000 repeated noisy scans
from those same conditions. It excludes experimental fitting tables,
which contain conflicting annotations for duplicated raw spectra, and 120 older simulations
without individual peak labels. The six IGOR workbooks are reprocessings of existing cases.

| Region | Primary scans | Repeated scans | Total | Peaks per spectrum | Common binding-energy grid |
| --- | ---: | ---: | ---: | ---: | --- |
| C 1s | 1,020 | 46,000 | 47,020 | 3 | 271–291 eV, 401 points |
| S 2p | 100 | 5,000 | 5,100 | 2 | 165–174.5 eV, 191 points |
| O 1s liquid/gas | 380 | 0 | 380 | 2 | 528–542 eV, 281 points |

The repeated scans share the peak parameters of their primary scan. They add noise
realizations, not new chemical conditions. All scans from a workbook stay in one
train/validation/test split to prevent leakage.

The archive includes `C1s.npz`, `S2p.npz`, `O1s.npz`, `manifest.csv`, and `metadata.json`.
Each NPZ has noisy input counts, clean peak-signal counts, background counts, a common
energy axis, peak centers, Gaussian **sigma** widths, area fractions, integrated peak
areas, source and condition IDs, scan names, photon energies, and train/validation/test assignments. The
workbook is the split group, so its photon energies remain together. The O 1s grid uses only the interval measured by all
water workbooks. The C 1s and S 2p area labels follow the generator's fixed peak-weight
times total-area rule; they are synthetic labels, not independently fitted component
areas.

## Train and inspect

From the `spectraVAE` directory, use an environment with NumPy, OpenPyXL and PyTorch:

```bash
python -m prophesy_peakfit.vanilla_vae --region C1s --epochs 100
python -m prophesy_peakfit.vanilla_vae --region S2p --epochs 100
python -m prophesy_peakfit.vanilla_vae --region O1s --epochs 100
```

Each run saves `best.pt`, `metrics.json`, and
`test_peak_fits.csv` in `outputs/prophesy_vanilla_vae/<region>/`. The
[notebook](../prophesy_vanilla_vae.ipynb) provides an interactive entry point and a
plot of a held-out fit.

The model retains the existing vanilla VAE's fully connected encoder and decoder.
An added supervised head predicts peak centers, widths, and area fractions from the
latent mean. It trains a separate model for each spectral region because their energy
axes and peak meanings differ. A saved checkpoint can fit a new spectrum already
sampled on its `axis_eV`:

```python
from pathlib import Path
from prophesy_peakfit.vanilla_vae import fit_spectrum

result = fit_spectrum(Path("outputs/prophesy_vanilla_vae/C1s/best.pt"), counts)
print(result["center_eV"], result["sigma_eV"], result["area_counts_eV"])
```

The current synthetic peak weights are often fixed across many cases, so held-out
workbook metrics do not demonstrate generalization to new chemical compositions.
Use a group split and evaluate any model intended for experimental spectra on
reviewed measured data.

## Rebuild the archive

```bash
python prophesy_peakfit/build_archive.py \
  --source /absolute/path/to/PROPHESY/data \
  --output datasets/prophesy_peakfit.zip
```
