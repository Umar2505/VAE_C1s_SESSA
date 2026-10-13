# Atmospheric C 1s dataset with SESSA 2.2.2

This workflow creates 12,000 independently parameterized physical C 1s cases in SESSA,
saves a full spectrum and a zero-inelastic-collision spectrum for each case, and produces
five independent counting-noise acquisitions per case. The default result is exactly 60,000
spectra on a 280-300 eV, 401-point binding-energy grid.

## 1. Generate the SESSA session

From the `spectraVAE` directory:

```bash
python sessa_atmospheric/generate_sessa_batch.py \
  --cases 12000 \
  --shards 8 \
  --output sessa_generated
```

First test one case before committing to the full run:

```bash
python sessa_atmospheric/generate_sessa_batch.py \
  --cases 1 \
  --output sessa_pilot
```

## 2. Run the session in SESSA

For a single session, open the SESSA CLI with Ctrl+9 and enter this command, replacing the path with the absolute
path printed by the generator:

```text
PROJECT LOAD SESSION "/absolute/path/to/sessa_generated/atmospheric_c1s_batch.ses"
```

For a Linux installation, run generated shards concurrently with:

```bash
python sessa_atmospheric/run_sessa_shards.py \
  --input sessa_generated \
  --sessa /absolute/path/to/sessa \
  --library-dir /absolute/path/to/sessa/lib \
  --workers 8
```

Before running 12,000 cases, run the one-case pilot and enter `SAMPLE PEAK LIST` in SESSA.
Confirm that peaks 1-5 are, in order: `CC_CH`, `C_O`, `C_eq_O`, `O_C_eq_O`, and
`pi_pi_star`. SESSA assigns peak numbers when it interprets the material. If a particular build
uses a different order, change the peak-number order in `commands_for_case` before the full run.

The training run uses a 1% Monte Carlo convergence criterion. It can take many hours or days.
Run independent case ranges on separate machines/processes if necessary; do not launch several
writers into the same output directory.

## 3. Build the 60,000-spectrum VAE file

After every SESSA output is present:

```bash
python sessa_atmospheric/build_vae_dataset.py \
  --input sessa_generated \
  --replicates 5 \
  --output atmospheric_c1s_60000.npz
```

SESSA 2.3.0 is driven on its native kinetic-energy scale because its binding-energy region
setting does not populate the C 1s peak list reliably in console mode. The conversion back to
binding energy uses Al K-alpha = 1486.6 eV and is performed automatically by the dataset builder.

The resulting NPZ contains the energy axis, noisy spectra, expected totals, zero-loss clean
signals, SESSA-derived inelastic backgrounds, component centres, FWHMs, nominal material
fractions, fitted component heights in counts, fitted areas in counts eV, and fitted area
fractions, plus physical case IDs and noise-realization IDs. Split
training/validation/testing by `case_id`, not by row.

### Preferred self-contained HDF5 layout

Create one HDF5 file per physical case, with five acquisitions and all labels embedded:

```bash
python sessa_atmospheric/build_h5_dataset.py \
  --input sessa_generated \
  --output atmospheric_c1s_h5 \
  --replicates 5
```

This creates one `.h5` file per physical case. The current dataset has 14,000 cases and
70,000 acquisitions. Newly built files include the spectra, energy axis, peak names,
centres, FWHMs, nominal material fractions, fitted heights and areas, presence mask,
atmospheric family, aging index, O:C ratio, density, band gap,
charging shift, count scale, noise labels,
SESSA version, photon energy, and a train/validation/test assignment. `dataset_index.csv`
provides a lightweight catalogue of every file.

`peaks/height_counts` has shape `(acquisitions, components)`. It is the maximum intensity
of each **isolated** zero-loss Gaussian component, after that acquisition's count scaling.
The builder obtains it by non-negative least squares against the SESSA zero-loss spectrum
using the recorded centres and FWHMs. Absent components have height zero, and
`peaks/height_fit_nrmse` records the decomposition error. `peaks/area_counts_eV`
integrates each fitted Gaussian over the recorded 280–300 eV grid, and
`peaks/fitted_area_fraction` normalizes those areas to sum to one per case. These are
spectral peak areas. The existing `peaks/area_fraction` field and its clearer alias
`peaks/material_fraction` are nominal material fractions, not fitted spectral areas.
SESSA's `SAMPLE PEAK SET HEIGHT 1.0` specifies a relative subpeak height in arbitrary
units; it is not a detector-count label. The loader derives height and area labels from
older HDF5 files (schemas 1.0 and 1.1), so SESSA simulations do not need to be repeated.
New files use schema 1.2; existing files are left intact.

## Targeted continuation after the 12,000-case baseline

The saved vanilla VAE is weakest on low-count C 1s spectra, especially when the
pi-pi* component is strong or peaks are narrow. `continue_dataset.py` prepares **only** IDs
12000-13999. It selects new physical cases from the original parameter ranges:
1,200 have strong pi-pi* with negative, central, or positive charging shifts;
400 have a moderate pi-pi* component with positive shift; and 400 have weak or
absent pi-pi* but a complex positively shifted multiplet. Each case gets four
200-1,000-count acquisitions and one 1,000-10,000-count acquisition. This is
2,000 new physical cases and 10,000 new acquisitions, not a rebuild of the old
12,000 cases.

```bash
python sessa_atmospheric/continue_dataset.py prepare --cases 2000 --shards 8
python sessa_atmospheric/run_sessa_shards.py \
  --input sessa_generated \
  --sessa /absolute/path/to/sessa \
  --library-dir /absolute/path/to/sessa/lib \
  --workers 8 \
  --pattern 'atmospheric_c1s_continuation_*.ses'
python sessa_atmospheric/continue_dataset.py package --cases 2000
```

The runner skips completed new cases on retry. Packaging writes only new HDF5
files and then appends their rows to `dataset_index.csv` and
`physical_cases.csv`. The original splits stay fixed; the continuation adds
1,400 training, 300 validation, and 300 test cases, stratified within the five
new regimes. The vanilla notebook now samples weak, pi-rich, and narrow-peak **training**
acquisitions more often; validation and test sampling remain unchanged. Its
saved plots and metrics still describe the original 12,000-case checkpoint.
The old test cases informed the choice of continuation regimes, so report the
new held-out continuation test cases separately when judging the retrained model.

To audit that checkpoint on the original test set and, after packaging, the
new continuation test set without running notebook cells:

```bash
python sessa_atmospheric/evaluate_vanilla_slices.py
```

After retraining the notebook, point `--checkpoint` at its newly saved
`outputs/atmospheric_vanilla_vae_zero_loss/best_vanilla_vae_zero_loss.pt`
and choose a separate `--output` name to compare both checkpoints.
