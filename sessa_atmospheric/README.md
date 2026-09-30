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
signals, SESSA-derived inelastic backgrounds, component centres, FWHMs, fractions, physical
case IDs, and noise-realization IDs. Split training/validation/testing by `case_id`, not by row.

### Preferred self-contained HDF5 layout

Create one HDF5 file per physical case, with five acquisitions and all labels embedded:

```bash
python sessa_atmospheric/build_h5_dataset.py \
  --input sessa_generated \
  --output atmospheric_c1s_h5 \
  --replicates 5
```

This creates 12,000 `.h5` files containing 60,000 acquisitions in total. Each file includes
the spectra, energy axis, peak names, centres, FWHMs, fractions, presence mask, atmospheric
family, aging index, O:C ratio, density, band gap, charging shift, count scale, noise labels,
SESSA version, photon energy, and a train/validation/test assignment. `dataset_index.csv`
provides a lightweight catalogue of every file.
