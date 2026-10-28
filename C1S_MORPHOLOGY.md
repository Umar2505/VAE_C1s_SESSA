# C 1s peak morphology workflow

The [notebook](c1s_lmfit_vae.ipynb) maps a noisy C 1s count vector to six peak slots: presence, profile type, center, width, shape parameters, height, area, and area fraction. Its latent space is variational, but it is a supervised peak-parameter predictor, not an input-reconstructing VAE. The labels are the parameters used to generate the synthetic spectra; they were not fitted back from noisy data.

## Generate and read the dataset

```bash
python c1s_lmfit_dataset.py --cases 10000 --replicates 5
```

The default run writes 50,000 scans across three family files in `datasets/c1s_nexus/c1s_atmospheric_v2/`:

| File | Cases | Scans | Main character |
| --- | ---: | ---: | --- |
| `c1s_hydrocarbon_rich.nxs` | 3,500 | 17,500 | More C–C/C–H intensity |
| `c1s_oxidized_organic.nxs` | 4,000 | 20,000 | More oxygenated carbon |
| `c1s_soot_rich.nxs` | 2,500 | 12,500 | More graphitic carbon and satellites |

`manifest.json` lists the family files and their scan counts. It is published only after every file finishes. Each `.nxs` file remains self-contained: `/entry/data` holds the common-grid noisy spectrum and clean expectation, `/entry/acquisition` holds native scans, `/entry/labels` holds peak and case properties, and `/entry/simulation` holds component profiles and provenance. Global `case_id` values remain unique across files; all repeats of a case use the same train, validation, or test split. The notebook waits for the complete manifest, then reads all three files into one training set. It preserves the local wait-cell behavior and raises a timeout after 24 hours if generation does not finish.

The family split is an organizational label, not a chemical identification made by the model. Every file still contains per-peak presence, shape, center, width, and area labels, so results can also be filtered by peak details. The generator refuses to overwrite an existing dataset; choose a new `--output-dir` for another run.

## See where the model succeeds

The held-out test section now saves `test_morphology_breakdown.csv` with detection, shape, center, width, and area-fraction metrics by chemical family, background kind, true number of peaks, count level, component, and true profile. Count-level thresholds come from training scans. `test_case_difficulty.csv` ranks entire held-out cases after combining their repeated acquisitions. `test_morphology_overview.png` and `test_easy_hard_examples.png` show component and family detection rates and representative easy and hard spectra. The original overall summary and per-peak prediction CSV remain available in `outputs/c1s_nexus_peak_vae/`. CSV rows include the source file and row so a hard case can be found in its NeXus shard.

The numeric errors are calculated on detected true peaks; detection recall and false negatives must be read alongside them. A small center error alone can therefore hide missed peaks. Case difficulty is a review ranking based on detection F1, center error, and shape accuracy, not a scientific uncertainty estimate.

## Reusable lmfit refinement

[`peak_fitting.py`](peak_fitting.py) accepts any number of predicted peak records. Names are identifiers, not hard-coded chemical formulas. Each present record needs a center and FWHM (or sigma); area, height, profile type, and shape parameters are optional. For example:

```python
from peak_fitting import fit_predicted_peaks

predictions = [
    {"center_ev": 284.8, "fwhm_ev": 1.0, "area_counts_ev": 5000},
    {"center_ev": 286.3, "fwhm_ev": 1.2, "area_counts_ev": 1500},
]
fit = fit_predicted_peaks(energy_ev, counts, predictions)
print(fit.background_kind, fit.peaks)
```

The adapter creates lmfit models from these records, estimates missing starting values, and compares linear, Shirley, and Tougaard backgrounds by BIC. If `profile` is omitted or set to `"auto"`, it compares Gaussian, Voigt, pseudo-Voigt, and broadened Doniach profiles. The final notebook cell applies this function to a held-out scan and saves the fitted curve, components, residual, parameters, and plot. A different predictor only needs to output the same small record schema; a specialist does not need to write a new formula for these supported shapes.

lmfit is optional refinement and a residual check. The network's peak predictions can be used directly when their accuracy is sufficient. Fitting can adjust centers, widths, areas, and backgrounds, but it cannot reliably recover a peak the network omitted; the subgroup detection metrics therefore remain central.

This fitting interface is flexible across peak counts and labels; the current neural network still has six fixed C 1s output slots and was trained only on this generator. Its predictions are not universal for other spectral regions or compounds. The built-in profile and background catalogue also cannot represent every measured spectrum. A poor residual, unstable fit, or systematic subgroup failure should trigger model review and measured-data validation, not automatic chemical interpretation.

## Scientific scope

The generator varies charging shift, peak widths, native scan spacing, count level, and Poisson noise. It uses illustrative hydrocarbon-rich, oxidized-organic, and soot-rich priors. Only the soot-rich graphitic component uses the Gaussian-broadened Doniach profile, and a π–π* satellite appears only with its graphitic parent. Component areas refer to the finite 280–300 eV window. Linear, Shirley, and Tougaard backgrounds are forward-generated from known clean peaks; they are not background-removal algorithms applied to measured data. The Tougaard kernel is limited to the 20 eV simulation window. These synthetic results do not establish accuracy on measured XPS spectra.
