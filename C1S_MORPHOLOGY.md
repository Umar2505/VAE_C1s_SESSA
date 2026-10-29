# C 1s peak morphology workflow

The [notebook](c1s_lmfit_vae.ipynb) maps a noisy C 1s count vector to six peak slots: presence, profile type, center, width, shape parameters, height, area, and area fraction. Its latent space is variational, but it is a supervised peak-parameter predictor, not an input-reconstructing VAE. The labels are the parameters used to generate the synthetic spectra; they were not fitted back from noisy data.

## Compact dataset for the next peak model

The generator now writes format-5 compact training files by default:

```bash
python c1s_lmfit_dataset.py --cases 10000 --replicates 5 --scans-per-file 100
```

The output directory is `datasets/c1s_nexus/c1s_peak_training_v1/`. The manifest is written only after every family shard finishes; each `.nxs` file contains at most 100 scans from one family. These files retain the labelled peak/background parameters needed for training and the clean curves needed for evaluation:

| NeXus location | Role |
| --- | --- |
| `/entry/data/binding_energy`, `counts` | Energy axis and noisy model input |
| `/entry/data/expected_counts`, `background_counts` | Clean full-spectrum and background references for evaluation; clean peak-only reference is their difference |
| `/entry/data/native_step_ev` | Known sampling interval needed to reproduce the clean target after acquisition and interpolation |
| `/entry/labels/peak_present`, `peak_shape_code`, `peak_center_ev`, `peak_model_sigma_ev`, `peak_area_counts_ev` | Independent peak targets |
| `/entry/labels/peak_lorentz_gamma_ev`, `peak_pseudo_voigt_fraction`, `peak_asymmetry_gamma` | Parameters used only by the corresponding profile family |
| `/entry/labels/background_kind_code`, `background_parameters`, `instrument_gaussian_fwhm_ev` | Background and instrument targets needed to reconstruct the full clean spectrum |
| `/entry/labels/case_id`, `split` | Case-separated train, validation, and test groups |

The labels group also contains small component, profile, and background name lists to decode numeric codes. A fixed Tougaard `C` value is an attribute, not repeated for every scan. Height, FWHM, and area fraction can be calculated from the peak profile and are not duplicate training labels. Native scan arrays, stored per-case profiles, and full simulation history are omitted. The manifest keeps the generator seed and code hash for reproducibility. The compact generator varies ordinary profile families independently by peak instead of assigning one ordinary profile to the whole case. The scientific ranges are still illustrative and need calibration against measured spectra.

## Three format-5 training notebooks

The new notebooks are [`c1s_peak_cnn.ipynb`](c1s_peak_cnn.ipynb), [`c1s_peak_vae_v5.ipynb`](c1s_peak_vae_v5.ipynb), and [`c1s_peak_transformer.ipynb`](c1s_peak_transformer.ipynb). Each contains its own architecture, dataset reader, supervised loss, training loop, and evaluation code. They do not import [`c1s_peak_pipeline.py`](c1s_peak_pipeline.py) at runtime. All wait for the generator's completed manifest, use identical case splits, and check that the renderer can recreate spectra from the stored true parameters before starting training. They are prepared for execution on a training machine; **no new training has been run here**.

The CNN is a deterministic six-slot residual convolutional model. The VAE uses a convolutional encoder, a sampled latent state, physical parameter heads, gradual KL weighting, and free bits. The Transformer uses eight learned peak queries with a no-peak class and matching to the six chemical labels. All three train directly against stored parameter and class labels, without spectrum rendering in the loss. Their predicted spectra are rendered only for held-out evaluation. The Transformer's number of predicted peaks is flexible within eight queries, while its chemical vocabulary is still limited to the six synthetic labels.

Each run writes `best.pt`, `latest.pt`, `training_history.csv`, validation threshold selection, full held-out scan and peak predictions, task metrics, subgroup breakdown, shape confusion, and best/worst case galleries below `outputs/c1s_peak_models/supervised_parameters_v1/<model>/`. `outputs/c1s_peak_models/supervised_parameters_v1/model_comparison.csv` contains completed runs on the same manifest and objective. Earlier renderer-loss checkpoints remain in their original directories. The notebooks show how to read these reports. The VAE also saves a sample-spread report, which is exploratory rather than a calibrated confidence interval. See [`C1S_MODEL_COMPARISON.md`](C1S_MODEL_COMPARISON.md) for run order and metric definitions.

**The existing VAE notebook and saved checkpoint use format 3/4.** Format 5 is the data contract for the three new supervised parameter models; the existing notebook has not been converted to train on it. Keep the older files and checkpoint as the baseline. The generator refuses to overwrite either dataset.

## Generate and read the legacy notebook dataset

```bash
python c1s_lmfit_dataset.py --legacy-schema --cases 10000 --replicates 5 --scans-per-file 100 --output-dir datasets/c1s_nexus/c1s_atmospheric_v2_copy
```

The previously generated legacy dataset contains 50,000 scans across 500 small NeXus files in `datasets/c1s_nexus/c1s_atmospheric_v2/`. Each file contains at most 100 scans and holds whole cases from one chemical family, so silx view can open one shard at a time:

| Family and filename pattern | Total cases | Total scans | Files | Main character |
| --- | ---: | ---: | ---: | --- |
| `c1s_hydrocarbon_rich_*.nxs` | 3,500 | 17,500 | 175 | More C–C/C–H intensity |
| `c1s_oxidized_organic_*.nxs` | 4,000 | 20,000 | 200 | More oxygenated carbon |
| `c1s_soot_rich_*.nxs` | 2,500 | 12,500 | 125 | More graphitic carbon and satellites |

`manifest.json` lists each shard, its family, case count, and scan count. It is published only after every file finishes. Each `.nxs` file remains self-contained: `/entry/data` holds the common-grid noisy spectrum and clean expectation, `/entry/acquisition` holds native scans, `/entry/labels` holds peak and case properties, and `/entry/simulation` holds component profiles and provenance. Global `case_id` values remain unique across files; all repeats of a case use the same train, validation, or test split. The notebook waits for the complete manifest, then reads all files into one training set. It also accepts the earlier three-file manifest (format version 3). It raises a timeout after 24 hours if generation does not finish.

The notebook uses the format-4 shard directory when its manifest exists. It can also read supplied files in `dataset/` and a checkpoint in `output/`. If it finds both layouts, it compares every model input and label with the supplied three-file dataset before reusing that checkpoint. A changed dataset triggers training instead.

The family split is an organizational label, not a chemical identification made by the model. Every file still contains per-peak presence, shape, center, width, and area labels, so results can also be filtered by peak details. The generator refuses to overwrite an existing dataset; choose a new `--output-dir` for another run.

## See where the model succeeds

The held-out test section saves `test_task_metrics.csv` with a separate score and evaluated-peak count for detection, shape, center, FWHM, height, area, and area fraction. `test_morphology_breakdown.csv` reports these tasks by chemical family, background kind, true number of peaks, count level, component, true profile, peak area fraction, and separation from the nearest true peak. Count-level thresholds come from training scans. `test_hard_case_factors.csv` and `.png` focus on weak and overlapping peaks. `test_case_difficulty.csv` ranks whole held-out cases after combining their repeated acquisitions. `test_morphology_overview.png` shows six task views by component, `test_shape_confusion.png` shows profile mistakes and missed peaks, and `test_easy_hard_examples.png` shows representative spectra. Per-peak predictions remain available in `outputs/c1s_nexus_peak_vae/`. CSV rows include the source file and row so a hard case can be found in its NeXus shard.

The numeric errors are calculated on detected true peaks; detection recall and false negatives must be read alongside them. Shape, center, FWHM, height, and area also have success rates over **all** true peaks, counting a missed peak as incorrect. Center and FWHM use a 0.2 eV tolerance. Height and area use a 20% tolerance and have median absolute percentage error on detected peaks, making errors interpretable across count levels. Case difficulty is a review ranking based on detection F1, center error, and shape accuracy, not a scientific uncertainty estimate. The loss and failure analysis is in [MODEL_LOSS_AUDIT.md](MODEL_LOSS_AUDIT.md).

The latest GitHub commit (`6ea7a914`) contains predictions from the **previous** three-file training run. The supplied files were used to calculate the new tables and figures from those saved predictions, so the current results are real evaluations of that checkpoint, not a newly trained model. On its 5,000-scan synthetic test split, peak-presence F1 is 0.945, shape accuracy among detected true peaks is 0.714, and correct shape across **all** true peaks is 0.666. Center MAE is 0.139 eV and FWHM MAE is 0.099 eV. Median height and area errors are 26.2% and 27.4%. Height and area are within 20% for only 39.4% and 37.7% of detected true peaks. Presence recall falls from 1.000 on one-peak spectra to 0.814 on six-peak spectra. Voigt and pseudo-Voigt profiles are often confused. These results are in `outputs/c1s_nexus_peak_vae/`.

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

The adapter creates lmfit models from these records, estimates missing starting values, and compares linear, Shirley, and Tougaard backgrounds by BIC. If `profile` is omitted or set to `"auto"`, it compares Gaussian, Voigt, pseudo-Voigt, and broadened Doniach profiles. The final notebook cell selects an easy and a difficult held-out case from each family. It fits one representative scan from each case using only VAE predictions as starting values, then saves a gallery, individual plots, curves, residuals, and fitted parameters in `outputs/c1s_nexus_peak_vae/lmfit_examples/`. A summary CSV includes the VAE starting error, final error, predicted and true peak counts, and convergence status. The supplied saved predictions were used to produce the six current examples without retraining. All six fits converged, yet two difficult cases had fewer predicted peaks than true peaks and one had an extra predicted peak. A small fitted residual therefore does not establish correct peak morphology. A different predictor only needs to output the same small record schema; a specialist does not need to write a new formula for these supported shapes.

lmfit is optional refinement and a residual check. The network's peak predictions can be used directly when their accuracy is sufficient. Fitting can adjust centers, widths, areas, and backgrounds, but it cannot reliably recover a peak the network omitted; the subgroup detection metrics therefore remain central.

This fitting interface is flexible across peak counts and labels; the current neural network still has six fixed C 1s output slots and was trained only on this generator. Its predictions are not universal for other spectral regions or compounds. The built-in profile and background catalogue also cannot represent every measured spectrum. A poor residual, unstable fit, or systematic subgroup failure should trigger model review and measured-data validation, not automatic chemical interpretation.

## Scientific scope

The generator varies charging shift, peak widths, native scan spacing, count level, and Poisson noise. It uses illustrative hydrocarbon-rich, oxidized-organic, and soot-rich priors. Only the soot-rich graphitic component uses the Gaussian-broadened Doniach profile, and a π–π* satellite appears only with its graphitic parent. Component areas refer to the finite 280–300 eV window. Linear, Shirley, and Tougaard backgrounds are forward-generated from known clean peaks; they are not background-removal algorithms applied to measured data. The Tougaard kernel is limited to the 20 eV simulation window. These synthetic results do not establish accuracy on measured XPS spectra.
