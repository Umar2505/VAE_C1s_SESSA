# C 1s peak-property models

This document describes `c1s_peak_cnn.ipynb`, `c1s_peak_transformer.ipynb`, and `c1s_peak_vae.ipynb`. All three notebooks train on the same SESSA data and predict the same peak properties. They differ in how they represent the input spectrum.

The notebooks use `datasets/sessa_c1s_v2`, with seven possible labels (`CC`, `CN`, `CO`, `CARBONYL`, `OCO`, `COO`, `COOH`) and at most five present in a case. Their saved v1 outputs were cleared when the notebooks changed. The revised models need to be trained before their results can be evaluated; the v1 and v2 labels must not be mixed.

## What one prediction contains

The dataset has seven possible C 1s components: `CC` (C–C/C–H), `CN` (single C–N bond), `CO` (C–O), `CARBONYL` (C=O), `OCO` (O–C–O), `COO` (ester-like O–C=O), and `COOH` (carboxylic acid carbon). For **each** component, every model predicts:

| Property | Meaning | Source of the training label |
| --- | --- | --- |
| Presence probability | Model score for whether the component is included in the SESSA simulation. A value of at least 0.5 is called present; if more than five pass, only the five highest probabilities are kept. | `presence` |
| Center, eV | Binding-energy position assigned to that peak. | `center_ev` |
| FWHM, eV | Effective full width at half maximum of the simulated component after SESSA processing and analyzer broadening. | `fwhm_ev` |
| Area | Area of the component over the 280–300 eV window, in relative-intensity × eV units. | `component_area_window` |
| Shape | One of Gaussian, Lorentzian, or pseudo-Voigt. | `shape_code` |

The models output enough information to construct one approximate peak curve for every component called present. The area is relative to the SESSA clean spectrum's normalization; it is not an absolute measured count rate or an atomic percentage. The FWHM is an *effective width of the saved component curve*, not the intrinsic Gaussian or Lorentzian width entered into SESSA. A shape label is predicted separately from the width.

Each model's final layer has **49 raw values**: seven for each of seven components. Within each set of seven, one value is a presence score, one produces the center, one produces FWHM, one produces area, and three are scores for the possible shapes. The center is constrained to within 0.6 eV of that component's nominal position. FWHM is constrained to 0.5–3.0 eV, and area is constrained to be positive. The generator checks that measured FWHM labels for present components are within this range. If analyzer broadening makes a saved component slightly wider than 3.0 eV, it reduces that case's analyzer width just enough to meet the limit and records both the sampled and used widths in `metadata.json`.

## Shared data pipeline

The notebooks expect `dataset.npz` from `generate_sessa_training_dataset.py`. The first code cell contains `data_path`; set it to the complete dataset on the training machine. The current generator uses a 401-point C 1s grid from 280 to 300 eV and saves three noisy count observations per simulated case. The notebook reads the noisy observations as inputs and repeats the same peak labels for all three observations of a case.

Before a spectrum enters a model, the notebook divides its nonnegative intensities by that spectrum's maximum. It does not subtract an endpoint baseline or clip an intensity floor. The peak labels, including measured FWHM from the noise-free component curves, come directly from the NPZ file.

The NPZ's `train`, `validation`, and `test` case labels are preserved. All noisy observations generated from the same SESSA case stay in the same split. This prevents a noisy copy of a training case from appearing in validation or test data.

## CNN architecture

`c1s_peak_cnn.ipynb` processes the spectrum as a one-dimensional signal. Three convolution layers look for local patterns and reduce the number of energy positions. A small dense head then produces the 49 raw property values.

| Stage | Output size for a 401-point input |
| --- | --- |
| Input spectrum | 1 × 401 |
| Convolution 1: 16 filters, width 7, stride 2, ReLU | 16 × 201 |
| Convolution 2: 32 filters, width 5, stride 2, ReLU | 32 × 101 |
| Convolution 3: 64 filters, width 5, stride 2, ReLU | 64 × 51 |
| Adaptive average pooling | 64 × 16 |
| Flatten and dense layers, 1024 → 64 → 49 | 49 values |

Keeping 16 pooled positions allows the dense layers to use information about where a peak occurs on the energy axis. The notebook saves the best validation checkpoint as `outputs/sessa_models_v2_raw/cnn.pt` when run.

## Transformer architecture

`c1s_peak_transformer.ipynb` first converts the 401-point spectrum into 51 tokens, each with 32 values. It adds a learned position vector to each token so that a feature at 285 eV is distinguishable from one at 289 eV. Two Transformer encoder layers then let different energy regions exchange information through attention.

| Stage | Output size |
| --- | --- |
| Convolutional embedding, 1 → 32 channels, width 7, stride 8 | 51 × 32 tokens |
| Learned position vectors | 51 × 32 tokens |
| Two encoder layers, 4 attention heads, 64-unit feed-forward part | 51 × 32 tokens |
| Mean of the 51 encoded tokens | 32 values |
| Dense layers, 32 → 32 → 49 | 49 values |

Encoder dropout is 0.1. The best validation checkpoint is saved as `outputs/sessa_models_v2_raw/transformer.pt` when run.

## VAE architecture

`c1s_peak_vae.ipynb` combines a variational autoencoder with a supervised property head. An encoder compresses the whole spectrum, and two small layers describe a distribution over 16 latent values. During training, the model samples from this distribution. During evaluation it uses the latent mean, making predictions deterministic.

| Stage | Output size |
| --- | --- |
| Encoder, 401 → 128 → 64 with ReLU | 64 values |
| Separate mean and log-variance layers | 16 + 16 values |
| Sampled latent vector during training; mean at evaluation | 16 values |
| Decoder, 16 → 128 → 401 with ReLU and sigmoid | Reconstructed normalized input spectrum |
| Property head, 16 → 49 | 49 property values |

The VAE's decoder reconstructs the **maximum-normalized noisy input**, including its floor. The property head supplies the peak table. The best validation checkpoint is saved as `outputs/sessa_models_v2_raw/vae.pt` when run.

## Training loss and model selection

All models use a batch size of 128, Adam with learning rate 0.001, a default of 100 epochs, and random seed 42. The notebook uses a GPU when available. After each epoch, it measures validation loss and saves the weights if that loss improved. The test and real-data cells reload these best saved weights. The epoch count is a starting setting; use the validation loss to judge whether training should continue.

The property loss contains five terms:

- Binary cross-entropy for presence.
- Squared error for center, after division by 0.5 eV.
- Squared error for FWHM, after division by 1 eV.
- Squared error for area, after division by 2 relative-intensity × eV units.
- Shape cross-entropy, multiplied by 0.25.

Center, FWHM, area, and shape losses use only peaks that are truly present. The VAE also adds `0.1 × reconstruction MSE` and `0.001 × KL loss` to regularize its latent representation. The weights set the relative importance of the loss terms; they do not change the units shown in the evaluation table.

## Accuracy on held-out SESSA cases

Each notebook prints and saves a test table for every component and an `ALL` row. It reports:

| Metric | Interpretation |
| --- | --- |
| Presence precision | Among peaks called present, the share truly present. |
| Presence recall | Among truly present peaks, the share detected. |
| Presence F1 | Combined precision and recall. |
| Center MAE, eV | Average absolute center error for truly present peaks. |
| FWHM MAE, eV | Average absolute width error for truly present peaks. |
| Area MAE | Average absolute area error for truly present peaks, in relative-intensity × eV units. |
| Shape accuracy | Share of truly present peaks assigned the correct shape class. |

Property errors are calculated on truly present peaks **even if the presence decision was wrong**, so missed peaks do not disappear from the center, FWHM, area, or shape assessment. Presence uses the 0.5 threshold and the five-peak cap. The test table is saved as `outputs/sessa_models_v2_raw/<model>_test_metrics.csv` when a notebook runs.

## Hardest successes and easiest failures

Each notebook uses a simple peak formula to draw spectra from the **true** SESSA peak properties and from its **predicted** properties. It integrates the absolute difference between these two curves and divides by the integrated true curve area. The plot also shows the maximum-normalized noisy scan, with its floor, that was given to the model.

A successful scan has every peak-presence decision correct and a reconstruction error at or below the test median. An unsuccessful scan has a reconstruction error in the worst 10% of the test set. Among successful scans, lower input counts and more true peaks rank as harder. Among unsuccessful scans, higher counts and fewer true peaks rank as easier. Up to three distinct SESSA cases are shown for each group; unused panels stay blank when a small test set has fewer qualifying cases. The figure is saved as `outputs/sessa_models_v2_raw/<model>_best_worst.png`.

The true and predicted component curves are reconstructions from *peak-property tables*. They are not exact SESSA physical simulations: SESSA also includes instrument and scattering effects that the simple profile formula does not reproduce.

## Predictions on real measurements

The original sucrose and malonic acid `.xy` measurements from the [Zenodo dataset](https://zenodo.org/records/4624072) are saved in `datasets/MalonicSucrose/`. A separate `datasets/MalonicSucrose/real_c1s_spectra.csv` analysis file contains the six available C 1s spectra: sucrose at 0, 0.2, 1, and 5 mbar, and malonic acid at 0.2 and 1 mbar. It uses the first C 1s region in each file, averages repeated scans, converts kinetic to binding energy, and interpolates to the SESSA grid. The notebooks use `counts_per_second` and divide each scan by its maximum. The `MalonicAcid_0mbar.xy` file is also saved locally but has no C 1s region, so it is omitted from the analysis CSV.

For each real measurement, the prediction cell displays the predicted presence probability, presence decision, center, FWHM, area, and shape for each component. It does not export a peak table. These properties can be used later to construct approximate peak curves. A final plotting cell shows all six real model inputs, the predicted peak components, and the mapped components from the published fits. It saves `outputs/sessa_models_v2_raw/<model>_real_data.png` when the notebook runs.

The measured curves in those plots use maximum-normalized raw count rates, with the intensity floor retained. The published table provides peak centers, widths, and area fractions but not exact fitted Voigt curves, so the plotting cell draws approximate 50:50 pseudo-Voigt components. It aligns the published C–C center to the smoothed measured C–C maximum to remove the reported absolute-energy offset without using a model prediction. The published and predicted sums are separately scaled to a maximum of 1 for display; each set of component curves uses its sum's scale. The plotted peak sums do not include a fitted background. These plots therefore show peak locations and relative shapes, not agreement in absolute counts per second. Only model components selected by the 0.5 threshold and five-peak cap are drawn. Malonic acid has only a partial published component mapping.

The best/worst plots use a Gaussian (`η = 0`), Lorentzian (`η = 1`), or fixed 50:50 pseudo-Voigt (`η = 0.5`). With binding energy `E`, center `c`, FWHM `w`, curve height `H`, and Lorentz fraction `η`, the peak is:

```text
d = (E - c) / w
G(E) = exp(-4 ln(2) × d²)
L(E) = 1 / (1 + 4d²)
peak(E) = H × [(1 - η)G(E) + ηL(E)]
```

For the best/worst plots, `H` is calculated so that the curve integrates to the predicted area over the 280–300 eV grid. The pseudo-Voigt fraction is a fixed plotting choice, not a learned SESSA mixing parameter. This simple reconstruction does not include a fitted experimental background or absolute count calibration.

The [paper supplement](acp-21-4709-2021-supplement.pdf) contains experimental C 1s peak fits in Tables S3 (sucrose) and S5 (malonic acid). These values are preserved in `datasets/MalonicSucrose/real_c1s_published_fits.csv`. A second file, `datasets/MalonicSucrose/real_c1s_comparable_fits.csv`, transforms the 16 peaks with a usable model mapping into comparison columns. Sucrose's three peaks map to `CC`, `CO`, and `OCO`. For malonic acid, only adventitious `CC` and the acid peak are mapped; the notebooks change that peak's historical `COO` label to `COOH` in memory for v2. The molecular C–C and decomposition features have no separate model outputs. The malonic acid comparison is therefore partial.

The real-data comparison cell prints these mapped published peaks beside the model predictions. For each sample, it compares centers as distances from the C–C center and areas as fractions of the mapped peak areas. An absent predicted peak contributes zero area; center-spacing and width errors use detected peaks only. This avoids claiming a common absolute energy calibration or intensity scale. The [main article](https://acp.copernicus.org/articles/21/4709/2021/) gives 0 mbar sucrose centers 0.70 eV below the corresponding Table S3 values, so absolute-center error is not reported. The cell also reports mapped-peak detection, center-spacing MAE, area-share MAE, and a width MAE under an explicit unit assumption. These are descriptive comparisons for six measurements, not a held-out real-data accuracy estimate.

The supplement labels FWHM in eV but prints values near 1500–2200. The comparable table divides those numbers by 1000 under the assumption that they are meV. The paper describes source and instrumental broadening in meV, supporting this inference, but the authors do not provide a corrected table. The inferred experimental widths (1.67–2.20 eV) fit within the revised model's 0.5–3.0 eV output range. Shape is not scored because the experimental fits use Voigt profiles while the models predict broad shape classes and use a fixed 50:50 pseudo-Voigt only for plots.
