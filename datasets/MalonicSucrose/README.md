# Real C 1s measurements

The seven original `.xy` files in this directory are the sucrose and malonic acid measurements from the [Zenodo record](https://zenodo.org/records/4624072) accompanying [the ACP paper](https://acp.copernicus.org/articles/21/4709/2021/). Their file sizes and MD5 checksums were checked against the Zenodo record. The filename `Sucrose_1mbar .xy` contains a space before `.xy` in the original record.

`real_c1s_spectra.csv` is the separate analysis table used by all three model notebooks. It contains six C 1s measurements, each on a 280–300 eV binding-energy grid at 0.05 eV intervals (401 points). For each original file, the first C 1s region was selected, count rates at repeated kinetic energies were averaged, binding energy was calculated as excitation energy minus kinetic energy, and the result was interpolated to this grid. `counts_per_second` is the averaged, interpolated measurement. `model_input` is a historical endpoint-baseline-subtracted, clipped, maximum-normalized column used by the v1 models. The v2 notebooks use `counts_per_second` instead.

`MalonicAcid_0mbar.xy` is retained as an original file, but it has no C 1s region, so it has no rows in `real_c1s_spectra.csv`.

## Saved v1 normalization and current v2 input

Each C 1s region contains multiple scans at the same kinetic energies. For every kinetic energy, their counts per second were averaged. Binding energy was then calculated as `excitation energy − kinetic energy` (the files give 1253.64 eV excitation energy), and the averaged counts were linearly interpolated onto 401 points from 280 to 300 eV. These interpolated values are saved unchanged in the `counts_per_second` column.

For **each spectrum separately**, let `x[i]` be the interpolated counts per second at grid point `i`, where `i = 0, …, 400`. The historical `model_input` column was calculated as:

```text
left  = mean(x[0:10])                 # first 10 points
right = mean(x[391:401])              # last 10 points
baseline[i] = left + (right − left) × i / 400
corrected[i] = max(x[i] − baseline[i], 0)
model_input[i] = corrected[i] / max(max(corrected), 0.000001)
```

The baseline is a straight line between the two endpoint means. Clipping removes values below that line. Dividing by that spectrum's maximum puts `model_input` between 0 and 1; if every corrected value is zero, the result stays zero. This is **per-spectrum scaling**, so the original count-rate differences between samples are not retained in `model_input`. The original `.xy` files and the `counts_per_second` column retain those count rates. No dataset-wide mean or standard deviation was used.

The v2 notebooks apply `x / max(max(x), 0.000001)` directly to the nonnegative `counts_per_second` values. They do not subtract a baseline or clip the intensity floor. This matches how the v2 synthetic noisy counts are prepared before training. Peak curves reconstructed for comparison do not include a fitted background, so their sums are not expected to reproduce the retained floor.

`real_c1s_published_fits.csv` transcribes the reported C 1s fit values from Tables S3 and S5 of the [paper supplement](../../acp-21-4709-2021-supplement.pdf). Its FWHM columns preserve the numbers **as printed**: the table header says eV although the printed values are around 1500–2200 and appear to be in meV. The paper's 0 mbar sucrose centers also differ from Table S3 by 0.70 eV. `raw_c1s_available` is 0 for the malonic acid 0 mbar fit because the corresponding `.xy` file has no C 1s region.

`real_c1s_comparable_fits.csv` is a derived table for comparing the published fits with the model predictions. It includes the 16 peaks that have a usable model-component mapping: sucrose peaks 1–3 map to `CC`, `CO`, `OCO`; malonic acid peak 1 maps to `CC` and peak 2 has the historical `COO` label. The v2 notebooks map that acid peak to `COOH` in memory, leaving the source CSV intact. Malonic acid's second C–C peak and decomposition feature have no separate model outputs, so its comparison is marked `matched_peaks_only`. The 0 mbar malonic acid fits are excluded because no raw C 1s spectrum is available for prediction.

## How published fits and model outputs are compared

The input normalization above prepares spectra for prediction. **It does not by itself make published peak areas comparable with model areas.** The comparison cell of each notebook applies these additional transformations to the mapped peaks of each sample:

| Property | Published fit | Model prediction |
| --- | --- | --- |
| Peak-center spacing, eV | `peak center − published CC center` | `predicted peak center − predicted CC center` for detected peaks, if predicted CC is detected |
| Peak-area share | `published peak area / sum of published areas for mapped peaks` | Zero for an absent predicted peak; otherwise its predicted area divided by the sum of detected mapped-peak areas |
| Total FWHM, eV | `printed total FWHM / 1000`, assuming the printed numbers are meV | Predicted FWHM directly in eV for detected peaks |
| Presence | A peak included in the published fit is treated as present | Presence probability is called present at 0.5 or above, keeping at most the five highest probabilities |

For example, sucrose at 0 mbar has a published C–O center of 287.36 eV and C–C center of 285.90 eV, so its center spacing is **1.46 eV**. Its C–O area share is `728.57 / (4207.45 + 728.57 + 262.60) = 0.1401`. Its printed total FWHM of 2199.89 becomes **2.19989 eV** under the meV assumption. The notebook computes the corresponding model values using the predicted C–C center, the predicted areas of `CC`, `CO`, and `OCO`, and the model FWHM.

For sucrose, all three published C 1s peaks enter the area-share denominator. For malonic acid, only `CC` and `COOH` enter it in the v2 notebooks; the molecular C–C and decomposition peaks are excluded on both sides because the models have no separate outputs for them. This makes the malonic acid comparison partial. Area shares are dimensionless and are not atomic percentages. Subtracting the C–C center removes a common energy offset, so absolute binding-energy agreement is not measured.

The area-share error includes all mapped published peaks, counting a missed model peak as zero area. Center-spacing and width errors include only peaks that the model detects; center spacing is unavailable if it misses the C–C anchor. Properties printed for absent model peaks are raw network outputs and are ignored in both the comparison and reconstructed curves. The notebooks also count any additional detected sucrose components outside its published three-peak fit. Those extra components are reported separately and do not enter the mapped-peak area shares. Malonic acid has only a partial component mapping, so its unmapped components are not counted as extras.

The FWHM division is an inference from the printed magnitudes and the paper's description of broadening, not a corrected value supplied by the authors. Shape is not compared because the experimental fits use Voigt functions while the models predict broad shape classes. The v2 models can output FWHM from 0.5 to 3.0 eV, which includes the inferred experimental widths here (1.67–2.20 eV).
