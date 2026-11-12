# Real C 1s measurements

The seven original `.xy` files in this directory are the sucrose and malonic acid measurements from the [Zenodo record](https://zenodo.org/records/4624072) accompanying [the ACP paper](https://acp.copernicus.org/articles/21/4709/2021/). Their file sizes and MD5 checksums were checked against the Zenodo record. The filename `Sucrose_1mbar .xy` contains a space before `.xy` in the original record.

`real_c1s_spectra.csv` is the analysis table used by all three model notebooks. It contains six C 1s measurements, each on a 280–300 eV binding-energy grid at 0.05 eV intervals (401 points). For each original file, the first C 1s region was selected, count rates at repeated kinetic energies were averaged, binding energy was calculated as excitation energy minus kinetic energy, and the result was interpolated to this grid. `counts_per_second` is the averaged, interpolated measurement.

`MalonicAcid_0mbar.xy` is retained as an original file, but it has no C 1s region, so it has no rows in `real_c1s_spectra.csv`.

## Model input

The notebooks apply `x / max(max(x), 0.000001)` to each spectrum's nonnegative `counts_per_second` values. They keep the intensity floor, as they do for synthetic noisy counts. This is per-spectrum scaling, so the original count-rate differences between samples are not retained in the model input. The original `.xy` files and `counts_per_second` column retain those count rates. Peak curves reconstructed for comparison do not include a fitted background, so their sums are not expected to reproduce the retained floor.

The [paper supplement](../../acp-21-4709-2021-supplement.pdf) reports C 1s fit values in Tables S3 and S5. Its FWHM header says eV although the printed values are around 1500–2200 and appear to be in meV. The paper's 0 mbar sucrose centers also differ from Table S3 by 0.70 eV.

`real_c1s_comparable_fits.csv` is a derived table for comparing the published fits with model predictions. It includes the 16 peaks that have a usable model-component mapping: sucrose peaks 1–3 map to `CC`, `CO`, `OCO`; malonic acid peak 1 maps to `CC` and peak 2 to `COOH`. Malonic acid's second C–C peak and decomposition feature have no separate model outputs, so its comparison is marked `matched_peaks_only`. The 0 mbar malonic acid fits are excluded because no raw C 1s spectrum is available for prediction.

## How published fits and model outputs are compared

The input normalization above prepares spectra for prediction. **It does not by itself make published peak areas comparable with model areas.** The comparison cell of each notebook applies these additional transformations to the mapped peaks of each sample:

| Property | Published fit | Model prediction |
| --- | --- | --- |
| Peak-center spacing, eV | `peak center − published CC center` | `predicted peak center − predicted CC center` for detected peaks, if predicted CC is detected |
| Peak-area share | `published peak area / sum of published areas for mapped peaks` | Zero for an absent predicted peak; otherwise its predicted area divided by the sum of detected mapped-peak areas |
| Total FWHM, eV | `printed total FWHM / 1000`, assuming the printed numbers are meV | Predicted FWHM directly in eV for detected peaks |
| Presence | A peak included in the published fit is treated as present | Presence probability is called present at 0.5 or above, keeping at most the five highest probabilities |

For example, sucrose at 0 mbar has a published C–O center of 287.36 eV and C–C center of 285.90 eV, so its center spacing is **1.46 eV**. Its C–O area share is `728.57 / (4207.45 + 728.57 + 262.60) = 0.1401`. Its printed total FWHM of 2199.89 becomes **2.19989 eV** under the meV assumption. The notebook computes the corresponding model values using the predicted C–C center, the predicted areas of `CC`, `CO`, and `OCO`, and the model FWHM.

For sucrose, all three published C 1s peaks enter the area-share denominator. For malonic acid, only `CC` and `COOH` enter it in the notebooks; the molecular C–C and decomposition peaks are excluded on both sides because the models have no separate outputs for them. This makes the malonic acid comparison partial. Area shares are dimensionless and are not atomic percentages. Subtracting the C–C center removes a common energy offset, so absolute binding-energy agreement is not measured.

The area-share error includes all mapped published peaks, counting a missed model peak as zero area. Center-spacing and width errors include only peaks that the model detects; center spacing is unavailable if it misses the C–C anchor. Properties printed for absent model peaks are raw network outputs and are ignored in both the comparison and reconstructed curves. The notebooks also count any additional detected sucrose components outside its published three-peak fit. Those extra components are reported separately and do not enter the mapped-peak area shares. Malonic acid has only a partial component mapping, so its unmapped components are not counted as extras.

The FWHM division is an inference from the printed magnitudes and the paper's description of broadening, not a corrected value supplied by the authors. Shape is not compared because the experimental fits use Voigt functions while the models predict broad shape classes. The models can output FWHM from 0.5 to 3.0 eV, which includes the inferred experimental widths here (1.67–2.20 eV).
