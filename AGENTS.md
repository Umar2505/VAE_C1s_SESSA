# Project guide

This bachelor's thesis predicts C 1s XPS peak properties from noisy spectra. The current notebooks are `c1s_peak_cnn.ipynb`, `c1s_peak_transformer.ipynb`, and `c1s_peak_vae.ipynb`. Keep the three models separate and runnable in Jupyter.

## Current pipeline

- `generate_sessa_training_dataset.py` generates seven possible C 1s components: `CC`, `CN`, `CO`, `CARBONYL`, `OCO`, `COO`, and `COOH`. At most five are present in a simulated case.
- The generator uses SESSA 2.3 with `MODEL SET NCOL 0 REGION 1`, so SESSA's inelastic-scattering background is disabled. It adds a linear background afterward. Each endpoint is independently sampled at 20–55% of that case's broadened SESSA peak maximum. These settings are illustrative until checked against representative experimental scans.
- Both the noise-free total and peak components are divided by the same maximum after the background is added. Three Poisson observations per case have expected total-spectrum maxima of 50, 500, and 5,000 counts. The saved `background_relative` array contains the added line. Background is not assigned to a peak component.
- The dataset path expected by all notebooks is `datasets/sessa_c1s/dataset.npz`. The dataset and model outputs are ignored by Git. Generate and train them locally; do not add bulk data to a push unless the user explicitly changes that decision.
- Inputs have 401 points from 280 to 300 eV. The notebooks divide each nonnegative observation by its own maximum and retain its intensity floor. No baseline is subtracted. All noisy observations of one case stay in the same train, validation, or test split.
- Each model predicts presence, center, effective FWHM, relative area, and shape for each component. The VAE also reconstructs its input as a regularizer. The peak table is the scientific output.

## Evaluation and scientific cautions

Read `MODEL_ARCHITECTURES_AND_PIPELINES.md` and the relevant notebook before changing model behavior. The original sucrose and malonic-acid measurements and the mapped comparison table are in `datasets/MalonicSucrose/`; read its README before interpreting comparisons. The published mapping is complete for sucrose and partial for malonic acid. Compare centers relative to `CC` and areas as shares of mapped peaks. The published FWHM values appear to use meV-sized numbers under an eV header, so dividing by 1,000 is an explicit assumption.

Held-out SESSA performance does not establish accuracy on measured XPS. Report presence precision, recall, and F1 plus center, FWHM, area, and shape errors separately. Show hard successful cases and easy failed cases. Notebook outputs from the earlier generator were cleared; retrain before calling any model score current.

Keep the work appropriate for a bachelor's thesis: clear, defensible, and no more complex than needed to predict and evaluate peaks. Do not change the 401-point input, three model types, or seven-component output without a reason grounded in data and discussion with the user. Keep generated bulk data out of Git and report exactly what is pushed.
