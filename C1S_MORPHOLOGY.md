# C 1s peak morphology VAE

The [peak-parameter notebook](c1s_lmfit_vae.ipynb) reads the generated `.nxs` file and predicts six components' presence, profile family, binding-energy center, FWHM, lmfit shape parameters, height, integrated area, and area fraction from noisy synthetic C 1s spectra. Its encoder maps the input counts to a variational latent space, and its decoder outputs only peak data. It has no spectrum reconstruction objective. This is a supervised variational encoder–decoder with a VAE-style latent space, not a conventional input-reconstructing VAE. The peak labels are the parameters used to generate the spectra, not results fitted back from noisy data.

## Dataset

[`c1s_lmfit_dataset.py`](c1s_lmfit_dataset.py) is the single-file generator for the next dataset. It samples hydrocarbon-rich, oxidized organic, and soot-rich cases with lmfit Gaussian, Voigt, and pseudo-Voigt components. Ordinary peak shapes are sampled per case with illustrative probabilities of 30%, 35%, and 35%, respectively. Only the soot-rich graphitic component uses a Gaussian-broadened Doniach asymmetric profile. A π–π* satellite can appear only with its aromatic/graphitic parent. The generator varies charging shift, peak widths, linear, Shirley, and Tougaard backgrounds, native scan spacing, count level, and Poisson noise. Repeated acquisitions of one case stay in the same train, validation, or test split.

```bash
python c1s_lmfit_dataset.py --cases 1000 --replicates 5
```

When run, the script writes `datasets/c1s_nexus/c1s_atmospheric_v2.nxs` and refuses to overwrite an existing file. The NeXus/HDF5 file contains a common-grid noisy spectrum and clean expectation in `/entry/data`, native acquisitions in `/entry/acquisition`, ground-truth labels in `/entry/labels`, and per-case component profiles and provenance in `/entry/simulation`. Component areas refer to the finite 280–300 eV window. The numeric ranges and chemical-family probabilities are synthetic design choices to calibrate against measured spectra.

The background functions generate a forward signal from the known clean peaks. They follow the linear, Shirley, and Tougaard families compared in the ATMOS background-subtraction notebook; they do not run background removal on measured data. The Tougaard kernel uses the universal B/C form inside the 20 eV simulation window, so it should be checked against wider measured scans before treating its shape as representative of an instrument or sample.

The existing `datasets/c1s_lmfit_morphology.npz` archive remains the Gaussian-only baseline with 1,000 cases and 5,000 spectra. The notebook now expects `datasets/c1s_nexus/c1s_atmospheric_v2.nxs` and will report a missing-file error until the generator is run. Training outputs go into `outputs/c1s_nexus_peak_vae/`. The final notebook cell takes predicted peak properties for a held-out scan as starting values, uses lmfit to test linear, Shirley, and Tougaard backgrounds, and saves the fitted spectrum, component curves, plot, and peak parameters to that output directory. It demonstrates fitting one synthetic scan; measured XPS spectra still require independent validation.

The generator is a controlled test of peak-parameter recovery. Results on these synthetic spectra alone do not establish performance on measured XPS spectra.
