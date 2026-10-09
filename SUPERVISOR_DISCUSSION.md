# C 1s XPS spectra: SESSA dataset and VAE project

**Discussion guide and technical record — 4 October 2026.** This describes what the repository actually generates and trains, what the saved results show, and which experiments are still proposals. It is intended to support a supervisor discussion; the numerical ranges below are generator settings, not universally accepted physical constants.

## The one-minute explanation

The goal is to recover the underlying C 1s photoelectron peak signal from a noisy X-ray photoelectron spectroscopy (XPS) spectrum that also contains an inelastic-scattering background. I generate paired spectra with NIST SESSA: the full simulated signal and a corresponding zero-collision signal. I then make five noisy acquisitions of each physical case, train a variational autoencoder (VAE) to map the noisy full signal to the clean zero-loss signal, and judge it on physical cases withheld from training. The first 12,000 cases sampled a broad synthetic distribution. An error audit found that weak-count spectra with several narrow or strong aromatic-satellite features and shifted peaks were difficult. I added 2,000 cases enriched for those combinations. A first retraining helped the difficult cases but hurt many original cases. A newer run with gentler targeted sampling and peak-focused loss improved both groups relative to that first retraining. The separate effect of each training change is still unknown.

**Defensible claim:** this is a controlled synthetic inverse problem and a tested data-generation and evaluation pipeline. **Claim requiring more evidence:** that the model can accurately analyze measured atmospheric aerosol XPS or infer actual chemical composition, aging, or morphology.

## 1. Scientific problem and terms

XPS irradiates a surface with X-rays and measures emitted electrons. The C 1s region is associated with carbon core electrons. Their energies and line shapes can change with chemical environment. The nominal source in this project is Al Kα at 1486.6 eV. The generator converts SESSA's kinetic-energy output to its synthetic binding-energy axis as `1486.6 − kinetic_energy`. Real binding-energy calibration also needs attention to instrument work function, charging, and reference choices; the code's `charge_shift` is an imposed simulation shift, not a completed experimental calibration. [NIST XPS overview](https://www.nist.gov/laboratories/tools-instruments/x-ray-photoelectron-spectroscopy); [NIST energy-scale reference](https://nvlpubs.nist.gov/nistpubs/Legacy/NSRDS/nbsnsrds14.pdf).

SESSA (Simulation of Electron Spectra for Surface Analysis) simulates electron spectra and transport through specified samples. Its output depends on both the chosen sample model and the simulator's physical transport models. The present sample morphology is **homogeneous planar** for every case. The five Gaussian components, their fractions, density, and band gap are supplied by this project's generator; they are not discovered by SESSA. [NIST SESSA description](https://www.nist.gov/srd/nist-standard-reference-database-100).

| Term | Meaning here |
| --- | --- |
| **Physical case** | One chosen set of component fractions, centers, widths, charge shift, density, and related inputs. It receives one full and one zero-collision SESSA simulation. |
| **Acquisition / spectrum** | One count-scaled, noisy observation of a physical case. Five acquisitions share the same underlying case but have different count levels and noise. |
| **Full spectrum** | SESSA's simulation with its modeled inelastic transport/background. |
| **Zero-loss / zero-collision spectrum** | A second SESSA simulation with zero inelastic collisions requested; used as the clean peak-signal target. It is a simulation-defined target, not a directly measured ground truth. |
| **Peak / component** | A Gaussian feature with a nominal center and full width at half maximum (FWHM). Wider FWHM means a broader feature. |
| **π–π\*** | An aromatic shake-up satellite at higher binding energy. This project models it as a fifth Gaussian pseudo-component; that is a useful approximation for shape generation, not a complete physical treatment of the satellite. [Original shake-up study](https://journals.sagepub.com/doi/10.1366/0003702864509565). |
| **VAE** | A neural network whose encoder maps a spectrum to a probability distribution in a compact latent space and whose decoder reconstructs a spectrum from a sampled point. The standard objective combines reconstruction fit and a regularizing KL term. [Kingma and Welling](https://arxiv.org/abs/1312.6114). |

The five **generator labels** are C–C/C–H, C–O/C–N, C=O/O–C–O, O–C=O/COO−, and π–π\*. Some labels cover multiple possible chemical environments; the simulated peaks alone cannot distinguish all of them. The model presently predicts a **curve**, not five independently identified chemical concentrations.

```text
draw a physical case → two SESSA simulations → common 280–300 eV grid
                           ↓
             full + zero-loss + generator metadata
                           ↓
                5 count-scaled noisy acquisitions
                           ↓
       HDF5 files → case-level train / validation / test
                           ↓
       noisy full spectrum → VAE → predicted zero-loss curve
                           ↓
      error by physical case, count level, and morphology
```

## 2. How the first 12,000 cases were designed

The script [generate_sessa_batch.py](sessa_atmospheric/generate_sessa_batch.py) draws each case with random seed `20260929`. These distributions create **synthetic diversity**. The repository does not contain an empirical calibration study proving that their limits match a particular atmospheric sample population. The rounded reference centers represent working C 1s component positions; they should be checked against relevant reference data and measured standards before making chemical-identification claims. [NIST XPS database description](https://www.nist.gov/srd/materials).

| Generator setting | Implemented value | Purpose and interpretation |
| --- | --- | --- |
| Reference binding-energy centers | `284.8, 286.25, 287.75, 289.05, 291.2 eV` | Places five nominal components in a plausible C 1s window. These are project choices, not fixed values for every material. |
| Synthetic `aging` index | `Beta(2,2)`, between 0 and 1 | A mixing variable that tends toward middle values. It controls the Dirichlet fraction parameters; it is **not elapsed time**, exposure dose, or a measured age. |
| Nominal component fractions | Dirichlet draw among active components; parameter vector shown below | Produces positive fractions summing to one and correlates more oxygen-containing groups with larger synthetic aging. These are generator inputs, not fitted spectral areas. |
| Peak presence | C–C/C–H always on; each of components 2–4 independently omitted with probability 0.12; π–π\* active with probability 0.30; at least two active peaks | Creates missing-feature cases and some rare aromatic shapes. An absent peak is passed to SESSA at tiny `10⁻⁶` fraction for numerical practicality but retained as zero in the dataset label. |
| Common energy shift | Uniform from `−1.5` to `+2.5 eV` | Emulates an imposed charging/reference shift. All components move together, then receive small individual jitter. |
| Individual center jitter | Normal standard deviations `0.10, 0.16, 0.18, 0.18, 0.25 eV` | Stops every case from having identical peak spacing. |
| Widths | Common FWHM uniform `0.90–1.60 eV`, multiplied by independent Normal(1, 0.10), clipped to `0.70–2.00 eV` | Covers overlapping and relatively narrow/broad components. SESSA Gaussian sigma receives `FWHM / 2.35482`. |
| O:C proxy | `clip(0.03 + f(C–O) + f(C=O) + 2 f(O–C=O), 0.03, 1.20)` | Heuristic composition input for SESSA's material/transport calculation. This is **not an independent measured atomic O:C ratio**. |
| Mass density | Uniform `1.10–1.75 g/cm³` | Varies a SESSA material input; this range is an assumption needing sample-specific support. |
| Band gap | `0 eV` if nominal π–π\* fraction exceeds `0.08`; otherwise uniform `2–7 eV` | Simple heuristic linking aromatic-like cases to a different band-gap setting. Not an experimentally measured band gap. |

For active peaks, the Dirichlet concentration vector is

`[3.5(1−a)+0.35, 0.7+2.2a, 0.4+1.7a, 0.2+1.8a, 0.15+0.45(1−a)]`, where `a` is the synthetic aging index. Larger `a` tends to reduce the first component's share and increase oxygen-containing components, but individual random draws can vary considerably. Dirichlet concentration parameters are distribution controls, **not observed fractions**. The derived `atmospheric_family` name in the HDF5 file is assigned by simple π-fraction and aging thresholds; it is an organizational label, not independent sample truth.

The SESSA sessions use Al Kα, a kinetic-energy window of `1186.6–1206.6 eV`, a homogeneous first layer, Gaussian subpeaks, and automatic trajectories/collisions for the full run. They request `NDIIMFP 1`, `SLA false`, `TA false`, `SE true`, `QEAPLUS true`, and convergence `10⁻²`; their scientific implications should be checked against the installed SESSA manual before interpreting fine physical effects. The script sets relative subpeak `HEIGHT 1.0`; **detector-count peak heights are obtained later**, after fitting the zero-loss spectrum and applying count scaling. It then requests `NCOL 0` for the zero-collision counterpart. The initial session was divided into eight shards and run with [run_sessa_shards.py](sessa_atmospheric/run_sessa_shards.py), which skips already-complete cases and retries failures. The HDF5 files identify the simulator as SESSA **2.3.0**. [NIST SESSA user guide](https://nvlpubs.nist.gov/nistpubs/NSRDS/NIST.NSRDS.100-2024.pdf).

## 3. From simulator output to paired training data

[build_vae_dataset.py](sessa_atmospheric/build_vae_dataset.py) reads the SESSA files, changes kinetic to binding energy as described above, and interpolates both simulations to **401 points from 280 to 300 eV inclusive** (`0.05 eV` spacing). Negative simulator output is clipped to zero; zero-loss values are also capped by full-spectrum values. Interpolation gives every example the same input length, but it does not add physical energy resolution.

[build_h5_dataset.py](sessa_atmospheric/build_h5_dataset.py) stores one HDF5 file per physical case with five acquisitions. For acquisition `j`, a log-uniform target maximum sets a scale `s_j = target_max / max(full)`. The stored clean target is `zero_loss × s_j`; the expected full signal is `full × s_j`; the nonnegative modeled background is `max(full − zero_loss, 0) × s_j`. The noisy input is `max(Poisson(expected_full) + Normal(0, σ_read), 0)`, with `σ_read` uniform from `0` to `1.5` counts. The first 12,000 cases have target maxima drawn log-uniformly from `200` to `150,000` counts. This makes count regime part of the experiment; it does not reproduce every real detector artifact. The two outputs from SESSA share the same physical case, so the training pair is aligned by construction.

The HDF5 file contains the energy grid; noisy, expected-full, clean-zero-loss, and background arrays; nominal peak centers, FWHM, presence and fractions; sample inputs; count/noise settings; and identifiers. To obtain a component **height** in counts, the packaging code fits nonnegative Gaussian amplitudes at the generator's known centers and widths to the zero-loss spectrum, records fit NRMSE, and multiplies by `s_j`. This height is an estimated isolated component height, whereas `area_fraction` is the **nominal material fraction**. They should not be equated. New files store peak heights directly (`schema 1.1.0`); the loader derives them for original `schema 1.0.0` files when needed. [atmospheric_dataset.py](atmospheric_dataset.py) reads and checks these files.

The **split is by physical case**, so all five noisy acquisitions of one case stay in the same set. This prevents the same underlying peak shape from appearing as both training and test data merely with different noise. The initial cases were randomly assigned using seed `20260930`, approximately `70% train / 15% validation / 15% test`. Validation chooses settings and checkpoints; test is for final assessment.

## 4. Why and how 2,000 targeted cases were added

The earlier model's error audit, recorded by [evaluate_vanilla_slices.py](sessa_atmospheric/evaluate_vanilla_slices.py), highlighted weak count levels and combinations of strong π–π\*, narrow or multiple peaks, and shift extremes. The goal of [continue_dataset.py](sessa_atmospheric/continue_dataset.py) was to increase the *frequency of those combinations* without regenerating cases `0–11999`. It drew from the **same parameter ranges** as the original generator and accepted draws that met five conditions. Thus it does not expand the allowed physics, energy range, or morphology.

| Continuation stratum | Condition in addition to ≥4 active peaks | New cases |
| --- | --- | ---: |
| Strong π, negative shift | π fraction `≥0.15`; shift `<−0.5 eV` | 400 |
| Strong π, central shift | π fraction `≥0.15`; shift `−0.5 to +1.5 eV` | 400 |
| Strong π, positive shift | π fraction `≥0.15`; shift `>+1.5 eV` | 400 |
| Moderate π, positive shift | π fraction `0.03–<0.15`; shift `>+1.5 eV` | 400 |
| Trace/absent π, several peaks, positive shift | π fraction `<0.03`; shift `>+1.5 eV` | 400 |

The last internal stratum is named `nonaromatic_multiplet_positive`; the actual condition allows **trace** π fraction, so it is not a proof of a nonaromatic material. Each stratum was divided into `280 train / 60 validation / 60 test`. The five acquisitions per new case emphasize weak spectra: **four** target maxima are log-uniform from `200–1,000` counts and **one** from `1,000–10,000` counts. These 2,000 cases contribute 10,000 acquisitions. The final index contains **14,000 physical cases, 70,000 acquisitions**, with `9,753 train`, `2,139 validation`, and `2,108 test` physical cases. The original case files were kept intact. Seeds and case IDs make the continuation reproducible and auditable.

The important statistical unit is the physical case: 70,000 acquisitions do **not** provide 70,000 independent shapes. Also, difficult groups were selected after inspecting prior model errors, including earlier test results. The final test set is case-disjoint, but it is no longer wholly untouched by the overall research process. A newly frozen external simulator sample and, ideally, real measurements would give stronger final evidence.

## 5. The current VAE pipeline

The working [atmospheric_vanilla_vae.ipynb](atmospheric_vanilla_vae.ipynb) reads the HDF5 set, sends the **noisy full** 401-point curve to the encoder, and uses the **clean zero-loss** 401-point curve as its target. Sample metadata and peak labels do **not** enter the encoder. The input and target are each transformed approximately as `T(c) = 2√(max(c,0)+3/8) / S`, then clipped to `[0,1]`; `S` is determined from training data only (the prior recorded value was about `813.30`). This Anscombe-style transform compresses the wide count range and approximately stabilizes Poisson variance. Because this dataset also adds Gaussian read noise and clips counts, stabilization is approximate. [Anscombe's original paper](https://academic.oup.com/biomet/article-abstract/35/3-4/246/280278).

The encoder has fully connected widths `401 → 256 → 128 → 64` with ReLU activations. It outputs a 12-dimensional mean `μ` and log-variance `log σ²`. During training it samples `z = μ + σϵ`, with `ϵ` drawn from a standard normal distribution; this is the **reparameterization trick**, which lets gradients pass through sampling. The decoder maps `12 → 64 → 128 → 256 → 401`. At evaluation it decodes `μ` to get a deterministic curve, then reverses the transform to count units. The model has about **291 thousand trainable parameters**. [VAE derivation](https://arxiv.org/abs/1312.6114).

With optional peak loss set to zero, the per-batch objective is a Gaussian reconstruction term `Σ_i(ŷ_i−y_i)² / (2×0.03²)` plus `KL[q(z|x) || N(0,I)]`, averaged across examples. The first term rewards a close clean-curve prediction in transformed units. The KL term keeps latent distributions near a standard normal prior; it can also encourage smoother or more average-looking predictions. This is a **conditional denoising-style use of a VAE**: the input is noisy full signal and the reconstruction target is a different clean signal. It is not evidence that the decoder has identified unique physical chemistry.

The notebook now includes an **optional third term**. It forms Gaussian windows around the known training-case component centers and widths, measures squared prediction error there relative to that target's maximum amplitude (floor `0.02` in transformed units), and multiplies it by `PEAK_LOSS_WEIGHT` (currently `5`). This aims to reduce broad or merged predictions where narrow components matter. With a positive weight, the training objective is **not the strict vanilla VAE ELBO**; it is a VAE architecture with an auxiliary peak-aware supervised loss. The peak labels guide training loss only; they are not needed to predict a new input. Setting `PEAK_LOSS_WEIGHT=0` restores the standard two-term objective for a controlled comparison.

Training uses Adam at `3×10⁻⁴`, at most 150 epochs, gradient clipping at 5, seed 42, and early stopping after 20 epochs without a meaningful validation-score improvement (`0.0002`). An **epoch** is one pass through the training draws. A **checkpoint** is saved model weights at the best validation score, so the final epoch need not be the selected model. The latest run sampled 20% continuation acquisitions in training draws, compared with their natural case share of about 14.3%, and used peak-loss weight `5`. The checkpoint score is `0.8 × median validation NRMSE(original cases) + 0.2 × median validation NRMSE(continuation cases)`. This makes the preferred tradeoff explicit. The run with the **same 20% sampler but peak-loss weight `0` remains to be done**; it is needed to isolate the peak term's contribution.

## 6. What the saved results do and do not show

The comparable audit evaluated three checkpoints on the **same 2,108 physical test cases** from the combined dataset. NRMSE is root-mean-square error between predicted and clean curves divided by the clean curve's intensity range (with a denominator floor of one count). Lower is better. The following values are medians over acquisitions, not estimates of real-world chemical accuracy.

| Test group | 12,000-case checkpoint | First 14,000-case checkpoint | Latest 14,000-case run: 20% targeted, peak weight 5 |
| --- | ---: | ---: | ---: |
| All test acquisitions | `0.0373` | `0.0465` | **`0.0358`** |
| Original cases | `0.0327` | `0.0428` | **`0.0335`** |
| New targeted cases | `0.0780` | `0.0693` | **`0.0520`** |
| Weak counts + strong π | `0.0974` | `0.0769` | **`0.0579`** |
| Weak counts + strong π + narrow peaks | `0.1213` | `0.0938` | **`0.0775`** |
| High-count acquisitions | `0.0217` | `0.0304` | **`0.0287`** |

The latest run lowered pooled median NRMSE by about **23% relative to the first 14,000-case run** and about **4% relative to the 12,000-case model**. Its 90th-percentile NRMSE is `0.0713`, down from `0.0898` in the first retraining; median peak-height error is `7.61%`, down from `9.82%`. Across physical cases, 70% of original cases and 83% of continuation cases improved relative to the first retraining. Original-case accuracy remains slightly worse than the 12,000-case model (`0.0335` versus `0.0327`), and high-count accuracy remains substantially worse (`0.0287` versus `0.0217`). The 53 test cases in the weak, strong-π, narrow-peak slice still have median NRMSE `0.0775` and median peak-height error `24.5%`.

**Interpretation:** the extra targeted data helped its intended region, and the latest training recipe recovered much of the original-cohort accuracy. However, the latest run changed **both** the training sampling share and the loss. It therefore cannot isolate which change produced the gain. The older first retraining also changed data and sampling together. The current notebook contains the latest saved output; the earlier saved figures are in [the archived 14,000-case notebook](outputs/atmospheric_vanilla_vae/atmospheric_vanilla_vae_14000_saved_outputs.ipynb). A local copy of the latest checkpoint and its history is in [outputs/best_zeroloss](outputs/best_zeroloss), although the notebook's printed checkpoint path refers to the machine on which training ran.

**Slight overfitting near 150 epochs** means training fit continues to improve while held-out validation fit levels off or deteriorates. In the latest run, the best validation selection score was `0.03685` at epoch **133**; it rose to `0.03796` by epoch 150 while training loss kept falling. This is mild late-stage overfitting, and the saved best checkpoint avoids using the final weights. The model ran all 150 epochs because the patience of 20 had not elapsed since epoch 133. The worst displayed original cases include **7906** and **3860**: even relatively clean, high-count inputs have narrow separated peaks that the VAE broadens, merges, or shifts. This points to a shape-reconstruction limitation, not merely inadequate count statistics. The latent audit found 7 of 12 mean coordinates varying appreciably; simply increasing latent width is unlikely to be the first fix.

## 7. Limitations to state plainly

1. **Synthetic-to-real gap.** SESSA transport is physically motivated, but the sample priors, peak decomposition, count scaling, and noise are designed here. No measured aerosol spectra validate the model yet.
2. **One morphology.** Every case is homogeneous planar. The current project does not teach the model layered films, particles, roughness, or changing take-off angles, although SESSA can represent more sample structures. [NIST SESSA description](https://www.nist.gov/srd/nist-standard-reference-database-100).
3. **Heuristic chemical labels.** Synthetic aging, O:C, band gap, and `atmospheric_family` are derived or sampled; they must not be presented as measured properties. A nominal π pseudo-component is a simplified representation of a shake-up satellite.
4. **Limited measurement model.** Fixed Al Kα source and 280–300 eV window; simplified Poisson plus Gaussian noise; no measured instrument transmission, response function, calibration drift, contamination, or real sample preparation effects.
5. **Nonunique inverse problem.** Different nearby component mixtures can yield similar full spectra, especially at low counts. A plausible reconstructed curve is not necessarily a unique composition estimate.
6. **Selection bias.** Earlier test-set observations helped choose continuation strata. Further optimization on the same test set would overstate final generalization. Preserve a new final holdout for the end of the study.

## 8. A practical next-experiment plan

| Priority | Experiment | What would count as progress |
| --- | --- | --- |
| 1 | Keep the same 14,000 cases and 20% sampler; run `PEAK_LOSS_WEIGHT=0` with the same seed and checkpoint-selection rule. | The difference from the completed peak-weight-5 run estimates what the peak loss adds. Compare original, targeted, high-count, and narrow-peak cases. |
| 2 | On validation data, compare a deterministic denoising autoencoder or direct regression model, and a VAE with weaker KL regularization. | See whether the VAE bottleneck/regularization causes broadening when sharp peaks are visible in the input. Test one change at a time. |
| 3 | Repeat promising settings with several random seeds and case-level uncertainty estimates. Freeze a new simulator holdout **before** looking at its results. | Improvement persists across seeds and on independent cases rather than depending on a lucky training run. |
| 4 | Compare simulated curves and labels with measured reference spectra; establish actual energy calibration, realistic count/noise distributions, and fitting conventions. | The chosen generator ranges and evaluation metrics match the intended instrument and sample type. |
| 5 | If validation still shows a coverage gap, add independent physical cases that explicitly cover **narrow peaks at both high and low counts** and varied shifts; then broaden physical mechanisms where evidence supports it. | Reduce the persistent sharp-peak failures. The existing continuation selected π and shift strata but did not directly require narrow widths. |

The key dataset-size distinction is **independent shapes versus repeated acquisitions**. If failures persist because an important regime is rare, add targeted physical cases. If failures come from unrealistically specified physics, generating millions more examples from the same assumptions will reinforce that mismatch. If failures occur on already well-covered original cases, adjust sampling and loss before expanding the dataset.

## 9. Questions to rehearse with the supervisor

**Why use SESSA?** It gives paired full and zero-collision spectra for controlled inputs, enabling supervised evaluation of background removal/denoising. It also encodes electron transport that a simple sum of Gaussians would omit. Its output remains conditional on our chosen sample model.

**Why a VAE instead of ordinary regression?** A VAE provides a compact probabilistic latent representation and a regularized decoder. That may help capture a family of spectral shapes, but it has a real cost: the KL term can smooth sharp or rare features. A deterministic denoising autoencoder or direct regression network is an appropriate future baseline; the current results do not prove that a VAE is optimal.

**Why clean zero-loss as target?** The research question is recovery of peak signal from noisy full signal. `expected_total_counts` remains in the file and could support a different denoising-only task. The target choice must be stated because it changes what a low error means.

**Are the 70,000 spectra independent?** No. They are five acquisitions of each of 14,000 physical cases. This is why splitting by case ID is essential and why a dataset-size claim should name both counts.

**Did the 2,000 cases work?** They improved the intentionally targeted difficult cases in the first retraining, including weak strong-π and narrow-peak subsets, but that first recipe hurt original cases. The latest run improves both groups relative to the first retraining. It still falls slightly short of the original model on original cases, and the exact contribution of its sampler versus peak loss is unresolved.

**Could the model just learn the generator?** Yes. A strong score on held-out samples from the same generator proves performance within that synthetic distribution, not across instruments or real atmospheric samples. External validation is a core next step.

**How were parameter values justified?** The numerical values are explicit working priors chosen to span recognizable C 1s-like shapes and signal levels; the repository does not document a sample-specific fit or calibration for each bound. SESSA's transport models are independent of that choice. The honest plan is to compare these priors with reference and measured spectra and revise them where evidence disagrees.

**Should the dataset contain other information?** For stronger physical interpretation, record instrument response/calibration, source and acquisition geometry, sample morphology and thickness, independently measured elemental ratios/chemistry where available, provenance and uncertainty of peak assignments, and real background/noise characteristics. These should be genuine inputs or verified metadata, not inferred labels presented as measurements.

## 10. Where to inspect the evidence

- Case drawing and SESSA settings: [generate_sessa_batch.py](sessa_atmospheric/generate_sessa_batch.py).
- Continued generation and strata: [continue_dataset.py](sessa_atmospheric/continue_dataset.py).
- Output conversion and peak-height fitting: [build_vae_dataset.py](sessa_atmospheric/build_vae_dataset.py).
- HDF5 schema, noise model, and split: [build_h5_dataset.py](sessa_atmospheric/build_h5_dataset.py); [dataset_index.csv](atmospheric_c1s_h5/dataset_index.csv).
- Latest completed peak-aware experiment: [atmospheric_vanilla_vae.ipynb](atmospheric_vanilla_vae.ipynb); local [checkpoint and history](outputs/best_zeroloss). Prior saved figures: [14,000-case notebook](outputs/atmospheric_vanilla_vae/atmospheric_vanilla_vae_14000_saved_outputs.ipynb).
- Error slicing and raw summaries: [evaluate_vanilla_slices.py](sessa_atmospheric/evaluate_vanilla_slices.py); [combined test slices](outputs/atmospheric_vanilla_vae/error_slices_combined_test.csv). The saved `.pt` checkpoint is model weights, not a dataset or a new notebook run.

**Suggested opening to the meeting:** “I built a reproducible SESSA-based paired dataset to learn clean C 1s peak signal from noisy full spectra. Error analysis showed specific weak, multi-peak failure modes, so I added 2,000 targeted physical cases. The first retraining improved those cases but hurt common ones. A newer run improved both groups relative to that first retraining, although sharp, separated peaks remain difficult. My next controlled comparison will isolate the peak-loss effect. I would like to agree on real-sample validation and which simulator assumptions need experimental support.”
