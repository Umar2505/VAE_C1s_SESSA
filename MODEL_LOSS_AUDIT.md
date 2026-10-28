# C 1s peak predictor: loss and failure audit

This review uses the saved checkpoint, training history, and 5,000 held-out synthetic scans from the current generator. It includes inference and analysis only. **No training or optimizer updates were run on this machine.** The test cases are separate from training cases, but they come from the same synthetic generator. They do not measure accuracy on external or measured XPS spectra.

## What the model currently learns

The notebook's model maps a noisy C 1s spectrum to six chemically named peak slots. Each slot predicts presence, profile family, center, width, height, area, and some shape-specific parameters. The latent variables have a KL penalty, but the decoder does not reconstruct the input spectrum. The objective is supervised peak-parameter prediction with a variational bottleneck; it is not an input-reconstructing VAE in the original sense of [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114). A supervised information-bottleneck formulation is a closer description ([Deep Variational Information Bottleneck](https://openreview.net/pdf/8c41248f88f7436402feb8fd572711569713d314)).

The loss implementation has no obvious indexing, mask, or reduction error: absent peaks are excluded from shape and numeric losses; `BCEWithLogitsLoss` uses slot-specific positive weights; `KLDivLoss` uses `batchmean`, the reduction recommended for a mathematical KL; and validation uses the latent mean. [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) and [KLDivLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html) document these choices. This is a code review finding, not proof that the loss is optimal.

## Evidence from the saved run

| Task on the held-out synthetic test set | Result |
| --- | ---: |
| Presence precision / recall / F1 | 95.7% / 93.4% / 94.5% |
| Correct shape among detected true peaks | 71.4% |
| Correct shape among **all** true peaks | 66.6% |
| Center within 0.2 eV among all true peaks | 70.6% |
| FWHM within 0.2 eV among all true peaks | 83.3% |
| Height within 20% among all true peaks | 36.7% |
| Area within 20% among all true peaks | 35.2% |

The last four rates count missed peaks as failures. Conditional on detection, center and FWHM mean absolute errors are 0.139 and 0.099 eV. Height and area median absolute percentage errors are 26.2% and 27.4%. See `outputs/c1s_nexus_peak_vae/test_task_metrics.csv`.

At the best recorded validation epoch (60 of 80), the total loss was 0.506. Presence contributed 0.181 and shape contributed 0.267 after its 0.5 weight: together about 89% of the **scalar** total. Height and area contributed 0.00110 and 0.00101, about 0.4% together. KL contributed 0.0327 after its 0.001 weight, about 6.5%. These values are not gradient norms, so they motivate an ablation rather than proving that the optimizer ignores intensity. With normalized targets mostly in `[0, 1]`, the default `SmoothL1Loss(beta=1)` gives the numeric tasks a quadratic penalty for most errors ([PyTorch SmoothL1Loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)). None of the task weights or the KL coefficient has been validated as an optimum.

## Difficult cases

| True-peak group | Number of true peaks | Detected | Detected with correct shape | Height within 20%, including misses |
| --- | ---: | ---: | ---: | ---: |
| Area fraction below 5% | 2,190 | 63.7% | 46.3% | 18.4% |
| Area fraction at least 40% | 4,945 | 99.9% | 76.5% | 43.3% |
| Six peaks in a scan | 510 | 81.4% | 40.4% | 25.9% |
| A neighbor within 0.5 eV | 3,200 | 87.8% | 65.1% | 30.4% |

The full breakdown is in `outputs/c1s_nexus_peak_vae/test_hard_case_factors.csv` and `test_morphology_breakdown.csv`. Profile recovery among all true peaks is 82.6% for Gaussian, 60.4% for Voigt, and 53.9% for pseudo-Voigt. The near-perfect Doniach score is partly explained by the generator reserving that profile for one distinctive component; it should not be taken as evidence of universal Doniach recognition.

Intensity is also less accurate in the high-count tier: median height error on detected peaks rises from 21.7% in the low-count tier to 35.1% in the high-count tier; the corresponding area errors rise from 24.4% to 34.2%. This is an observed association, not an identified cause. The six-peak result comes from 17 held-out cases, so it needs case-level uncertainty estimates.

## Specific issues and limits

1. **Area-fraction teacher forcing.** During training, the area-fraction KL normalizes predicted areas over **true-present** slots. Inference normalizes over **predicted-present** slots. When a weak component is missed or a false component appears, the remaining fractions change in a way that the fraction loss did not directly train. Test a predicted-presence-aware or soft-presence consistency loss against the current objective; do not assume it will improve results.
2. **Independent geometry heads.** Height, area, width, and profile parameters are predicted separately even though a profile imposes relations between them. The model can output combinations that are mutually inconsistent. A profile-rendering consistency term or fewer redundant targets might help, but would need a validated forward model and a controlled comparison.
3. **The evaluated synthetic profile mix is narrow.** The format-3/4 dataset used for this checkpoint selects one ordinary profile family for all ordinary peaks in a case. It therefore does not test mixed ordinary shapes within one spectrum. The new compact format-5 generator allows mixed ordinary profiles, but no model has been trained or evaluated on it. The six chemical output slots also limit the current predictor to at most six known components. The lmfit adapter can accept an arbitrary number of peaks, but the current predictor cannot supply them. This limits claims of universality.
4. **Checkpoint selection does not reflect all goals.** The best validation epoch differs by task: total loss at epoch 60, presence at 78, shape at 54, and FWHM at 77. Selection by one weighted sum can favor one task over another. The final checkpoint should be chosen using a predefined set of chemical and fitting metrics, including weak-peak recall and intensity accuracy.
5. **Thresholds trade off metrics.** Validation-only threshold tuning raised overall held-out presence F1 from 94.5% to about 94.8%, but lowered six-peak recall from 81.4% to about 80.0%. A single F1 maximum is not the answer if missing weak components is costly.
6. **Fitting noise model needs separate validation.** The lmfit adapter currently divides residuals by the square root of observed counts. A Poisson likelihood may be preferable for low-count **native** spectra; a high-throughput XPS fitting study uses that approach ([Matsumura et al., 2024](https://www.tandfonline.com/doi/full/10.1080/27660400.2024.2373046)). The model's common-grid scans are interpolated, so their points are not independent raw Poisson observations. Compare fit objectives on the original native counts and evaluate residuals and parameter recovery; do not simply replace the current residual on the interpolated grid. See also [Practical guide for curve fitting in XPS](https://doi.org/10.1116/6.0000377).

## Experiments to run on a training machine

Keep the existing case-separated train/validation/test split and freeze the test set. Run the same seeds and report the joint metrics above with confidence intervals over cases. Change one factor at a time:

1. Establish a deterministic encoder-decoder baseline with no latent sampling, and separately sweep the KL coefficient around the current `0.001`. Setting the KL coefficient to zero by itself still leaves the notebook's training-time latent sampling active. The current KL term is a regularizer, not evidence that stochastic latent variables help parameter prediction.
2. Tune numeric task weighting and Smooth L1 transition scale against height, area, center, and FWHM metrics. Learned task uncertainty weighting ([Kendall et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)) or gradient balancing ([GradNorm](https://proceedings.mlr.press/v80/chen18a.html)) are candidate methods, not guaranteed improvements.
3. Increase representation of low-area-fraction, closely spaced, and five- or six-peak cases in **training only**. Compare a visibility-weighted presence objective with the existing positive class weighting. A focal loss is another candidate for easy-versus-hard imbalance ([Lin et al., 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)); report false positives as well as weak-peak recall.
4. Compare independent parameter heads with physically consistent profile rendering and an area-fraction loss that uses predicted or soft presence. Test whether gains survive lmfit refinement, not only label-level scoring.
5. Expand the generator to include mixed ordinary profiles, shape mixtures, varying backgrounds and instruments, and then evaluate on measured spectra held out by sample and instrument. A model trained only on the present generator cannot establish generality to arbitrary C 1s spectra.

No replacement loss can be called “most optimal” from the existing saved run alone. The current objective is a reasonable baseline for its synthetic labels, while the intensity and weak-peak results give clear targets for controlled experiments elsewhere.
