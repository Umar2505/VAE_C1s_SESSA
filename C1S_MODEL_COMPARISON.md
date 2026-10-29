# Run and compare the three C 1s peak models

The three format-5 notebooks use exactly the same dataset and held-out cases:

Each notebook is self-contained: its architecture, data loading, loss, training, and evaluation code are visible in code cells. The generator and its completed NeXus dataset remain separate inputs. The shared `c1s_peak_pipeline.py` file is retained as a reference implementation and is not imported by the notebooks.

| Notebook | Predictor | Saved checkpoint directory |
| --- | --- | --- |
| [`c1s_peak_cnn.ipynb`](c1s_peak_cnn.ipynb) | Deterministic 1D residual CNN, six named peak slots | `outputs/c1s_peak_models/supervised_parameters_v1/cnn/` |
| [`c1s_peak_vae_v5.ipynb`](c1s_peak_vae_v5.ipynb) | Convolutional variational parameter predictor, six named peak slots | `outputs/c1s_peak_models/supervised_parameters_v1/vae/` |
| [`c1s_peak_transformer.ipynb`](c1s_peak_transformer.ipynb) | Transformer with eight unordered peak queries | `outputs/c1s_peak_models/supervised_parameters_v1/transformer/` |

**No model training was run while preparing these notebooks.** Run their training cells on the intended training machine.

## Run order

1. Set up the packages from [`environment.yml`](environment.yml) and open Jupyter in the project directory.
2. Start `python c1s_lmfit_dataset.py --cases 10000 --replicates 5 --scans-per-file 100` in another process. Or point each notebook's `dataset_dir` to an existing compact format-5 dataset.
3. Run all cells in each of the three notebooks. They may be opened before generation finishes; they wait for `manifest.json`, which the generator writes after every shard is complete. Each notebook first verifies on two validation batches that the evaluation renderer agrees with the stored clean spectra.
4. Review `outputs/c1s_peak_models/supervised_parameters_v1/model_comparison.csv`, `model_comparison.png`, and `case_comparison.csv` once all three notebooks have completed. The comparison refuses to combine reports with different dataset manifest hashes or training objectives. `case_comparison.csv` aligns the same held-out cases across models.

Each notebook saves `latest.pt` after every epoch and selects `best.pt` using the validation split. It can resume an interrupted run using the same supervised objective. Earlier checkpoints trained with a rendered-spectrum loss belong to a different experiment and remain in the old directories; do not resume them into this run. A threshold for declaring a peak present is chosen on validation cases only. The test split is used once for final reporting.

## If training is slow

The default generator makes 50,000 spectra (`10,000` cases × `5` scans), with about 40,000 in training. With batch size `64`, an epoch has about 625 training batches plus a full validation pass. Training and validation compare predicted peak and background values directly with stored labels; they do not render a spectrum. The Transformer still matches its unordered queries to the true peaks.

Check the notebook's first output for `Device: cuda` versus `Device: cpu`. On a CUDA machine, check GPU use and memory with `nvidia-smi`. Run the three notebooks one after another if they share one GPU. The epoch message reports separate training, validation, and checkpoint times; `training_history.csv` stores training and validation seconds. A slow checkpoint points to disk speed. Low GPU use with a long training pass suggests data loading, CPU matching, or synchronization overhead.

The revised objective is a new experiment. Restart the Python kernel with the updated files and start training in the new `supervised_parameters_v1` output directory. Earlier renderer-loss checkpoints cannot establish this objective's validation score. These changes have not been benchmarked on the training machine, so use the new timings to measure their effect rather than assuming a particular speedup.

## What the reports mean

| File | What to inspect |
| --- | --- |
| `training_history.csv`, `learning_curves.png` | Training versus validation loss, with separate detection, shape, center, width, and area terms. |
| `validation_thresholds.csv` | Peak-presence precision/recall tradeoff on validation cases. |
| `test_summary.json`, `task_overview.png` | Test detection, shape, center, FWHM, height, area, background, and clean-spectrum scores. |
| `test_true_peaks.csv` | Every true peak, including missed peaks; absolute and relative errors, shape-specific parameter errors, weakness, overlap, and source NeXus row. |
| `test_predicted_peaks.csv` | Every predicted peak, including extra peaks and full predicted parameters. |
| `test_scans.csv` | Whole-spectrum and background error for every scan. |
| `test_breakdown.csv`, `failure_subgroups.png` | Success rates by family, background, number of peaks, signal level, chemical component, shape, area fraction, and nearest-peak separation. |
| `shape_confusion.csv`, `shape_confusion.png` | Confused profile types and missed peaks. |
| `test_cases_ranked.csv`, `best_worst_cases.csv`, `best_worst_cases.png` | Three easiest and three hardest independent synthetic cases, with representative scans and component curves. |
| `test_latent_sampling.csv`, `test_latent_sampling.png` | VAE only: how sampling spread relates to actual center and area errors. This is not calibrated uncertainty. |

The **all true peaks** success rates count a missed peak as wrong. Median numeric errors are calculated only where the peak was detected; therefore read them together with detection recall. The `difficulty_score` is a review ranking combining missed/extra peaks, shape errors, center failures, and clean-spectrum error. It is not a probability or scientific confidence measure.

Peak and background parameters are supervised from the stored labels. `background_counts` and `expected_counts` are reserved for the renderer audit and held-out evaluation; they do not enter the training loss. Peak height and final FWHM are calculated from rendered curves during evaluation and are not separately predicted by the network. The CNN and VAE use six fixed chemical slots. The Transformer can leave queries empty, but it still learns only the six component classes present in this dataset.

## Scientific interpretation

These scores are for held-out cases from the *same synthetic generator*. Repeated scans of one synthetic case remain in one split. A low spectrum residual does not prove the right chemical decomposition, especially for overlapping peaks. Before using the best model for measured C 1s spectra, evaluate measured spectra from different samples and instruments and check for unfamiliar background or peak shapes.
