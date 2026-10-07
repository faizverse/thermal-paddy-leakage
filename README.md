# Leakage-aware evaluation of transfer learning on the thermal paddy leaf dataset

Code and per-fold results for: *Leakage-aware transfer learning evaluation for thermal paddy leaf classification*
(M. F. Bukhori, S. A. Syed Mustafa, N. Zainal and M. H. Mohd Zaman; submitted to Jurnal Kejuruteraan).

## Data
Public dataset: Aarthy & Sujatha (2020), "Thermal images – diseased & healthy leaves – paddy", Kaggle,
https://www.kaggle.com/datasets/sujaradha/thermal-images-diseased-healthy-leaves-paddy (636 FLIR E8 images, six classes).
Put the six class folders (`BLB`, `blast`, `healthy`, `hispa`, `leaf folder`, `leaf spot`) in `data/`, or set the `PADDY_DATA` environment variable.

## Pretrained weights
Put the Keras ImageNet "notop" weight files in `weights/`:
`resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5`, `mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5`,
`inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5`, `efficientnetb7_notop.h5`.
(If the ResNet50 file is absent, Keras downloads it.) The authors' runs used the ResNet50 backbone of the original project model (`weights/paddy_224.json`, `weights/paddy_224.weights.h5`; not included), whose backbone was frozen during training; `src/common.py` uses it if present and otherwise the Keras ImageNet ResNet50.

## Pipeline (CPU is sufficient; run from the repository root)
The scripts in step 2 write their outputs (`rows_*.json`, `oof_*.npy`, `groups_*.npy`, `perclass.json`) to the repository root. `results/` holds the outputs reported in the paper; to use your own re-run instead, copy the new `rows_*.json` into `results/` before steps 3 and 5.
1. `python src/extract.py` – overlay crop (bottom 15%, right 20%), resizing and frozen-backbone feature caching.
2. `python src/cv.py resnet50_crop:grouped mobilenetv2_crop:grouped inceptionv3_crop:grouped efficientnetb7_crop:grouped` – leaf-grouped stratified 5-fold CV × 3 seeds.
   Ablations: `resnet50_crop:imagelevel resnet50_nocrop:grouped resnet50_nocrop:imagelevel resnet50_crop:grouped_testval resnet50_crop:imagelevel_testval resnet50_crop:grouped_cw`.
   Test-fold early stopping with an identical training set: `python src/testval_same.py`.
   Grouping sensitivity (coarser groupings + control): `python src/grouping_sensitivity.py` (also saves the group labels `groups_*.npy`); mechanism, group-weighted accuracy, fold composition and readout counts: `python src/mechanism_check.py`.
   Temperature-based inputs (raw radiometric data in the JPEGs) and temperature-level confound probe: `python src/thermal.py`.
   Random-group control: `python src/randgroup.py` – group labels permuted within each class (same number and sizes of groups), leaf-grouped protocol repeated.
3. `python src/stats.py` – per-class report (and descriptive Wilcoxon tests); `python src/stats_corrected.py` – corrected resampled t-tests (test-to-training ratio 0.32, recomputed from the folds when `labels.npy` and `paths.npy` are present) with Holm correction and significance level 0.05, as reported in the paper; writes `results/corrected_tests.json`.
   `python src/save_summaries.py` – saves the other numbers reported in the paper that are computed from the out-of-fold predictions and group files (`results/summary_values.json`, `results/coarse_groupings.csv`).
4. Supplementary (not reported in the paper): `python src/compress.py resnet50 5` and `python src/compress.py mobilenetv2 5` – TensorFlow Lite variants: size, accuracy, latency, largest activation tensor.
5. Figures (written to `figs/`): `python figures/fig1_pipeline.py`, `python figures/fig2_dataset.py`, `python figures/fig3_inflation.py` (Figures 1–3); `python src/figs.py` (Figure 4, confusion matrix and ROC curves; Figure 5, Grad-CAM; and `perclass.json`); `figures/fig6_screenshot.py` (Figure 6, screenshot of the web prototype; it needs the web-app files of the Hugging Face Space, see its header). `python src/deploy.py` – classification head for the web prototype.

## Results included
- `results/rows_*.json` – per-fold metrics for every configuration (Tables 3 and 4).
- `results/perclass.json` – per-class metrics (Table 5); `results/corrected_tests.json` – corrected t-tests; `results/summary_values.json` – multi- vs single-image and group-weighted accuracy, readout images and their accuracy, per-class F1 with temperature input, confusion matrices, group counts of the coarser groupings, files without a numeric identifier, and the original project's single-split values (with their source); `results/coarse_groupings.csv` – group label of every image under each grouping; `results/compress_*.json` – supplementary TensorFlow Lite results (not reported in the paper).
- `results/fold_composition_leaf_grouped.csv` – images and groups per class in the training, validation and test parts of every fold and seed.
- `results/fold_assignments_leaf_grouped.csv` – leaf group and test fold of every image for seeds 0–2.
- `web_parity_check/` – web-prototype parity check on 120 images (`parity120.json`, `raw120.json`, `test120.js`) and browser timings of the live prototype (`web_timing_M3.json`, measured with `web_timing.js`).
- `models/paddy_head_weights.npz` – dense-head weights (W1, b1, W2, b2) of the deployed web prototype (on top of the ImageNet ResNet50 backbone).

Verified: re-running step 2 for `resnet50_crop:grouped` reproduces 77.22 ± 6.80% accuracy.
Software: Python 3.11, TensorFlow 2.21.0, Keras 3.15.1, scikit-learn 1.8.0.
Web prototype (in-browser, TensorFlow.js): https://huggingface.co/spaces/drfaizb/thermal-paddy-classifier
License: MIT.
