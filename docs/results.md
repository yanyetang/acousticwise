# Results

Measured results for AcousticWise v1.0. Every number is per 5 s clip, on the official UniqueShip folds, where the test vessels are never seen in training (technical design, section 8).

**Summary:** on test, the CNN reaches 0.322 ± 0.044 headline macro-F1 (9 classes) against 0.259 ± 0.045 for the MFCC + random forest baseline, higher on all five folds. The CRNN (0.315 ± 0.055) does not improve on the CNN. Random guessing would score about 0.11.

## Baseline: MFCC + random forest (R1, 2026-10-24)

Features: 20 MFCC means and 20 standard deviations from each clip's log-mel, with RMS normalization. Classifier: scikit-learn `RandomForestClassifier` with default settings and seed 42, after a `StandardScaler`. Trained locally from the feature cache, about 15 s per fold.

### Choosing the settings on validation

Six settings were trained on all five folds and compared on **validation** headline macro-F1 only. The test split was not used for this choice.

| Setting | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± std |
|---|---|---|---|---|---|---|
| **RMS + random forest** | 0.264 | 0.285 | 0.299 | 0.321 | 0.267 | **0.287 ± 0.024** |
| None + random forest | 0.271 | 0.281 | 0.310 | 0.316 | 0.253 | 0.286 ± 0.026 |
| Peak + random forest | 0.248 | 0.279 | 0.287 | 0.317 | 0.260 | 0.278 ± 0.027 |
| None + SVM | 0.244 | 0.273 | 0.287 | 0.283 | 0.244 | 0.266 ± 0.021 |
| RMS + SVM | 0.249 | 0.275 | 0.282 | 0.282 | 0.241 | 0.266 ± 0.019 |
| Peak + SVM | 0.240 | 0.266 | 0.276 | 0.281 | 0.240 | 0.261 ± 0.020 |

- **Random forest over SVM:** better by about 0.02 for every normalization, on almost every fold, and about 6 times faster to train.
- **RMS over no normalization:** a tie on validation (0.001 apart, far inside the spread). RMS was chosen because it removes the recording level as a clue. The dataset report found one hydrophone 18 dB louder in 2016–18, and classes are spread unevenly over those years, so without normalization the model can partly learn the recording instead of the vessel. RMS also makes the app's predictions independent of an upload's volume.

### Test results

The chosen setting, scored once on each fold's test split. Std is the sample standard deviation over the five folds.

| Fold | Test clips | Accuracy | Headline macro-F1 (9 classes) | All-class macro-F1 | Classes without test clips |
|---|---|---|---|---|---|
| 1 | 4,477 | 0.261 | 0.264 | 0.218 | dredger |
| 2 | 5,659 | 0.116 | 0.189 | 0.142 | none |
| 3 | 3,636 | 0.256 | 0.268 | 0.221 | pilotvessel |
| 4 | 4,280 | 0.214 | 0.260 | 0.221 | pilotvessel |
| 5 | 3,605 | 0.310 | 0.315 | 0.288 | dredger, pilotvessel |
| **Mean ± std** | | **0.231 ± 0.073** | **0.259 ± 0.045** | **0.218 ± 0.052** | |

Random guessing would score about 0.11 headline macro-F1.

**Fold 2 is low because of one boat.** 3,565 of its 5,659 test clips (63%) are pilotvessel, all from a single vessel, and the model trained on one other pilot boat. It gets none of them right. Accuracy counts every clip, so that one boat pulls fold 2's accuracy to 0.116. The headline score drops too (0.189), even though pilotvessel is not a headline class: the 3,565 clips are predicted as headline classes and lower their precision. Fold 4 has the same pattern on a smaller scale: 1,560 dredger test clips (36%) from one vessel. Without fold 2, the headline mean would be 0.277; it is kept, because the folds are the official ones.

**Validation was higher than test** (0.287 against 0.259). Part of that is choosing the best of six settings on validation, which favours whichever one happened to suit the validation vessels.

### Per class

F1 on test, averaged over the folds where the class has test clips. Vessels are distinct MMSI (background: distinct days). "Largest vessel" is the share of a class's test clips from its single biggest test vessel, per fold.

| Class | Mean F1 | Range | Train vessels | Test vessels per fold | Largest vessel per fold (%) |
|---|---|---|---|---|---|
| background | 0.58 | 0.39–0.69 | 641–642 days | 82 / 80 / 81 / 80 / 79 | 1 / 1 / 1 / 1 / 1 |
| tanker | 0.40 | 0.36–0.44 | 186–187 | 34 / 10 / 20 / 19 / 28 | 3 / 10 / 5 / 5 / 4 |
| cargo | 0.31 | 0.21–0.39 | 2,017–2,018 | 311 / 87 / 237 / 196 / 286 | 0 / 2 / 1 / 1 / 0 |
| tug | 0.27 | 0.15–0.37 | 237–238 | 39 / 12 / 28 / 26 / 30 | 3 / 9 / 4 / 4 / 4 |
| pleasurecraft | 0.24 | 0.10–0.29 | 324–325 | 50 / 13 / 39 / 21 / 48 | 2 / 8 / 3 / 5 / 2 |
| sailing | 0.18 | 0.09–0.25 | 169–170 | 26 / 7 / 22 / 20 / 27 | 4 / 14 / 5 / 5 / 4 |
| passengership | 0.15 | 0.06–0.26 | 88 | 13 / 5 / 10 / 9 / 10 | 8 / 20 / 11 / 11 / 10 |
| fishing | 0.14 | 0.08–0.24 | 190–191 | 31 / 9 / 20 / 14 / 22 | 3 / 12 / 5 / 7 / 5 |
| other | 0.06 | 0.00–0.12 | 135–136 | 20 / 2 / 18 / 8 / 20 | 5 / 50 / 6 / 13 / 5 |
| rescue | 0.02 | 0.00–0.05 | 18–19 | 3 / 3 / 3 / 2 / 1 | 56 / 44 / 48 / 83 / 100 |
| dredger | 0.02 | 0.00–0.05 (3 folds) | 4–5 | 0 / 1 / 1 / 1 / 0 | – / 100 / 100 / 100 / – |
| pilotvessel | 0.00 | 0.00–0.01 (2 folds) | 1–2 | 1 / 1 / 0 / 0 / 0 | 100 / 100 / – / – / – |

- **Background, tanker and cargo** score best, and each of their test vessels supplies only a few percent of the clips, so these scores rest on many ships.
- **Rescue, dredger and pilotvessel** are near zero. They have 1 to 19 training vessels, and in most folds one test vessel supplies half or all of their test clips, so their scores say more about single ships than about the vessel type. This is why they are outside the headline.
- **Other** is a mixed catch-all class and stays near zero.

### Reproducing

```
uv run python scripts/build_manifest.py          # artifacts/splits.json; check its SHA-256 in DATA.md
uv run python -m acousticwise.train --fold N     # for N = 1 … 5, with configs/default.yaml
uv run python -m acousticwise.evaluate artifacts/runs/baseline_fold*_random_forest_*
```

The runs above used `configs/default.yaml` as committed with this file (with `data.runs_dir` pointed elsewhere; each run's `config.json` holds the exact config), the split manifest with SHA-256 `84591f54…916a18`, scikit-learn's random forest with seed 42, and the 32 kHz feature cache.

## Model comparison: baseline, CNN and CRNN (S15a, 2026-10-25)

The CNN and the CRNN (technical design, section 5) read the same RMS-normalized log-mel cache as the baseline and were trained on all five folds with the committed config: 25 epochs, batches of 64, AdamW with learning rate 0.001 and weight decay 0.0001, seed 42. Each run kept the epoch with the best validation headline macro-F1; nothing else was tuned. They trained on a Kaggle T4 GPU, about 13–14 minutes per run, and were scored once on test on the laptop.

### Test results

Mean ± std over the five folds; the last column is the headline score of folds 1 to 5.

| Model | Macro-F1, 9 classes | Macro-F1, 12 classes | Accuracy | 9-class per fold |
|---|---|---|---|---|
| Random forest | 0.259 ± 0.045 | 0.218 ± 0.052 | 0.231 ± 0.073 | 0.264, 0.189, 0.268, 0.260, 0.315 |
| CNN | 0.322 ± 0.044 | 0.272 ± 0.053 | 0.275 ± 0.092 | 0.344, 0.260, 0.332, 0.301, 0.375 |
| CRNN | 0.315 ± 0.055 | 0.270 ± 0.070 | 0.273 ± 0.095 | 0.320, 0.236, 0.337, 0.298, 0.386 |

- **The CNN beats the baseline.** It is higher on every fold, by 0.041 to 0.080 headline macro-F1 (mean +0.063), and the mean gain is larger than the spread between folds. The folds differ in difficulty (fold 2 is lowest for all three models, see the baseline section), so the per-fold comparison is the stronger evidence: on the same test vessels, the CNN wins five times out of five.
- **The CRNN does not beat the CNN.** It is lower on three folds and higher on two, by −0.024 to +0.011 (mean −0.007), far inside the spread. The bidirectional LSTM adds about 100,000 parameters and does not help on 5 s clips at this data size, so the CNN is the model to ship.
- **The 12-class score moves with the 9-class score** (+0.054 for the CNN). The three classes outside the headline stay near zero for every model, so the gain comes from the nine classes with many vessels.
- **Validation was higher than test** for both networks (CNN 0.365 against 0.322, CRNN 0.363 against 0.315), as for the baseline: the kept epoch is the one that suited the validation vessels best.
- **Most runs kept a late epoch** (21 to 25 of 25 for seven of the ten runs), so longer training might still help. Trying that is left for later and would be chosen on validation.

### Per class

Test F1 averaged over the folds where the class has test clips.

| Class | Random forest | CNN | CRNN |
|---|---|---|---|
| background | 0.58 | 0.63 | 0.64 |
| tanker | 0.40 | 0.52 | 0.48 |
| cargo | 0.31 | 0.38 | 0.37 |
| tug | 0.27 | 0.31 | 0.30 |
| pleasurecraft | 0.24 | 0.27 | 0.27 |
| passengership | 0.15 | 0.26 | 0.24 |
| sailing | 0.18 | 0.21 | 0.22 |
| fishing | 0.14 | 0.20 | 0.20 |
| other | 0.06 | 0.11 | 0.11 |
| rescue | 0.02 | 0.05 | 0.08 |
| dredger | 0.02 | 0.02 | 0.02 |
| pilotvessel | 0.00 | 0.00 | 0.00 |

The CNN improves every class with enough vessels, most for tanker (+0.12) and passengership (+0.11). The MFCC baseline keeps only the average shape of the spectrum over the clip; the CNN sees the full log-mel, including narrow tones and how they change over the 5 s. Dredger and pilotvessel stay at zero for every model: with 1 to 5 training vessels there is too little to learn the type rather than the ship.

### Reproducing

The ten runs were trained on Kaggle by `notebooks/cnn_train/` (code uploaded by `notebooks/spectrogram_grid/upload_src.sh`, which records the commit), which checks the manifest's SHA-256 and writes `checksums.txt` for the run files. Locally:

```
uv run python -m acousticwise.train --fold N --model cnn                      # CNN, N = 1 … 5
uv run python -m acousticwise.train --fold N --model cnn --time-pooling lstm  # CRNN
uv run python -m acousticwise.evaluate RUN_DIR... --cache-dir data/cache --out summary_cnn.json
uv run python -m acousticwise.compare "Random forest=summary_random_forest.json" \
    CNN=summary_cnn.json CRNN=summary_crnn.json
```

`--cache-dir` is needed only for runs trained on another machine, whose `config.json` records that machine's cache path. Training on a GPU is not bit-for-bit repeatable, so a retrain is expected to match within the ±2 point tolerance of the reproducibility gate, not exactly.
