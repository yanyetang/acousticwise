# Results

Measured results for AcousticWise v1.0. Every number is per 5 s clip, on the official UniqueShip folds, where the test vessels are never seen in training (technical design, section 8).

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
