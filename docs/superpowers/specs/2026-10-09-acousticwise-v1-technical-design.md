# AcousticWise v1.0 — Technical Design

- **Status:** Draft for review
- **Date:** 2026-10-09
- **Implements:** `2026-10-08-acousticwise-v1-prd.md` (requirement IDs below refer to it)
- **Scope:** structure, interfaces and data flow. Values marked *provisional* are fixed after the D3 dataset report.

## 1. Stack and layout

Python package managed with `uv`; PyTorch for the CNN; scikit-learn for the baseline; librosa/soundfile for audio; Streamlit for the UI; pytest for tests.

```
acousticwise/
  pyproject.toml
  DATA.md                  # provenance, license (D1)
  README.md                # E4
  configs/default.yaml     # all hyperparameters and audio settings
  src/acousticwise/
    data/
      dataset.py           # Recording, DatasetLoader protocol
      shipsear.py          # ShipsEar loader (only dataset-specific code)
      splits.py            # recording-level split, persisted
      report.py            # dataset report (D3)
    audio/
      io.py                # load WAV, validate (A3, A4)
      preprocess.py        # resample, mono, normalize, window (D2)
      features.py          # MFCC (M1), log-mel (M2)
    models/
      baseline.py          # MFCC + classical classifier
      cnn.py               # compact CNN
    train.py               # CLI: train baseline or CNN
    evaluate.py            # metrics, confusion matrix, aggregation (M4)
    inference.py           # Predictor: the only API the app uses
    errors.py              # typed input/inference errors
  app/streamlit_app.py     # UI only; no ML logic
  scripts/download_data.py # fallback if redistribution is not allowed
  tests/
  artifacts/               # saved models (see section 5)
```

**Boundary rule (E1):** `app/` imports only `inference.py`; `train.py` and `evaluate.py` import `data/`, `audio/`, `models/`. Nothing imports `app/`. Preprocessing lives in `audio/preprocess.py` and is the single code path for training and inference, so train/serve skew cannot occur.

## 2. Data interface (the extensibility seam for B)

```python
@dataclass(frozen=True)
class Recording:
    id: str            # unique source-recording id
    path: Path
    label: str         # class name
    group: str         # leakage-safe grouping key, the unit of splitting
                       # (ShipsEar: vessel name, plus date if needed)
    meta: dict         # dataset-specific (site, date, raw vessel type)

class DatasetLoader(Protocol):
    name: str
    def recordings(self) -> list[Recording]: ...
```

`shipsear.py` is the only module that knows ShipsEar's file naming (`<id>__<DD_MM_YY>_<name>.wav`), its metadata table, and the mapping from the 11 raw vessel types to the five classes A–E. It reads only the original whole-event recordings; the pre-cut 5-second clip versions are never supported, since they carry no recording identifiers. Data must be obtained from the maintainers (D1), so `scripts/download_data.py` prints access instructions and verifies a user-supplied directory against the expected file list instead of downloading automatically. A second dataset (P2) is a new loader returning `Recording`s plus a label-mapping config; nothing downstream changes.

## 3. Data flow

```
Recording ─▶ load_wav ─▶ resample/mono/normalize ─▶ window ─▶ features ─▶ model ─▶ window scores
                                                                                    │
                                    per-recording aggregation (mean of softmax) ◀────┘
```

**Preprocessing (D2), provisional defaults in `configs/default.yaml`:**
- Sample rate 16 kHz, mono, peak or RMS normalization.
- Window 5 s, hop 2.5 s (training); windows shorter than the minimum are dropped, not padded.
- MFCC: 20 coefficients plus mean/std summary per window.
- Log-mel: 64 bands, `n_fft=1024`, hop 256, log-compressed, per-window standardization.

Windows are cut **after** the split is made, so windows from one recording can only ever fall into one split.

## 4. Splitting (M3)

- Unit of splitting is `Recording.group` (for ShipsEar, the vessel name), not the file, because one ship can appear in several recordings. If the metadata shows repeated vessels across dates and that leaves too few groups, date is added to the key; this is decided from the D3 report.
- Classes are the five ShipsEar classes A–E. Class D has about 9 recordings, so a single split is unstable.
- Method: scikit-learn `StratifiedGroupKFold`, repeated over several seeds, with groups held fixed per fold. Each repeat yields train/val/test partitions; headline numbers are the mean and spread across repeats. One additional fixed split (seed in config, roughly 70/15/15) is saved as the canonical held-out test set used for the single-model artifact reported in the app and README.
- If a class has too few groups to put one in every partition, the split code fails loudly with the class and counts, and the D3 report says what was done (for example merging or excluding), rather than silently producing an empty class.
- Splits are computed once and saved to `artifacts/splits.json` (record ids and groups per split, seed, method, repeat index). Training and evaluation load it; they never re-split. The manifest contains ids only, no audio, so it can be committed (D1).
- Tests: (1) the train/val/test id sets are disjoint and cover all recordings; (2) the group sets are disjoint, so zero shared vessel names; (3) no window's `recording_id` appears in more than one split. All three run on a synthetic fixture dataset with repeated vessels, so they need no real audio.

## 5. Models and artifacts (M1, M2, M6)

- **Baseline:** MFCC summary features into a scikit-learn pipeline (scaler + classifier; SVM or random forest, chosen on validation).
- **CNN:** a compact stack of 3–4 conv blocks (conv, batch norm, ReLU, pool), global average pooling, linear head. Class-weighted cross-entropy for imbalance (P1). Model selection on validation macro-F1.
- **Seeds:** one config seed controls Python, NumPy and PyTorch RNGs; deterministic algorithms are enabled where practical. The reproducibility gate uses a ±1–2 point macro-F1 tolerance.

Each trained model is saved as a directory:

```
artifacts/<run_id>/
  model.pt | model.joblib
  config.json        # full resolved config, seed, library versions
  labels.json        # ordered class names
  splits.json        # copy of the split used
  metrics.json       # val + test results
```

`Predictor.load(run_dir)` reads `config.json` so inference reproduces the exact preprocessing used in training.

## 6. Inference and validation (A1–A4)

```python
class Predictor:
    @classmethod
    def load(cls, run_dir: Path) -> "Predictor": ...
    def predict(self, wav_path_or_bytes) -> Prediction: ...

@dataclass
class Prediction:
    label: str
    scores: dict[str, float]     # "model confidence", not calibrated
    waveform: np.ndarray         # for display
    sr: int
    mel: np.ndarray              # for display
```

`predict` windows the file, scores each window, and returns the mean softmax across windows.

**Validation (`audio/io.py`)** raises typed errors from `errors.py`; the UI maps each to a friendly message and never shows a traceback:

| Condition | Error |
|---|---|
| Not a readable PCM WAV, or corrupt | `UnsupportedFileError` |
| File larger than limit | `FileTooLargeError` |
| Shorter than one window (provisional minimum) | `AudioTooShortError` |
| Longer than the duration limit | `AudioTooLongError` |
| Near-silent (RMS below a threshold) | `SilentAudioError` |
| Anything unexpected in model code | wrapped as `InferenceError` |

Limits are *provisional* and set from D3.

## 7. Application

Single Streamlit page: file uploader, "Classify" button, result card (predicted class, bar chart of per-class model confidence with the "not calibrated" label), waveform and spectrogram plot, and an error banner. The model run directory is set in config. One-command launch: `uv run acousticwise-app`, which wraps `streamlit run`.

## 8. Evaluation (M4, M5)

`evaluate.py` computes, from a run directory and the saved split:
- accuracy, macro-F1, per-class precision/recall/F1, confusion matrix, on the test set;
- both per-window and per-recording (aggregated) results;
- cross-validation mean and spread of macro-F1 and per-class recall across repeats, for both models;
- a baseline-vs-CNN comparison table written to `metrics.json` and rendered in the README.

Misclassified recordings are listed with their ids so the P1 error analysis can pull spectrograms.

## 9. Testing (E3)

| Area | Tests |
|---|---|
| Preprocessing | Output rate, mono, shape and range on synthetic tones; resampling from several input rates; windowing counts |
| Split | Disjoint ids; window-level no-leakage check; split is stable for a fixed seed |
| Features | Expected shapes for MFCC and log-mel |
| Inference | Loads a tiny fixture model; predicts on a synthetic WAV; scores sum to ~1 |
| Invalid input | One test per row in the section 6 error table |
| Reproducibility | Training for a few steps with a fixed seed twice gives the same loss (smoke test) |

Tests use synthetic audio, so the suite runs without the dataset.

## 10. Provisional values to settle after D3

Sample rate and window/hop length (the source audio is 52,734 Hz, 24-bit; the 16 kHz default must be checked against the content of the vessel signals); file size and duration limits (A4); silence threshold; the exact grouping key (vessel name alone vs vessel plus date); the number of CV repeats; what to do with any class that has too few groups.

## 11. Second dataset (P2) seam

A second dataset needs its own `DatasetLoader`, a coarse label mapping to ShipsEar's classes defined in advance in config, and its own `group` key. Evaluation reports it as a separate section. Candidates and constraints are in `reports/Underwater vessel dataset sources.md`; nothing is built for it in v1.0 P0/P1.
