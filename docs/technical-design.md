# AcousticWise v1.0 — Technical Design

- **Status:** Draft v2 for review (revised 2026-10-09 for UniqueShip as the primary dataset)
- **Date:** 2026-10-09
- **Implements:** [`prd.md`](prd.md) (requirement IDs below refer to it)
- **Scope:** structure, interfaces and data flow. Values marked *provisional* are fixed after the D3 dataset report ([`dataset-report.md`](dataset-report.md)).

## 1. Stack and layout

Python package managed with `uv`; PyTorch for the CNN; scikit-learn for the baseline; librosa/soundfile for audio; Streamlit for the UI; pytest for tests.

```
acousticwise/
  pyproject.toml
  DATA.md                  # provenance, licence, checksums (D1)
  README.md                # E4
  configs/default.yaml     # all hyperparameters and audio settings
  src/acousticwise/
    data/
      dataset.py           # Recording, DatasetLoader protocol
      uniqueship.py        # UniqueShip loader (reads the official fold CSVs)
      shipsear.py          # P2 only: ShipsEar loader
      splits.py            # load official folds / generic grouped split; leakage checks
      report.py            # dataset report (D3)
    audio/
      io.py                # load WAV, validate (A3, A4)
      preprocess.py        # resample, mono, normalize, window (D2)
      features.py          # MFCC (M1), log-mel (M2)
    models/
      baseline.py          # MFCC + classical classifier
      cnn.py               # compact CNN; CRNN variant via a time-pooling setting
    train.py               # CLI: train baseline or CNN, device auto-select
    evaluate.py            # metrics, confusion matrix, 12- and 9-class (M4)
    compare.py             # model comparison table across folds (M5)
    inference.py           # Predictor: the only API the app uses
    errors.py              # typed input/inference errors
  app/streamlit_app.py     # UI only; no ML logic
  app/samples/             # 36 sample clips (3 per class) and manifest.csv, for the sample browser
  scripts/download_data.py # prints access instructions, verifies a user-supplied directory
  scripts/cache_features.py# builds the feature cache (runs on Kaggle or locally)
  scripts/export_samples.py# picks the sample clips and writes their manifest (runs on Kaggle with the bundle)
  tests/
  artifacts/               # saved models and manifests (see section 5)
```

**Boundary rule (E1):** `app/` imports only `inference.py`; `train.py` and `evaluate.py` import `data/`, `audio/`, `models/`. Nothing imports `app/`. Preprocessing lives in `audio/preprocess.py` and is the single code path for training, feature caching and inference, so train/serve skew cannot occur.

## 2. Data interface (the extensibility seam for B)

```python
@dataclass(frozen=True)
class Recording:
    id: str            # unique id of the unit being classified
    path: Path
    label: str         # class name
    group: str         # leakage-safe grouping key, the unit of splitting
    meta: dict         # dataset-specific (date, hydrophone, source fold role)

class DatasetLoader(Protocol):
    name: str
    def recordings(self) -> list[Recording]: ...
```

**UniqueShip.** `uniqueship.py` is the only module that knows the bundle layout (`audio/<class>/<start>_<end>.wav`; `fold_k/metadata_5s_{train,validation,test}.csv`, path relative to the CSV). Here a `Recording` is one 5-second clip:
- `id` is the clip filename stem (UTC start and end).
- `group` is the MMSI as a string; background clips (MMSI 0) use `bg-<YYYYMMDD>`, because the dataset splits background by whole day.
- `meta` carries the optional AIS and CTD columns. They are often missing and are never needed by the model.

Several CSV columns have mixed types, so they are read with `low_memory=False` and only the core columns (`label`, `path`, `duration_sec`, `sample_rate`, `date`, `MMSI`, `hydrophone`) are relied on. The loader asserts on load that `sample_rate` is 32,000 and warns on clips under 5 s.

Data comes from the dataset's Google Drive links, so `scripts/download_data.py` prints the access instructions and verifies a user-supplied directory against the checksums in `DATA.md`, instead of downloading automatically.

**ShipsEar (P2).** `shipsear.py` is the only module that would know ShipsEar's file naming (`<id>__<DD_MM_YY>_<name>.wav`), its metadata and the mapping from its 11 vessel types to A–E. A recording there is a whole event, with `group` = vessel name. It reads original recordings only, never the pre-cut clips. A new dataset is a new loader plus a label-mapping config; nothing downstream changes.

## 3. Data flow

```
Recording ─▶ load_wav ─▶ resample/mono/normalize ─▶ window ─▶ features ─▶ model ─▶ scores
                                                                               │
                      (uploaded files longer than one window: mean of softmax) ◀┘
```

**Preprocessing (D2), provisional defaults in `configs/default.yaml`:**
- Source audio is 32 kHz, mono, 32-bit. Target rate is 32 kHz (native, Nyquist 16 kHz), set in config. RMS normalization was chosen in R1 (`docs/results.md`): it tied no normalization on validation and removes the recording level as a clue. Because a gain change is a constant offset in log-mel, it is applied when features are loaded, not baked into the cache. The dataset report found an 18 dB jump in recording level between periods (about −30.5 dBFS median RMS for one hydrophone in 2016–2018, about −48 dBFS otherwise), and classes are not spread evenly over the years, so raw loudness would let a model learn the period instead of the vessel. The spectrum check (2026-10-09, fold 1 train, up to 200 clips per class) found the 8-16 kHz band holds only 0.3 to 7.8% of power but is 13 to 18 dB above background for most ship classes, so 16 kHz would discard real signal. Once the baseline works, a 16 kHz versus 32 kHz comparison on validation data confirms the choice. Uploads at other rates are resampled to the target rate.
- Window 5 s. Training clips already are 5 s, so no windowing; training clips under 5 s (minimum 4.0 s) are zero-padded to 5 s and kept; the cache build prints how many were padded, and the choice is revisited if it is more than about 1% of clips. Uploads of 4.0 s or more are zero-padded to one window in the same way; shorter uploads are rejected. Longer uploads use window 5 s and hop 2.5 s.
- Log-mel: 64 bands up to the Nyquist, `n_fft=2048`, hop 512 at 32 kHz (the same 64 × 313 frames per window as `n_fft=1024`, hop 256 at 16 kHz, so the cache size does not change), in dB. Normalization and per-window standardization are applied on load.
- MFCC: 20 coefficients computed from the 64-band log-mel (DCT), plus mean/std summary per window. Derived on load, so baseline settings change without rebuilding the cache.

Windows are cut **after** the split is made, so a recording's windows can only fall into one split.

## 4. Splitting and leakage checks (M3)

- UniqueShip ships its own five folds (grouped by MMSI; background by day). `splits.py` reads those CSVs and converts them to the manifest below. It does **not** re-split. The dataset's grouping is the leakage guarantee, and the test re-verifies it.
- For a dataset without official folds (ShipsEar, P2), `splits.py` offers scikit-learn `StratifiedGroupKFold` on `Recording.group`, repeated over seeds, and fails loudly with class and counts if a class has too few groups for every partition.
- Manifest: `artifacts/splits.json` stores clip ids and groups per fold and split, the fold number, and the source (official CSV or computed, with seed). It holds ids only, no audio. At about 28 MB it is not committed: `scripts/build_manifest.py` rebuilds it from the official CSVs, checks it for leakage and prints its SHA-256, which `DATA.md` records (D1). Training and evaluation load it.
- **Leakage tests, run on every fold of the real manifest and on a synthetic fixture:** (1) the train/val/test id sets are disjoint; (2) the group sets are disjoint, so zero shared MMSI and zero shared background days; (3) no window's `recording_id` appears in more than one split. On 2026-10-09 the real bundle gave zero overlap for all of these.
- Headline results are the mean and spread across the five folds. The single-model artifact in the app uses one named fold (set in config).

## 5. Models, features and artifacts (M1, M2, M6)

- **Baseline:** MFCC summary features (each clip's log-mel shifted by its normalization offset, then 20 MFCC means and stds) into a scikit-learn pipeline: `StandardScaler`, then an RBF SVM wrapped in `CalibratedClassifierCV(ensemble=False)` for per-class probabilities, or a random forest. `baseline.classifier` in config picks one; R1 chose the random forest on validation (`docs/results.md`).
- **CNN:** `models/cnn.py`, the fold 1 spike's network: 4 conv blocks (3×3 conv, batch norm, ReLU, max pool; 32, 64, 128, 128 filters; the last two pool frequency only, leaving 78 time steps), the mean over the 4 remaining mel rows, then a time-combining step set by `cnn.time_pooling`: the mean over time (`mean`, the CNN, 242 k parameters) or a bidirectional LSTM with 64 units per direction whose two final states are joined (`lstm`, the CRNN, 341 k parameters). Then dropout 0.3 and a linear head.
  - **Input:** each clip's log-mel plus its normalization offset (as for the baseline), then standardized with the mean and standard deviation of the fold's training clips. Those two numbers are saved in the run's `config.json` (`input_scaling`), so evaluation and inference scale inputs the same way.
  - **Training:** cross-entropy, AdamW (`cnn.lr`, `cnn.weight_decay`) with a cosine learning-rate decay over `cnn.epochs`, batches of `cnn.batch_size`, shuffled each epoch by a generator seeded from the config seed. After every epoch the model is scored on validation with `clip_scores`; the weights from the epoch with the best validation headline macro-F1 are kept. Class-weighted cross-entropy is P1.
- **Device:** `train.py` picks cuda, then mps, then cpu, so the same code runs on Kaggle and on the author's Apple-silicon laptop.
- **Seeds:** one config seed controls Python, NumPy and PyTorch RNGs; deterministic algorithms are enabled where practical. The reproducibility gate has two tiers (PRD section 3): tests plus a fixture smoke run for anyone, and a cache-based retrain for the author within ±1 point macro-F1 (baseline) or ±2 (CNN).
- **Feature cache (E5):** `scripts/cache_features.py` reads every clip from fold 1's train, validation and test CSVs (together, every clip once) and runs `audio/cache.py`, which uses the same preprocessing and feature code as inference (each clip appears in all five folds, so the cache is not per fold). Training reads the cache, so the 26 GB audio is only needed once. The cache is generated and never committed. One cache per sample rate (`cache/sr32000/`, `cache/sr16000/`, both built in the same run; at 16 kHz `n_fft` and hop are halved so the shape stays 64 × 313), each holding:
  - `logmel.npy`: one array, clips × 64 × 313, float16, un-normalized dB (about 1.7 GB for about 43,200 clips), read with memory mapping;
  - `index.csv`: `clip_id`, `row` (the clip's row in `logmel.npy`), `label`, `padded`, `peak_db` and `rms_db` (the level of the 5 s window, so peak or RMS normalization is a constant subtracted on load). The vessel or day behind each clip is in the split manifest, not repeated here;
  - `meta.json`: the feature settings used (sample rate, window, minimum length, `n_fft`, hop, mel bands) and a `cache_format` number bumped by hand when feature code changes, the padded-clip count, and the clips that failed to load with their reasons. Library versions, build date and bundle checksums are deferred; the Kaggle build writes a `checksums.txt` of the output files.

  The **clip id** is the clip's path under `audio/` without `.wav` (for example `cargo/<start>_<end>`); the build fails if two clips share an id. A clip that cannot be read is skipped and listed in `meta.json`, and training leaves it out; the build fails if more than 0.1% of clips fail, since that points to a bug. Both checks run before anything is written. `load_cache` compares the settings in `meta.json` with the config field by field and refuses a mismatched cache with an error naming each field that differs and both values.

Each trained model is saved as a directory:

```
artifacts/runs/<run_id>/       # e.g. baseline_fold1_svm_20261024-142300
  model.pt | model.joblib   # CNN weights (state dict, saved on the cpu) | baseline pipeline
  config.json        # full resolved config, fold, library versions
  labels.json        # ordered class names
  splits.json        # the trained fold's split from the manifest (not committed)
  metrics.json       # "validation" from training (CNN: of the kept epoch, plus best_epoch and a per-epoch history); evaluate.py adds "test"
```

`uv run python -m acousticwise.train --fold N [--model baseline|cnn]` trains the baseline or the CNN (the CRNN is the CNN with `cnn.time_pooling: lstm`) and writes a run directory. `evaluate.py` loads either kind of model from the run directory. Training reads only the fold's train and validation clips, and scores validation the same way `evaluate.py` scores test (section 8), so the two are comparable. The test split is first touched by `evaluate.py`, after the settings are chosen. Both load features through `load_features` (`audio/cache.py`), so they cannot read the cache differently.

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
| Shorter than 4.0 s | `AudioTooShortError` |
| Longer than the duration limit | `AudioTooLongError` |
| Near-silent (RMS below a threshold) | `SilentAudioError` |
| Anything unexpected in model code | wrapped as `InferenceError` |

Limits, set from D3: at most 200 MB (Streamlit's default upload cap) and 300 s (60 windows); silence below RMS 1e-4 (−80 dBFS), 29 dB below the quietest clip in the dataset (−51 dBFS).

## 7. Application

Single Streamlit page: file uploader, "Classify" button, result card (predicted class, bar chart of per-class model confidence labelled "not calibrated"), waveform and spectrogram plot, a note that the model was trained on one hydrophone network, a few-vessels note when the top prediction is rescue, dredger or pilotvessel, and an error banner. The model run directory is set in config. One-command launch: `uv run acousticwise-app`, which wraps `streamlit run`.

## 8. Evaluation (M4, M5)

`uv run python -m acousticwise.evaluate RUN_DIR [RUN_DIR ...] [--out FILE] [--cache-dir DIR]` scores each run on its own fold's test split, using the run's `config.json`, `labels.json` and `splits.json`, and adds a `"test"` section to its `metrics.json`. Every preprocessing setting comes from the run's `config.json`; only the cache location can be overridden with `--cache-dir`, for a run trained on another machine (Kaggle), whose record holds that machine's path. The run's files are not changed by it.
- accuracy, the 9-class macro-F1 (headline) and the 12-class macro-F1. The 9-class subset is a config list (every class with at least 50 distinct training vessels in every fold, plus background: dredger, pilotvessel and rescue are out); its score is F1 averaged over those 9 labels on the full test set from the same 12-class model (scikit-learn `labels=`). Both averages use only the classes with clips in the split; the classes without any (for example dredger in fold 1's test) are listed in `classes_without_clips` instead of counting as 0;
- per class: precision, recall, F1 and clips, beside the number of distinct train and test vessels and the share of test clips from the largest test vessel (background grouped by day);
- the confusion matrix (rows true, columns predicted) with its class order;
- the misclassified clips with their ids and groups, so the P1 error analysis can pull spectrograms.

It then writes one summary (default `artifacts/summary.json`): a row per run with its fold and the three scalar scores, and, for two or more runs, their mean and sample standard deviation (`across_folds`). One run has no spread, so it gets no `across_folds`.

`uv run python -m acousticwise.compare NAME=SUMMARY [NAME=SUMMARY ...]` prints one markdown table from several models' summaries: per model, the mean ± std of the 9-class macro-F1, the 12-class macro-F1 and accuracy, then the 9-class macro-F1 of each fold in fold order, so models can be compared fold by fold. The table is in `docs/results.md`.

Added later: a vessel-vote table (mean softmax over a vessel's clips, P1), and the comparison table in the README.

## 9. Testing (E3)

| Area | Tests |
|---|---|
| Preprocessing | Output rate, mono, shape and range on synthetic tones; resampling from several input rates; windowing counts |
| Splits | Disjoint ids and groups on a synthetic fixture shaped like the UniqueShip CSVs; the same checks on the real manifest when the data is present; stable for a fixed seed on computed splits |
| Loader | Parses a miniature fixture bundle; rejects a wrong sample rate |
| Features | Expected shapes for MFCC and log-mel; gain change shifts log-mel by a constant; cache build on a fixture writes array, index and meta; duplicate clip ids fail; unreadable clips are skipped and listed; too many failures stop the build; cache refuses a mismatched config and names the fields |
| Inference | Loads a tiny fixture model; predicts on a synthetic WAV; scores sum to ~1 |
| Invalid input | One test per row in the section 6 error table |
| Reproducibility | Training for a few steps with a fixed seed twice gives the same loss; end-to-end smoke run on the synthetic fixture (train, evaluate, load in the app) |

Tests use synthetic audio, so the suite runs without the dataset.

## 10. Provisional values to settle after D3

Which CNN fold ships in the app. The A4 limits and the silence threshold were set from the dataset report (section 6).

## 11. Second dataset (P2) seam

A second dataset needs its own `DatasetLoader`, a coarse label mapping defined in advance in config (UniqueShip classes without a clean equivalent are excluded), and its own `group` key. Evaluation reports it as a separate section. The recording hardware differs (ShipsEar is 52.7 kHz, 24-bit), so all audio goes through the same resampling. Candidates and constraints are in the local dataset-review research notes (not in the repository); nothing is built for it in v1.0 P0/P1.

## 12. Compute workflow (E5)

1. **Kaggle:** attach the bundle as a Kaggle dataset, join and unzip it into `/tmp` (the temporary disk, about 1 TB), run `cache_features.py` for 32 kHz and 16 kHz, and save the caches and manifest to `/kaggle/working` as notebook output. The output is also saved as a private Kaggle dataset ("New Dataset from output"), so a CNN run on Kaggle can attach it without extracting the audio again (to be verified on the first run; if it fails, the cache is uploaded from the laptop).
2. **Local:** download only the caches and the manifest (about 3.5 GB) to `data/cache/` (git-ignored). Train, evaluate and run the app locally on the laptop; device auto-selection uses mps.
3. **Plain path** (README): download the bundle, run `scripts/download_data.py` to verify it, run `cache_features.py` locally. Needs about 44 GB free before the zip is deleted.

Kaggle's `/tmp` is wiped when a session ends, so step 1 is rerun from the zip if needed; `cache_features.py` is idempotent. Unzipped copies of the audio are not kept on Kaggle: the extraction (about 30 minutes) is paid once per run, and runs are batched (cache build and sample export together) to limit repeats.

**Development data.** Code and tests run on the laptop against synthetic fixtures (generated WAVs and CSVs shaped like the bundle) and the 36 sample clips in `app/samples/`. The baseline is trained locally on the full cache, not on the sample. The CNN and CRNN are trained on Kaggle (one epoch took 95 s and 9 GB of memory on the laptop): `notebooks/spectrogram_grid/upload_src.sh` uploads the last commit's code, its commit id and the split manifest as a private dataset, and `notebooks/cnn_train/` runs the training command line for every fold on a GPU with the cache dataset attached, then writes a `checksums.txt` of the run files. The runs are downloaded, checked against it and evaluated on the laptop with `--cache-dir` (section 8).
