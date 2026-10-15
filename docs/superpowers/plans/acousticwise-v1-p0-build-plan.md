# AcousticWise v1.0 — P0 build plan

- **Date:** 2026-10-15
- **Implements:** the P0 requirements (D1–E5) of `../specs/acousticwise-v1-prd.md`, built as described in `../specs/acousticwise-v1-technical-design.md`.
- **Budget:** about 10 hours a week.
- **Target:** v1.0 on **2026-11-25**, with a baseline-only fallback decided on **2026-11-18** (see "Date and cut line").

## How each slice works

We pair test-first, with fixed roles (`CLAUDE.local.md`):

1. Claude writes one small failing test and runs it, to show it fails for the right reason.
2. The author writes the production code in `src/`, `app/` or `scripts/` until it passes.
3. Repeat until the slice's tests are all green, then commit under the author's name.

A slice is done when its tests pass and its "Done when" line holds. Tests run on synthetic fixtures, so no slice needs the dataset except S6, S8, S9 and S15 (real runs).

## Slices

Hours are for the pair, including tests, code and review. They are estimates; the actual hours are logged as we go.

### Phase A: foundations (no real data)

| # | Slice | Req. | First tests Claude writes | Code the author writes | Hours |
|---|---|---|---|---|---|
| S0 | **Scaffold** | E1, E2 | `pytest` runs and finds an empty test; `import acousticwise` works | Nothing (Claude writes `pyproject.toml`, pytest and ruff config, `configs/default.yaml`, empty package folders) | 1 |
| S1 | **Fold loader and leakage test** | D1, M3 | A fixture bundle (tiny CSVs shaped like UniqueShip) loads into `Recording`s with clip id, label and group; background gets its day as group; train/val/test ids and groups are disjoint in every fold; a planted leak fails loudly; `splits.json` round-trips | `data/dataset.py`, `data/uniqueship.py`, `data/splits.py` | 4 |
| S2 | **Audio input and preprocessing** | D2, A3, A4 | Load a synthetic WAV; stereo becomes mono; 48 kHz resamples to 32 kHz; a 4.0 s clip pads to 5 s; a 3.9 s clip raises `AudioTooShortError`; a 12 s file gives the expected number of 5 s windows at 2.5 s hop; non-WAV and corrupt files raise `UnsupportedFileError`; silence raises `SilentAudioError` | `errors.py`, `audio/io.py`, `audio/preprocess.py` | 4 |
| S3 | **Features** | M1, M2 | Log-mel is 64 × 313 for a 5 s clip at 32 kHz and at 16 kHz; a gain change shifts log-mel by a constant; MFCC summary has 40 values; peak, RMS and no normalization give the expected offsets | `audio/features.py` | 3 |
| S4 | **Feature cache builder** | E5 | On a fixture bundle: writes `logmel.npy`, `index.csv` and `meta.json`; duplicate clip ids fail; an unreadable clip is skipped and listed; more than 0.1% failures stop the build; padded clips are counted; loading with a changed setting fails and names the field | `scripts/cache_features.py`, cache reader in `audio/features.py` | 4 |
| S5 | **Sample export** | P1 (batched) | From a fixture bundle: picks 3 clips per class from fold 1 test, falls back to other folds when a class has no fold 1 test vessel, and writes `manifest.csv` (class, fold, subset, MMSI, file name) | `scripts/export_samples.py` | 1.5 |

**Phase A: 17.5 hours.**

### Phase B: the one Kaggle run and the data report

| # | Slice | Req. | First tests Claude writes | Code the author writes | Hours |
|---|---|---|---|---|---|
| S6 | **Kaggle batch run** | E5 | (No new unit tests; S4 and S5 cover the code.) Checklist: the run's log shows both caches and 36 clips; checksums of downloaded files match the run's log | Claude prepares and pushes the notebook with the CLI; the author reviews it. The notebook installs the package from a small private Kaggle dataset holding the repo code (notebook internet needs phone verification, so this avoids it), extracts the bundle, builds `sr32000` and `sr16000`, exports the samples and saves the output as a private Kaggle dataset. Download to `data/cache/` | 2 |
| S7 | **Dataset report** | D3, A4 | Report on the fixture lists class, clip and vessel counts per fold, durations and padded clips | `data/report.py`; the author then sets the A4 file limits and silence threshold in config from the real report | 2 |

**Phase B: 4 hours** (plus about an hour of Kaggle waiting, not counted).

### Phase C: baseline end to end (first real results)

| # | Slice | Req. | First tests Claude writes | Code the author writes | Hours |
|---|---|---|---|---|---|
| S8 | **Baseline training** | M1, M6 | On a fixture cache: trains, writes a run directory (`model.joblib`, `config.json`, `labels.json`, `splits.json`, `metrics.json`); same seed gives the same model; a stale cache is refused | `models/baseline.py`, `train.py` | 4 |
| S9 | **Evaluation** | M4 | Hand-built predictions give the expected 9-class and 12-class macro-F1; a class with no test clips is left out of the 12-class score and listed; test-vessel count and largest-vessel share are correct; five folds give mean and spread | `evaluate.py` | 4 |
| — | **Real baseline run** | M3, M4 | — | Train and evaluate the baseline on all five folds locally on the cache; pick SVM or random forest on validation; record results | (compute) |

**Phase C: 8 hours.**

### Phase D: inference, app, reproducibility, release

| # | Slice | Req. | First tests Claude writes | Code the author writes | Hours |
|---|---|---|---|---|---|
| S10 | **Predictor** | A1–A4 | Loads a tiny fixture run; predicts on a synthetic WAV; scores sum to about 1; a long file averages window softmax; each error-table row raises its typed error; unexpected model errors become `InferenceError` | `inference.py` | 3 |
| S11 | **Streamlit app** | A1–A3, E2 | App imports only `inference`; `uv run acousticwise-app` entry point exists; an error maps to a friendly message (Streamlit's testing API) | `app/streamlit_app.py`: upload, classify, result card with "model confidence", few-vessels note for rescue/dredger/pilotvessel, waveform and spectrogram, error banner | 4 |
| S12 | **Reproducibility gate** | §3 gates | Tier 1: an end-to-end smoke test on the fixture (train a few steps, evaluate, load in the app); same seed twice gives the same loss | Fix whatever the smoke test exposes; then run tier 2 on a clean clone with the cache (baseline within ±1 point) | 2 |
| S13 | **README and release candidate** | E4, E5 | — | README with diagram, provenance, results table, limitations, screenshots, the plain path and the two-tier gate; `DATA.md` citation; a clean-environment launch check. This is a releasable **baseline-only v1.0** | 3 |

**Phase D: 12 hours.**

### Phase E: CNN (last; first to cut)

| # | Slice | Req. | First tests Claude writes | Code the author writes | Hours |
|---|---|---|---|---|---|
| S14 | **CNN model and training** | M2, M6 | Forward pass gives 12 logits for a 64 × 313 batch; a few steps on a fixture lower the loss; same seed gives the same loss; device picks cuda, then mps, then cpu | `models/cnn.py`, CNN path in `train.py` | 5 |
| S15 | **CNN results and comparison** | M2, M5 | Comparison table has both models for 9-class and 12-class | Train on five folds (locally on mps, or on Kaggle by attaching the cache dataset if too slow); tier 2 check within ±2 points; update README with the comparison and the explanation if the CNN does not win; tag v1.0 | 4 |

**Phase E: 9 hours** (plus training time, not counted).

## Totals

| | Hours |
|---|---|
| Phases A–D (baseline-only v1.0) | 41.5 |
| Phase E (CNN) | 9 |
| Subtotal | 50.5 |
| Buffer (20%) | about 10 |
| **Total** | **about 60** |

## Date and cut line

At 10 hours a week, starting 2026-10-15:

| Week | Dates | Slices | Checkpoint |
|---|---|---|---|
| 1 | Oct 15–21 | S0, S1, S2 | Leakage test and preprocessing pass on fixtures |
| 2 | Oct 22–28 | S3, S4, S5 | Cache builder passes on fixtures |
| 3 | Oct 29–Nov 4 | S6, S7, S8 | Caches downloaded; baseline trains |
| 4 | Nov 5–11 | S9, real baseline run, S10 | **First real results** (baseline, five folds) |
| 5 | Nov 12–18 | S11, S12, S13 | **Baseline-only v1.0 candidate.** Cut decision on Nov 18 |
| 6 | Nov 19–25 | S14, S15 | **v1.0 with CNN, 2026-11-25** |

**Cut line.** On 2026-11-18, if S13 is done but fewer than about 10 hours remain before the date you want, tag the baseline-only release as v1.0 and do the CNN as v1.1. If a week slips, push the dates rather than cut P0 items other than the CNN. P1 work (sample browser page, demo, error analysis, vessel-vote table, public cache) starts only after v1.0.

## Risks to the plan

- **Kaggle code access (S6).** The notebook needs the package. Plan: upload the repo code as a small private Kaggle dataset. Fallback: paste the cache and export functions into the notebook.
- **"New Dataset from output" (S6).** Unverified. Fallback: upload the cache from the laptop only if the CNN has to train on Kaggle.
- **Kaggle output download of 36 WAVs plus two 1.7 GB arrays (S6).** Unverified at that size. Fallback: zip the output in the notebook.
- **CNN training time on mps (S15).** Unknown until S14. Fallback: train on Kaggle with the cache dataset attached.
- **Padded clips over about 1% (S6).** The ticket 03 rule says revisit the padding choice; this would add time to week 3.
