# AcousticWise v1.0 — Product Requirements Document

- **Status:** Shipped as v1.0 on 2026-11-11, with the baseline, CNN and CRNN (primary dataset changed from ShipsEar to UniqueShip on 2026-10-09)
- **Date:** 2026-10-08
- **Source:** `AcousticWise v1.docx` (Definition of Done, plan, stretch objective)

## 1. Problem, audience, goals, non-goals

### Problem
Many audio-classification demos report inflated accuracy because clips from the same recording or the same ship land in both train and test sets. AcousticWise v1.0 is a small, honest, reproducible vessel-type classifier for passive underwater sound.

### Audience
- **Primary:** portfolio reviewers, who judge the README, the rigor of the evaluation and whether the app runs.
- **Secondary:** the author, extending the project toward maritime acoustic intelligence. This shapes design: clean module boundaries and a dataset loader behind a small interface. It does not add v1.0 features.

### Goals
- **G1:** classify a 5-second (or longer) hydrophone WAV into a UniqueShip vessel class, or "background" (no vessel), with model confidence scores.
- **G2:** show a reproducible, leakage-free comparison of an MFCC baseline against a log-mel CNN.
- **G3:** run as a local app that another person can launch with one command.
- **G4:** document the data, limitations and failure cases candidly.

### Non-goals (v1.0)
- Identifying individual ships.
- Real-time or streaming input, or hydrophone integration.
- A pre-promised accuracy target, or calibrated probabilities (scores are labeled "model confidence").
- Hosted or cloud deployment, accounts, or databases.
- Multi-dataset training. Cross-source testing is P2.
- Claims about other sites, hydrophones or regions (see Scope decision).

### Scope decision
Closed-set classification on **UniqueShip, "12 Class - 5 Hours Each" bundle, v1.0** (Ocean Networks Canada hydrophones, Strait of Georgia; CC-BY 4.0 stated on the dataset site; labels derived from AIS). It uses the dataset's own five vessel-disjoint folds, not a split of our own.

The 12 classes are: background (no vessel), cargo, dredger, fishing, other, passengership, pilotvessel, pleasurecraft, rescue, sailing, tanker, tug.

**Why UniqueShip and not ShipsEar.** ShipsEar has no stated licence and access is by email request (sent 2026-10-09, no reply yet), so it cannot be the foundation of a reproducible project. UniqueShip is openly licensed, large, and its splits were verified leakage-free on 2026-10-09 (below). ShipsEar becomes the P2 cross-source test if access is granted.

**Verified facts about the bundle (inspection of 2026-10-09, all five folds):**
- Mono WAV, 32 kHz, 32-bit PCM, 5.0 s per clip (a few are shorter, minimum 4.0 s). About 43,200 clips in total, roughly 60 hours (5 hours per class), across 4,218 vessels.
- Per fold, train/validation/test are about 80/10/10 by duration.
- Zero shared MMSI (vessel id) between train/validation, train/test and validation/test in every fold; zero shared background days within a fold; zero shared vessels between any two folds' test sets.
- The bundle README's layout shows only 5 classes; the actual bundle has 12. The site's other figure of 20 kHz is wrong; the audio is 32 kHz.

**Known weakness: the data is balanced by hours, not by vessels.** Distinct vessels per class in fold 1's training set: cargo 2,017; pleasurecraft 324; tug 237; fishing 190; tanker 186; sailing 169; other 135; passengership 88; rescue 18; **dredger 4; pilotvessel 1**. A model can score well on dredger or pilotvessel by memorizing one ship, and those classes' per-fold test scores rest on one or a few vessels. Hence two numbers, with the 9-class one leading:
- **9-class macro-F1 (headline)**: the classes with at least 50 distinct training vessels in every fold, plus background. That rule excludes rescue (at most 19), dredger (at most 5) and pilotvessel (at most 2); every other class has at least 88. It is computed from the same 12-class model, averaging F1 over the 9 classes on the full test set, so excluded-class clips predicted as a trusted class count against that class's precision. `other` has only 2 test vessels in fold 2, so its per-fold score is noisy.
- **12-class macro-F1**, shown beside it for completeness. The dataset authors publish no 12-class result; their best figure (66.5% accuracy, SwinV2-Tiny on Mel spectrograms, 5-fold mean, arXiv 2609.13659) is on the 5 Class - Balanced bundle, so it is not directly comparable. In each fold it averages over the classes that have test clips (dredger has none in folds 1 and 5, pilotvessel none in folds 3 to 5), and the missing classes are listed beside the score.

The generalization claim is limited to **unseen vessels recorded by the same Strait of Georgia hydrophone network**.

The dataset reviews behind these decisions are kept as local research notes and are not part of the repository.

## 2. Requirements

### P0 — must ship (Definition of Done)

**Data**
- **D1.** Use the UniqueShip "12 Class - 5 Hours Each" bundle. The licence is CC-BY 4.0 per the dataset site (the bundle's own README does not state it). The repo ships code, id-only split manifests and download instructions; audio is not committed (26 GB), except possibly a few sample clips with attribution (P1-DEMO). `DATA.md` records the source URL, bundle name and version, the paper citation, the licence and where it was read, the access date and the file checksums. Required attribution to Ocean Networks Canada is included.
- **D2.** Preprocessing pipeline: resample to one fixed rate, convert to mono, normalize, cut fixed-length windows. The same function is used in training and inference.
- **D3.** Dataset report: class counts, clip counts, durations, and the per-class vessel counts above for every fold; short clips; metadata columns with mixed types and missing values; the confounds (single site, hydrophone and season versus class; AIS-derived labels; `other` as a heterogeneous catch-all; one dominant class per vessel); and the README/site inconsistencies found.

**Modeling and evaluation**
- **M1.** Baseline: MFCC features plus a classical classifier.
- **M2.** CNN: log-mel input, PyTorch, compact. A CRNN variant (the same convolution layers, with an LSTM reading the time frames in order instead of time pooling) is trained and reported alongside it.
- **M3.** Use the dataset's five official folds. An automated test asserts, for every fold, zero shared MMSI across train/val/test and zero shared background days, and that no clip appears in two splits. Results are the mean and spread across the five folds.
- **M4.** Held-out report: accuracy, macro-F1 (12-class and 9-class), per-class precision/recall with the vessel count beside each class, confusion matrix. The unit is the clip. Beside each class the table also shows, per fold, the number of test vessels and the share of test clips from the class's largest test vessel, so the reader sees how much each score rests on a few ships (background is grouped by day).
- **M5.** Baseline vs CNN vs CRNN comparison, with a written explanation when a model does not win. A CRNN gain over the CNN counts only if it is larger than the spread between folds.
- **M6.** Model configuration, random seeds, the fold used and evaluation methodology are saved with each model artifact.

**Application**
- **A1.** Upload a WAV and submit it for inference.
- **A2.** Display the predicted class, per-class confidence (labeled "model confidence") and a waveform or spectrogram. The app states that the model was trained on one hydrophone network. All 12 classes are shown; if the top prediction is rescue, dredger or pilotvessel, the app adds a note that the class has few vessels in training and may reflect one ship rather than the type.
- **A3.** Handle unsupported files, silence, too-short audio and inference failures with clear messages. The app never shows a stack trace.
- **A4.** Supported input: PCM WAV, mono or stereo, at least 16-bit, within a maximum file size and a minimum duration; longer audio than the maximum duration is classified on its first part, with a note. Exact limits are set after D3.

**Engineering and delivery**
- **E1.** Data preparation, training, evaluation and inference are separate, reusable modules.
- **E2.** Dependency instructions, model-loading instructions and a one-command launch.
- **E3.** Automated tests for preprocessing, the split check, inference and invalid input.
- **E4.** README with architecture diagram, dataset provenance, measured results, limitations and screenshots.
- **E5.** Documented data-and-compute path that does not need the 26 GB audio on the author's laptop: features are extracted on Kaggle, and training and the app run locally. The README also gives the plain path (download, unzip, run) with its disk requirement (about 18 GB zipped plus 26 GB unzipped), and the two-tier reproducibility gate (section 3).

### P1 — should ship
- Error analysis: at least five reviewed misclassifications with spectrograms.
- Model card and a bundled sample recording for the demo.
- **Sample browser** (not in v1.0). A page in the local app where the user picks a class and plays 3 sample clips of it (36 clips in the repo, 3 per class, about 23 MB, drawn from fold 1 test so the clips come from vessels the fold 1 model never trains on; dredger and pilotvessel use other folds where fold 1 has no test vessel). The clips carry CC-BY 4.0 attribution to Ocean Networks Canada.
- Class-imbalance handling, with the chosen approach documented. Imbalance here is of vessels, not clips, so class weights alone do not fix it; the 9-class headline is the main mitigation.
- Per-vessel results: each test vessel (or background day) gets one prediction from the mean softmax of its clips, scored as a separate table. Cargo vessels have only 1–2 clips each, so the vote changes little for cargo; the other ship classes have 8 to 35 clips per vessel or more.
- Public feature cache: the Kaggle cache dataset made public with ONC attribution (CC-BY 4.0), so a reviewer can run tier 2 of the reproducibility gate with a 3.5 GB download instead of the 26 GB audio.
- **Demo (P1-DEMO).** The README embeds a 60–90 second screen recording or GIF of the local app, and the repo includes three demo clips: one correctly classified, one misclassified, one invalid (for example silent or too short). The correct and misclassified clips are chosen from UniqueShip test clips once a model exists; the invalid clip is generated. UniqueShip clips may be included with attribution under CC-BY 4.0. The four public ShipsEar sample WAVs have no stated licence and are used only locally, never committed; if shown in the app, they are labelled "out-of-domain, not a validated result".

### P2 — stretch (only after v1.0)
- **Cross-source test with full ShipsEar**, if the maintainers grant access. It is reported separately from the same-source test, uses a coarse mapping from UniqueShip classes to ShipsEar's A–E defined in advance (classes with no clean equivalent, such as `other`, `rescue` and `pilotvessel`, are excluded), and states that audio from a different hardware chain (52.7 kHz, 24-bit) is resampled. With only the four public sample WAVs this is a qualitative demo, never a metric.
- Other candidate sources (DeepShip, Oceanship, QiandaoEar22; all unverified for licence, identifiers and mapping). Wolfset (lab-tank recordings of small motors) does not qualify and is at most an out-of-domain stress test.
- The UniqueShip **5 Class - Balanced** bundle (213 h, 3,175 vessels, no rare classes) as a second benchmark. It is 64.8 GB zipped, so it is attempted only if it can be downloaded directly in a Kaggle notebook.
- Probability calibration.
- An "uncertain" output when top confidence is below a threshold.
- A public hosted demo (for example Streamlit Community Cloud), subject to model size and dependency-weight constraints.

## 3. Success metrics

### Release gates (pass/fail at release)

| Metric | Pass condition |
|---|---|
| DoD completeness | All P0 items (D1–E5) checked, each linked to evidence |
| Leakage | Automated test shows zero shared MMSI, zero shared background days and zero shared clips across train/val/test in every one of the five folds |
| Reproducibility | Tier 1 (anyone, no data): a fresh clone passes `uv run pytest` and an end-to-end smoke run on the synthetic fixture (train a few steps, evaluate, load the model in the app). Tier 2 (author, clean clone plus the downloaded feature cache): retraining from the saved config and seed reproduces the reported test metrics within ±1 point macro-F1 for the baseline and ±2 for the CNN |
| Launch | A person other than the author (or a clean environment following the README) runs the app with one command |
| Robustness | Silent, too-short, non-WAV and corrupt files each yield a friendly error, covered by tests |
| Inference | The saved model predicts on a previously unseen recording |
| Test suite | All automated tests pass |

### Reported, not gated

| Metric | Reported as |
|---|---|
| Macro-F1 | Baseline vs CNN, 9-class (headline) and 12-class, mean ± spread over five folds |
| Accuracy | Per clip |
| Per-class P/R/F1 | Table with vessel count per class, plus confusion matrix |
| Variance | Across folds, and across at least 3 seeds if time allows |
| Errors | At least five reviewed misclassifications (P1) |
| Latency | Seconds for one typical file, informational |

### Interpretation rules
- If the CNN does not beat the baseline, v1.0 is still complete provided the README explains why (class overlap, label noise, signal characteristics).
- Unusually high numbers for dredger, pilotvessel or rescue are expected to reflect single-ship memorization, and the README says so.
- The README states the claim as "generalizes to unseen vessels at the same hydrophone sites", not "works in the field" and not "works on other recording hardware".

## 4. Scope, risks, open questions

### Scope

The build was tracked as one GitHub issue per slice. P1 and P2 work starts only after v1.0. If the schedule had slipped, the CRNN would have been cut first, then the CNN (shipped as v1.1), before any other P0 item.

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Few vessels in some classes (dredger 4, pilotvessel 1, rescue 18 in fold 1 train) | Memorization of one ship; noisy per-class scores | 9-class headline; vessel count beside every class; per-vessel results; say so in the README |
| Single source: one hydrophone network, one region | Does not show generalization to other sites or hardware | Limit the claim; P2 cross-source test |
| Labels come from AIS, and `other` is a mixed class | Label noise; a class with no coherent sound | Report it; do not over-interpret `other` |
| The full dataset and even this bundle are large (17.8 GB zipped, 26 GB unzipped), hosted on Google Drive in 10 GB parts | Hard to reproduce; link rot or quota; Kaggle `/tmp` is wiped between sessions | Cache features; record checksums in `DATA.md`; document the Kaggle path and the plain path |
| ShipsEar access depends on one maintainer (P2 only) | P2 may not happen | Not on the critical path; P2 is optional |
| CNN underperforms baseline | Weak story | Acceptable; explain using class overlap and label noise |
| UI work slips (week 5) | Late release | Use Streamlit; no custom frontend |

### Open questions (do not block the PRD)
1. Final A4 file limits (depends on D3).
2. License for the project's own code (for example MIT).
3. Whether the ShipsEar maintainers will answer, and whether they permit sharing sample clips.
4. Whether the 5 Class - Balanced bundle can be downloaded directly into Kaggle.
