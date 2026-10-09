# AcousticWise v1.0 — Product Requirements Document

- **Status:** Draft v2 for review (revised 2026-10-09: primary dataset changed from ShipsEar to UniqueShip)
- **Date:** 2026-10-08
- **Target release:** 2026-10-31
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

**Known weakness: the data is balanced by hours, not by vessels.** Distinct vessels per class in fold 1's training set: cargo 2,017; pleasurecraft 324; tug 237; fishing 190; tanker 186; sailing 169; other 135; passengership 88; rescue 18; **dredger 4; pilotvessel 1**. A model can score well on dredger or pilotvessel by memorizing one ship, and those classes' per-fold test scores rest on one or a few vessels. Hence two headline numbers:
- **12-class macro-F1**, comparable with the dataset authors' published baseline (66.5% best, per the dataset site; not verified for this bundle).
- **9-class macro-F1**, excluding dredger, pilotvessel and rescue. This is the figure to trust.

The generalization claim is limited to **unseen vessels recorded by the same Strait of Georgia hydrophone network**.

See `docs/research_notes/Underwater vessel dataset sources/report.md` and `docs/research_notes/Open ShipsEar alternative datasets/report.md` for the dataset reviews behind these decisions.

## 2. Requirements

### P0 — must ship (Definition of Done)

**Data**
- **D1.** Use the UniqueShip "12 Class - 5 Hours Each" bundle. The licence is CC-BY 4.0 per the dataset site (the bundle's own README does not state it). The repo ships code, id-only split manifests and download instructions; audio is not committed (26 GB), except possibly a few sample clips with attribution (P1-DEMO). `DATA.md` records the source URL, bundle name and version, the paper citation, the licence and where it was read, the access date and the file checksums. Required attribution to Ocean Networks Canada is included.
- **D2.** Preprocessing pipeline: resample to one fixed rate, convert to mono, normalize, cut fixed-length windows. The same function is used in training and inference.
- **D3.** Dataset report: class counts, clip counts, durations, and the per-class vessel counts above for every fold; short clips; metadata columns with mixed types and missing values; the confounds (single site, hydrophone and season versus class; AIS-derived labels; `other` as a heterogeneous catch-all; one dominant class per vessel); and the README/site inconsistencies found.

**Modeling and evaluation**
- **M1.** Baseline: MFCC features plus a classical classifier.
- **M2.** CNN: log-mel input, PyTorch, compact.
- **M3.** Use the dataset's five official folds. An automated test asserts, for every fold, zero shared MMSI across train/val/test and zero shared background days, and that no clip appears in two splits. Results are the mean and spread across the five folds.
- **M4.** Held-out report: accuracy, macro-F1 (12-class and 9-class), per-class precision/recall with the vessel count beside each class, confusion matrix. The unit is the clip; per-vessel aggregation is added only if it makes sense for the data (most vessels have one or two clips).
- **M5.** Baseline vs CNN comparison, with a written explanation if the CNN does not win.
- **M6.** Model configuration, random seeds, the fold used and evaluation methodology are saved with each model artifact.

**Application**
- **A1.** Upload a WAV and submit it for inference.
- **A2.** Display the predicted class, per-class confidence (labeled "model confidence") and a waveform or spectrogram. The app states that the model was trained on one hydrophone network.
- **A3.** Handle unsupported files, silence, too-short audio and inference failures with clear messages. The app never shows a stack trace.
- **A4.** Supported input: PCM WAV, mono or stereo, at least 16-bit, within a maximum file size and minimum/maximum duration. Exact limits are set after D3.

**Engineering and delivery**
- **E1.** Data preparation, training, evaluation and inference are separate, reusable modules.
- **E2.** Dependency instructions, model-loading instructions and a one-command launch.
- **E3.** Automated tests for preprocessing, the split check, inference and invalid input.
- **E4.** README with architecture diagram, dataset provenance, measured results, limitations and screenshots.
- **E5.** Documented data-and-compute path that does not need the 26 GB audio on the author's laptop: features are extracted on Kaggle, and training and the app run locally. The README also gives the plain path (download, unzip, run) with its disk requirement (about 18 GB zipped plus 26 GB unzipped).

### P1 — should ship
- Error analysis: at least five reviewed misclassifications with spectrograms.
- Model card and a bundled sample recording for the demo.
- Class-imbalance handling, with the chosen approach documented. Imbalance here is of vessels, not clips, so class weights alone do not fix it; the 9-class headline is the main mitigation.
- Per-vessel results, so the reader can see how much each score rests on a few ships.
- **Demo (P1-DEMO).** The README embeds a 60–90 second screen recording or GIF of the local app, and the repo includes three sample clips: one correctly classified, one misclassified, one invalid (for example silent or too short). UniqueShip clips may be included with attribution under CC-BY 4.0. The four public ShipsEar sample WAVs have no stated licence and are used only locally, never committed; if shown in the app, they are labelled "out-of-domain, not a validated result".

### P2 — stretch (only after the Week 3 gate)
- **Cross-source test with full ShipsEar**, if the maintainers grant access. It is reported separately from the same-source test, uses a coarse mapping from UniqueShip classes to ShipsEar's A–E defined in advance (classes with no clean equivalent, such as `other`, `rescue` and `pilotvessel`, are excluded), and states that audio from a different hardware chain (52.7 kHz, 24-bit) is resampled. With only the four public sample WAVs this is a qualitative demo, never a metric.
- Other candidate sources (DeepShip, Oceanship, QiandaoEar22; all unverified for licence, identifiers and mapping). Wolfset (lab-tank recordings of small motors) does not qualify and is at most an out-of-domain stress test.
- The UniqueShip **5 Class - Balanced** bundle (213 h, 3,175 vessels, no rare classes) as a second benchmark. It is 64.8 GB zipped, so it is attempted only if it can be downloaded directly in a Kaggle notebook.
- Probability calibration.
- An "uncertain" output when top confidence is below a threshold.
- A public hosted demo (for example Streamlit Community Cloud), subject to model size and dependency-weight constraints.

## 3. Success metrics

### Release gates (pass/fail by 2026-10-31)

| Metric | Pass condition |
|---|---|
| DoD completeness | All P0 items (D1–E5) checked, each linked to evidence |
| Leakage | Automated test shows zero shared MMSI, zero shared background days and zero shared clips across train/val/test in every one of the five folds |
| Reproducibility | A fresh clone with the data present reproduces reported test metrics from the saved config and seed within ±1–2 points macro-F1 |
| Launch | A person other than the author (or a clean environment following the README) runs the app with one command |
| Robustness | Silent, too-short, non-WAV and corrupt files each yield a friendly error, covered by tests |
| Inference | The saved model predicts on a previously unseen recording |
| Test suite | All automated tests pass |

### Reported, not gated

| Metric | Reported as |
|---|---|
| Macro-F1 | Baseline vs CNN, 12-class and 9-class, mean ± spread over five folds |
| Accuracy | Per clip |
| Per-class P/R/F1 | Table with vessel count per class, plus confusion matrix |
| Variance | Across folds, and across at least 3 seeds if time allows |
| Errors | At least five reviewed misclassifications (P1) |
| Latency | Seconds for one typical file, informational |

### Interpretation rules
- If the CNN does not beat the baseline, v1.0 is still complete provided the README explains why (class overlap, label noise, signal characteristics).
- Unusually high numbers for dredger, pilotvessel or rescue are expected to reflect single-ship memorization, and the README says so.
- The README states the claim as "generalizes to unseen vessels at the same hydrophone sites", not "works in the field" and not "works on other recording hardware".

## 4. Timeline, risks, open questions

### Timeline

| Week | Dates (2026) | Deliverable | Gate |
|---|---|---|---|
| 1 | Oct 8–14 | Data pipeline over the official folds, cached features, baseline | Dataset decision recorded in `DATA.md` by **Oct 14**. Done so far: ShipsEar permission email sent Oct 9; UniqueShip inspected Oct 9 (sample rate, 12 classes, zero-leakage checks pass); leakage test in code passes |
| 2 | Oct 15–21 | Log-mel CNN, comparison, saved model and config | Baseline vs CNN macro-F1 recorded |
| 3 | Oct 22–28 | Inference module, UI, validation, tests | A second person or clean environment runs the app |
| 4 | Oct 29–31 | Bug fixes, README, diagram, demo, v1.0 tag | All release gates pass |

**Scope cut-off:** P1 and P2 work starts only after the Week 3 gate. If Week 2 slips, cut P1 items before touching any P0 item.

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Few vessels in some classes (dredger 4, pilotvessel 1, rescue 18 in fold 1 train) | Memorization of one ship; noisy per-class scores | 9-class headline; vessel count beside every class; per-vessel results; say so in the README |
| Single source: one hydrophone network, one region | Does not show generalization to other sites or hardware | Limit the claim; P2 cross-source test |
| Labels come from AIS, and `other` is a mixed class | Label noise; a class with no coherent sound | Report it; do not over-interpret `other` |
| The full dataset and even this bundle are large (17.8 GB zipped, 26 GB unzipped), hosted on Google Drive in 10 GB parts | Hard to reproduce; link rot or quota; Kaggle `/tmp` is wiped between sessions | Cache features; record checksums in `DATA.md`; document the Kaggle path and the plain path |
| ShipsEar access depends on one maintainer (P2 only) | P2 may not happen | Not on the critical path; P2 is optional |
| CNN underperforms baseline | Weak story | Acceptable; explain using class overlap and label noise |
| Week 3 UI work slips | Late release | Use Streamlit; no custom frontend |

### Open questions (do not block the PRD)
1. Final A4 file limits (depends on D3).
2. Final audio sample rate and window settings (depends on a spectrum check; see the technical design).
3. License for the project's own code (for example MIT).
4. Whether to also report a per-vessel aggregation, given that most vessels have one or two clips.
5. Whether the ShipsEar maintainers will answer, and whether they permit sharing sample clips.
6. Whether the 5 Class - Balanced bundle can be downloaded directly into Kaggle.
