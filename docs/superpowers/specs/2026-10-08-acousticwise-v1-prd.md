# AcousticWise v1.0 — Product Requirements Document

- **Status:** Draft for review
- **Date:** 2026-10-08
- **Target release:** 2026-10-31
- **Source:** `AcousticWise v1.docx` (Definition of Done, plan, stretch objective)

## 1. Problem, audience, goals, non-goals

### Problem
Many audio-classification demos report inflated accuracy because clips from the same recording land in both train and test sets. AcousticWise v1.0 is a small, honest, reproducible vessel-type classifier for passive underwater sound.

### Audience
- **Primary:** portfolio reviewers, who judge the README, the rigor of the evaluation and whether the app runs.
- **Secondary:** the author, extending the project toward maritime acoustic intelligence. This shapes design: clean module boundaries and a dataset loader behind a small interface. It does not add v1.0 features.

### Goals
- **G1:** classify a WAV into a ShipsEar vessel class, or ambient noise, with model confidence scores.
- **G2:** show a reproducible, leakage-free comparison of an MFCC baseline against a log-mel CNN.
- **G3:** run as a local app that another person can launch with one command.
- **G4:** document the data, limitations and failure cases candidly.

### Non-goals (v1.0)
- Identifying individual ships.
- Real-time or streaming input, or hydrophone integration.
- A pre-promised accuracy target, or calibrated probabilities (scores are labeled "model confidence").
- Hosted or cloud deployment, accounts, or databases.
- Multi-dataset training. Cross-source testing is P2.

### Scope decision
Closed-set classification on ShipsEar only, using the five classes the dataset authors define: A (fishing boats, trawlers, mussel boats, tugboats, dredgers), B (motorboats, pilot boats, sailboats), C (passenger ferries), D (ocean liners, ro-ro) and E (background noise, serving as the "no vessel" class). The 11 raw vessel types are not used as labels: several have only a handful of recordings. The generalization claim is limited to unseen recordings from the same source.

See `reports/Underwater vessel dataset sources.md` for the dataset review behind these decisions.

## 2. Requirements

### P0 — must ship (Definition of Done)

**Data**
- **D1.** Use ShipsEar. ShipsEar has no stated licence (access is "on request" from the University of Vigo), so the working assumption is that redistribution is **not** permitted unless the maintainers say otherwise. The repo ships code, split manifests keyed by record id and download instructions, and no audio or derived clips. `DATA.md` records the source URL, the paper citation, the access date, the absence of a stated licence, and the date and outcome of the permission request. Use only the original recordings, never the pre-cut 5-second clip versions, which have no recording identifiers.
- **D2.** Preprocessing pipeline: resample to one fixed rate, convert to mono, normalize, cut fixed-length windows. The same function is used in training and inference.
- **D3.** Dataset report: class counts, recording counts, durations, quality issues, known limitations. It must reconcile the source paper's inconsistent size figures (Table 1 sums to about 81 files; the headline says 90) against the files actually obtained, and note that site is confounded with class (for example the dredger and the noise recordings come from different sites).

**Modeling and evaluation**
- **M1.** Baseline: MFCC features plus a classical classifier.
- **M2.** CNN: log-mel input, PyTorch, compact.
- **M3.** Splits are grouped by vessel (vessel name, and date where it helps), since one ship can appear in several recordings; no recording and no vessel appears in more than one split. An automated test asserts zero shared record ids and zero shared vessel names across train/val/test, and reports both. Because some classes have very few recordings (class D has about 9), results are reported from repeated stratified grouped cross-validation with the spread shown, in addition to one fixed held-out test split.
- **M4.** Held-out report: accuracy, macro-F1, per-class precision/recall, confusion matrix. Reported per window and per recording, with the aggregation method stated.
- **M5.** Baseline vs CNN comparison, with a written explanation if the CNN does not win.
- **M6.** Model configuration, random seeds and evaluation methodology are saved with each model artifact.

**Application**
- **A1.** Upload a WAV and submit it for inference.
- **A2.** Display the predicted class, per-class confidence (labeled "model confidence") and a waveform or spectrogram.
- **A3.** Handle unsupported files, silence, too-short audio and inference failures with clear messages. The app never shows a stack trace.
- **A4.** Supported input: PCM WAV, mono or stereo, at least 16-bit, within a maximum file size and minimum/maximum duration. Exact limits are set after D3.

**Engineering and delivery**
- **E1.** Data preparation, training, evaluation and inference are separate, reusable modules.
- **E2.** Dependency instructions, model-loading instructions and a one-command launch.
- **E3.** Automated tests for preprocessing, the split, inference and invalid input.
- **E4.** README with architecture diagram, dataset provenance, measured results, limitations and screenshots.

### P1 — should ship
- Error analysis: at least five reviewed misclassifications with spectrograms.
- Model card and a bundled sample recording for the demo.
- Class-imbalance handling, with the chosen approach documented.
- **Demo (P1-DEMO).** The README embeds a 60–90 second screen recording or GIF of the local app, and the repo includes three sample clips: one correctly classified, one misclassified, one invalid (for example silent or too short). Clips are included only if the dataset license permits redistribution (D1); otherwise the demo uses clips the author is allowed to share, and the README says so. A public hosted demo is not required (see P2).

### P2 — stretch (only after the Week 3 gate)
- Cross-source evaluation on a second dataset, clearly separated from the same-source test. The second source must have vessel-type labels that map onto ShipsEar's classes through a coarse mapping defined in advance, and must pass a licence and recording-identifier check. DeepShip is the leading candidate (unverified); Oceanship and QiandaoEar22 are alternatives. Wolfset (a lab-tank recording of five small motors, CC BY 4.0) does not qualify; it may be used only as a clearly labelled out-of-domain stress test or engine-ID demo, never as a cross-source generalization claim. Even with DeepShip, the result is an independent-source test of the method (retrain on each source, compare), not zero-shot transfer.
- Probability calibration.
- An "uncertain" output when top confidence is below a threshold.
- A public hosted demo (for example Streamlit Community Cloud), subject to the license, model size and dependency-weight constraints.

## 3. Success metrics

### Release gates (pass/fail by 2026-10-31)

| Metric | Pass condition |
|---|---|
| DoD completeness | All P0 items (D1–E4) checked, each linked to evidence |
| Leakage | Automated test shows zero source recordings and zero vessels shared across train/val/test |
| Reproducibility | A fresh clone reproduces reported test metrics from the saved config and seed within ±1–2 points macro-F1 |
| Launch | A person other than the author (or a clean environment following the README) runs the app with one command |
| Robustness | Silent, too-short, non-WAV and corrupt files each yield a friendly error, covered by tests |
| Inference | The saved model predicts on a previously unseen recording |
| Test suite | All automated tests pass |

### Reported, not gated

| Metric | Reported as |
|---|---|
| Macro-F1 | Baseline vs CNN on the held-out test set |
| Accuracy | Per window and per recording |
| Per-class P/R/F1 | Table plus confusion matrix |
| Variance | Results across at least 3 seeds, if time allows |
| Errors | At least five reviewed misclassifications (P1) |
| Latency | Seconds for one typical file, informational |

### Interpretation rules
- If the CNN does not beat the baseline, v1.0 is still complete provided the README explains why (dataset size, class overlap, signal characteristics).
- Unusually high test numbers must be accompanied by a discussion of possible residual leakage (for example, the same vessel recorded in different files).
- The README states the claim as "generalizes to unseen recordings from the same source", not "works in the field".

## 4. Timeline, risks, open questions

### Timeline

| Week | Dates (2026) | Deliverable | Gate |
|---|---|---|---|
| 1 | Oct 8–14 | Data pipeline, grouped split, baseline | Permission request emailed to the ShipsEar maintainers by **Oct 10** (D1); "no reply or unclear" is pre-accepted as the no-audio fallback, so Week 1 is not blocked waiting; leakage test passes |
| 2 | Oct 15–21 | Log-mel CNN, comparison, saved model and config | Baseline vs CNN macro-F1 recorded |
| 3 | Oct 22–28 | Inference module, UI, validation, tests | A second person or clean environment runs the app |
| 4 | Oct 29–31 | Bug fixes, README, diagram, demo, v1.0 tag | All release gates pass |

**Scope cut-off:** P1 and P2 work starts only after the Week 3 gate. If Week 2 slips, cut P1 items before touching any P0 item.

### Risks

| Risk | Impact | Mitigation |
|---|---|---|
| ShipsEar has no stated licence; access is by email request to a single maintainer who may not answer; the host is a 2016 single-maintainer site (link-rot risk) | Blocks public audio and may block data access | Assume no redistribution; ship code, manifests and download instructions only (D1); email the maintainers by Oct 10; record the outcome in `DATA.md`; obtain the data early |
| Few source recordings (about 90, possibly 81) and only about 9 in class D give a small, noisy test set | Unstable metrics | Grouped, class-stratified splits; repeated cross-validation with spread; caveat the claim |
| The same vessel recorded in several files; site confounded with class | Inflated results; model learns place, not vessel | Group by vessel (and date); document the site confound, which cannot be removed |
| Class imbalance; environment-specific cues for a class | Misleading accuracy | Macro-F1, class weights, discuss site/channel confounds |
| Residual leakage (same vessel or session in different recordings) | Inflated results | Document in README; P2 cross-source test if time allows |
| CNN underperforms baseline | Weak story | Acceptable; explain using dataset size and class overlap |
| Week 3 UI work slips | Late release | Use Streamlit; no custom frontend |

### Open questions (do not block the PRD)
1. Final A4 file limits (depends on D3).
2. Second dataset for P2: verify licence, access, identifiers and class mapping for DeepShip, Oceanship and QiandaoEar22 (all unverified); decide after the Week 3 gate.
3. License for the project's own code (for example MIT).
4. Per-type file counts for ShipsEar, and whether the full metadata table (id, vessel name, date, site) is available with the data. The dataset paper lists these fields, so vessel-and-date grouping is expected to be possible; confirm once the data is obtained.
5. Whether the ShipsEar maintainers will answer, and whether they permit sharing sample clips.
