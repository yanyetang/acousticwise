# Wolfset (Lobo, Pessanha Santos & Moura, Sci Data 2025): suitability for AcousticWise v1.0

Provenance key: [PAPER] = stated in the local PDF (Sci Data 12:1323, doi 10.1038/s41597-025-05564-x); [WEB] = verified live on 2026-10-09 (figshare API, PMC); [DATA] = I downloaded the index CSV and one WAV and checked them; [INFER] = my inference.

## Name, size, sampling rate, bit depth, channels, format

### Takeaway
Wolfset is a small (about 5 h, 168 WAV files, 1.5 GB) controlled anechoic-tank recording set of outboard motors and a model-boat electric motor. It is mono, 44.1 kHz, 16-bit, uncompressed WAV. It is not ship recordings from the sea.

### Cited Findings
- Dataset name "Wolfset", about 1.5 GB, 168 WAVE files, about 5 hours, plus two annotation files (Wolfset_Index.xlsx and .csv) — [PAPER, Data Records] [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12311032/)
- Sampling rate 44.1 kHz, mono; 16-bit sound card; transfer function almost flat 50 Hz to 20 kHz; amplifier low-pass cutoff 22.4 kHz — [PAPER, Data Logging]
- Verified on a downloaded file (A20000T01N00.wav): 1 channel, 2 bytes per sample (16-bit), 44100 Hz, uncompressed PCM, 60.0 s — [DATA] via [figshare file](https://ndownloader.figshare.com/files/47094538)
- figshare record lists 170 files (168 WAV + 2 index files), total 1,580,072,787 bytes — [WEB] [figshare API](https://api.figshare.com/v2/articles/25791978)
- Index CSV has 168 rows. Durations sum to about 4.97 h. Most files are 60 s (94 files); 29 are 10 s (air-rifle shots); about 40 are 5-minute-plus files (typically 301 to 371 s); one is 1266 s (about 21 min) — [DATA] [index CSV](https://ndownloader.figshare.com/files/53442590)
- Hydrophone Bruel & Kjaer 8104, placed 2.5 m deep mid-tank. Amplifier B&K 2636 (two-stage, gains logged per file). HP oscilloscope and spectrum analyser monitoring — [PAPER]
- Paper states "No dedicated scripts or custom code were used" — [PAPER]

### Inferences
- [INFER] 44.1 kHz with a 22.4 kHz low-pass is fine for MFCC and log-mel. ShipsEar is also about 52 kHz or less, so you would resample to a common rate (e.g. 22.05 kHz or 16 kHz) as the pipeline probably already does.
- [INFER] The recordings are clipped in places ("saturation of the signal with loud transient noises", [PAPER]), and amplifier gain varies per file (20 to 40 dB in the index). Per-file level normalisation is advisable, and gain is a possible shortcut feature.

### Gaps
- Exact per-class file counts for any vessel-type scheme: none exist (see next section).

## Classes and labels, with counts, and mapping to ShipsEar

### Takeaway
Wolfset has no vessel-type classes. Labels are per-motor activity codes (5 motors, 6 states each), plus noise and transient codes. A closed-set vessel-type classifier does not map onto it cleanly. At most a coarse, contrived mapping is possible, and it would not be comparable to ShipsEar's five classes.

### Cited Findings
- Filename template XxxxxxTttNnn.WAV: X = A (regular) or E (event/anomaly); 5-digit motor code (motors 1 to 4 outboard combustion, motor 5 electric RC ship model; values 0 absent, 1 idle disengaged, 2 idle engaged forward, 3 slow forward, 4 medium forward, 5 varying; motor 5: 0 absent, 1 slow fwd, 2 fast fwd, 3 slow reverse, 4 fast reverse); T = transient code (sum of weights 1 compressed air, 2 water bucket, 4 mallet on tube, 8 hammer on tube, 16 air rifle); N = noise code (1 intense air bubbling, 2 low air bubbling, 4 low-flow hose, 8 high-flow hose); optional xNN repeat suffix — [PAPER, Table 1]
- Motors: Mercury 4.5 hp, 18 hp, 8 hp, 3.6 hp (3-blade propellers), plus a small RC ship model with a 3-blade propeller. Paper notes the Yamaha-branded motor in Fig. 8 but lists them as Mercury — [PAPER, Figs 7 and 8]
- Noise-only: N01 2x1 min + 1x21.1 min; N02, N04, N08 2x1 min each; N09 (N01+N08) 1x5.03 min. Total 34.13 min — [PAPER, Table 2]
- Transient-only: T01, T02, T04, T08 1x5 min each; T16 air rifle 32x10 s — [PAPER, Table 3]
- Motor 1 to 4 recordings total 54, 52, 57 and 45.8 min respectively — [PAPER, Data Records]. Motor 5 (electric) 7 min (1 min each of slow/fast fwd, slow/fast reverse, slow fwd with rudder, 2x fast fwd with rudder) — [PAPER, Table 5]. 11 pairwise or multi-motor combos, 1x1 min each — [PAPER, Table 6]
- Count of index rows by number of active motors: 109 single-motor files (including those with added noise or transients), 46 with motor code 00000 (noise-only or transient-only), 13 multi-motor — [DATA] (my own count of the index CSV; paper gives only durations)
- Index CSV has a free-text Observations column flagging failures (e.g. "Engine 4 stalled several times", "The 4th shot failed", "Silence from 3'40\"", "It was in reverse gear") — [DATA]
- No mention of cargo, tanker, passenger ferry, tug, ro-ro etc. anywhere in the paper — [PAPER]

### Inferences
- [INFER] No honest mapping to ShipsEar's classes (A fishing/tug/trawler-type, B motorboat/ferry/sail, C ro-ro/ferry, D ocean liner/ro-ro, E ambient noise). The nearest defensible, coarse mappings are:
  - Wolfset motors 1 to 4 (combustion outboards) on a single-motor basis maps loosely to ShipsEar "Class B" (motorboats, small craft). Treat that as a stretch: outboards on a floodgate in a tank are not hulls.
  - Motor 5 (electric) has no ShipsEar equivalent.
  - Motor code 00000 with noise/transient code 00 would be "ambient": but there are no such files (every 00000 file has added bubbles, hose or transients). The nearest is none; noise-only files are artificial.
- [INFER] A possible Wolfset-internal task: classify which of 5 motors is active (recording-level split, about 109 single-motor files). Useful as a demo but it is a different task from vessel-type classification.
- [INFER] Cross-source test is feasible only as a binary "small-engine present vs not" or as a negative control. I would not call it a class-mapped generalisation test.

### Gaps
- No official class-to-vessel-type mapping; any mapping above is mine and unvalidated.

## Recording setting, independence from ShipsEar

### Takeaway
Wolfset is genuinely independent from ShipsEar (different institution, site, hardware, era), but it is a laboratory tank, which is a large domain shift and not a "second field source".

### Cited Findings
- Recorded in the anechoic tank built in 1976 at the Lisbon Naval Base (Portuguese Navy, used for sonar calibration). Tank 8 m long, 5 m wide, 5 m deep; walls lined with cork-and-rubber absorber spikes. Outboards attached to the floodgate gate; hydrophone on a movable bridge — [PAPER, Methods]
- Added "real-world" disturbances: compressed-air bubbles, water pump/hose about 15 cm above the surface, metal-bar strikes, air rifle shots, water bucket discharge — [PAPER]
- Authors affiliated with Portuguese Navy Research Center (CINAV), NOVA IMS, CINAMIL, ISR/IST — [PAPER, p.1]
- Recording dates: not stated in the paper (Received 10 Apr 2025; Accepted 7 Jul 2025; published 30 Jul 2025 are publication dates) — [PAPER]. figshare record published 2025-06-09, version 1, not modified since — [WEB] [figshare API](https://api.figshare.com/v2/articles/25791978)
- ShipsEar (for contrast, from general knowledge, not re-verified here): field recordings in Spanish Atlantic coast ports (Vigo area) by a different group (Santos-Dominguez et al., 2016). [INFER, unverified in this session]

### Inferences
- [INFER] The common Portuguese/Iberian geography is incidental. There is no shared hardware, site or authors, so leakage between the two is not a worry.
- [INFER] Tank effects: reverberant-free water but a finite 8x5x5 m volume, motors fixed to a gate, close range (metres), a 50 Hz mains hum component (paper Fig. 15 removes it with a notch filter in the analysis), and no ship-hull propagation. A model trained on ShipsEar will probably degrade sharply, and the failure would reflect domain mismatch (lab vs field) rather than weak generalisation. This should be stated in any README.

### Gaps
- Recording dates and operator/session details are not given.

## Segmentation and grouping metadata (leakage control)

### Takeaway
Files are pre-cut, mostly 60 s clips (some 5-minute-plus and one 21 min). There is an index with per-file motor state, gains, and notes, but no session/date/vessel-ID field. Leakage control has to use the file name and motor ID.

### Cited Findings
- Index columns: ID, File Name, Length (bytes), Duration (s), Amplifier Gain 1 and 2 (dB), Intensity code, Description (Targets), Transient code, Transient note, Noise code, Noise note, Observations — [DATA]
- Repeated takes carry an xNN suffix (e.g. x01 to x05) of the same configuration — [PAPER, General Notes] and [DATA]
- Motor identity (1 to 5) is encoded in the code digits, so grouping by motor is possible — [PAPER]

### Inferences
- [INFER] Repeats (x01..x05) of the same configuration are near-duplicates (same motor, same RPM state, same tank, minutes apart). They must stay in the same split, so group by (motor-state, noise/transient code) or by motor ID. Splitting by file would leak.
- [INFER] With only 4 combustion motors and 1 electric motor, a "vessel" is effectively one of 5 physical engines. A held-out-vessel split would leave 1 motor out per fold.
- [INFER] Long files (5 to 21 min) would need windowing; windows from one file must not cross splits.

### Gaps
- No session, date or recording-run identifier. No time-of-day. No per-file hydrophone/position metadata (it is fixed per the paper).

## Licence, access, redistribution, citation, link status

### Takeaway
CC BY 4.0, open, no login. The link is live today. Audio can be redistributed with attribution, though at 1.5 GB it is better downloaded by script than committed.

### Cited Findings
- figshare record 10.6084/m9.figshare.25791978 (v1), status public, not embargoed, not confidential, licence CC BY 4.0 — [WEB] [figshare API](https://api.figshare.com/v2/articles/25791978)
- DOI resolves (HTTP 302 to figshare article page) on 2026-10-09 — [WEB] [doi.org](https://doi.org/10.6084/m9.figshare.25791978)
- Direct downloads work without authentication: ndownloader.figshare.com/files/<id> redirects (302) to an S3 URL; I fetched one WAV and the CSV successfully — [WEB/DATA]. MD5 checksums are supplied by the API for every file (useful for a fetch script).
- The figshare HTML page returned HTTP 403 to my fetch tool (bot protection), but the API works; this is a tooling limit, not evidence of the page being down — [WEB]
- Article is CC BY 4.0 open access: "permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit" — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12311032/). The dataset licence is CC BY 4.0 on figshare (above).
- Dataset citation per figshare: Pessanha Santos, Nuno; Moura, Ricardo; Lobo, Victor (2025). Wolfset: A High-Quality Underwater Acoustic Dataset for Algorithm Development and Analysis. figshare. Dataset. https://doi.org/10.6084/m9.figshare.25791978.v1 — [WEB]
- Paper citation: Lobo, V., Pessanha Santos, N. & Moura, R. Sci Data 12, 1323 (2025). doi 10.1038/s41597-025-05564-x — [PAPER]
- A copy of the paper is also in the NOVA repository (run.unl.pt, hdl 10362/186043) — [WEB, search result only; not opened]

### Inferences
- [INFER] CC BY 4.0 allows redistribution and derived features (e.g. log-mel tensors) with attribution and a note of any changes. For a public portfolio repo, a download script with MD5 checks plus a DATA_LICENSES/attribution note is the lighter path, and also keeps it consistent with whatever ShipsEar's terms require.

### Gaps
- Did not check ShipsEar's licence here (outside scope).

## Stated limitations and inferred caveats

### Takeaway
The authors state few limitations. The big caveats for AcousticWise are task mismatch, tiny diversity of sources, and artificial acoustics.

### Cited Findings
- Authors state low-frequency signals below 50 Hz are affected by the sound card's transfer function, and the signal sometimes saturates on loud transients because gain is fixed — [PAPER, Data Logging]
- Intended uses per authors: event monitoring, unsupervised classification, event detection in noise, anomaly detection, acoustic scene classification, signal enhancement, self-supervised learning, domain adaptation, few-shot — [PAPER, Usage Notes]. Vessel-type classification is not listed as an intended use.
- Some files flagged E or with notes of failures (stalled engines, failed shots) — [DATA]
- 50 Hz mains interference present in recordings — [PAPER, Figs 15 and 16 (authors filter it for analysis)]

### Inferences
- [INFER] As an alternative primary dataset: unsuitable. No vessel-type labels, 5 engines only, lab setting, short.
- [INFER] As a P2 cross-source generalisation test for ShipsEar-trained models: weak and potentially misleading. Class mapping is not defensible (see above), and the domain gap is tank vs harbour. If you want to keep it, frame it as an out-of-domain stress test or a negative control with an explicit caveat, not as a "second independent vessel source".
- [INFER] Recommended alternative: a field dataset with real vessel-type labels (e.g. DeepShip, or VTUAD/OSHEAN-type sets) is a better fit for P2. I have not researched those in this task.
- [INFER] Wolfset's best fit: a README "extension / out-of-domain demo" with 5-way engine classification using the same MFCC and log-mel code, with grouping by configuration to avoid repeat leakage.
- [INFER] Mains hum at 50 Hz and per-file gain differences are shortcut risks; apply a high-pass or notch consistently across both sources.

### Gaps
- Recording dates, per-class counts under any vessel taxonomy, and ShipsEar site and licence were not verified in this session.
