# Dataset report

UniqueShip *12 Class - 5 Hours Each*, v1.0 (requirement D3 in the PRD). The counts at the end are generated from the bundle's fold CSVs by `scripts/dataset_report.py`; the sections before them are written by hand from the same CSVs and the feature cache (`data/cache/sr32000/index.csv`). Provenance, licence and checksums are in [`DATA.md`](../DATA.md).

## Summary

- 43,208 clips, 12 classes, about 3,600 clips (5 hours) per class. Each fold's train, validation and test CSVs together hold every clip exactly once.
- **Balanced by hours, not by vessels.** The whole dataset has 2,522 cargo vessels but only 23 rescue vessels, 6 dredgers and 2 pilot vessels. In fold 1 all 3,565 pilotvessel training clips come from one vessel. This is why the headline score uses 9 classes (PRD, section 2).
- **Some splits have no clips of a class** (table at the end): pilotvessel is in no validation set, and dredger is missing from the test sets of folds 1 and 5. Those folds cannot score those classes.
- **Recording level changes with the year** by about 18 dB (see Confounds), so absolute loudness must not reach the model.

## Clip lengths

- 32 kHz, mono, in every row. 13 clips are shorter than 5 s (shortest 4.0 s, none below the 4.0 s minimum), so 13 clips are padded in the 32 kHz cache, as the cache build reported.
- One of them, a fishing clip, is a single sample short (4.999969 s). The tables round it to 5.0 s but count it as short. At 16 kHz the resampled clip reaches a full 5 s, which is why the 16 kHz cache has 12 padded clips, not 13.

## Levels (from the 32 kHz cache)

- RMS level per clip: −51.0 to −16.3 dBFS. The quietest clip of every class is between −51.0 and −49.6 dBFS, so no clip is near digital silence.
- Peak level: −48.4 to 0.0 dBFS. Four clips reach about 0 dBFS at 32 kHz and may be clipped: tanker/20161128T010045, cargo/20160913T032340, cargo/20170711T003625, other/20170507T210550. Four in 43,208 is too few to matter; they are kept.

## Metadata columns

Each CSV has 42 columns. The project reads `label`, `path`, `duration_sec`, `date`, `MMSI` and `hydrophone`; the rest is AIS and water (CTD) context.

- **No missing values:** `label`, `path`, `duration_sec`, `sample_rate`, `date`, `MMSI` and the seven `has_msg*` flags.
- **Missing for background only (8.3% of rows):** `hydrophone`, the position and motion means (`mean_lat`, `mean_lon`, `mean_sog`, `mean_cog`, `interval_mean_distance`). Background rows have `MMSI` 0.
- **Often missing:** `imo` 54%, `destination` 45%, `mean_rot` 41%, `mean_true_heading` and `draught` 38%, `nav_status` and `maneuver` 33%, `callsign` 25%, and the CTD columns (`t1`, `c1`, `p1`, `sal`, `sv` and their `_norm` versions) 66%.
- **Mixed types:** `callsign` and `destination` hold text with some purely numeric values, so read them as text. `MMSI` and `class_code` load as floats (`316001251.0`) because of missing values elsewhere in the file; the project reads the CSVs with the `csv` module, so MMSI stays the string written in the file.
- `class_code` (the AIS ship type) is missing for 1,050 clips, all labelled `other`.

## Confounds

- **One site.** Every vessel position is between 49.065 and 49.102 N and between 123.305 and 123.363 W (Strait of Georgia). A model trained here learns this site's background noise and propagation; nothing in the dataset tests another site.
- **Two hydrophones, and background has none listed.** Vessel clips come from ICLISTENAF2523 (78%) or ICLISTENAF2556 (22%) in similar proportions for most classes; pilotvessel (33% on AF2556) and dredger (29%) lean slightly more on AF2556. Background rows give no hydrophone, so whether background matches the vessel recordings' hardware cannot be checked from the metadata.
- **Recording level changes with the year.** Median RMS level is about −30.5 dBFS for ICLISTENAF2523 in 2016–2018 and about −48 dBFS for everything else (AF2523 in 2021 and 2023, AF2556 in 2018–2022). Background follows the same pattern by year. The jump is most likely a gain or deployment change, not the vessels. Classes are not spread evenly over the years (for example, 27% of dredger and 30% of pleasurecraft clips are from 2023, against 15% of cargo), so loudness alone carries some class information. Features are therefore normalized per clip when loaded (technical design, section 5), and the comparison of peak, RMS or no normalization is made on validation data.
- **Season.** Pleasurecraft and sailing are mostly summer clips (May to September), dredger has no March clips. Season changes the background (rain, wind, marine life), which a model could learn instead of the vessel. This is documented, not corrected.
- **Labels come from AIS.** A clip's label is the AIS ship type of the vessel recorded near the hydrophone, not something heard in the audio. An unlisted vessel nearby (one without AIS), or a wrong AIS type, gives a wrong label; the rate is unknown.
- **`other` is a catch-all.** Its AIS types include 90 (other), 35 (military), 99, 39, 97, 57 (law enforcement), 54 (anti-pollution) and more, plus 1,050 clips with no type. Expect `other` to be hard to classify and confused with many classes.
- **One class per vessel.** Each of the 4,218 vessels has a single label in every clip, so no vessel changes class between clips.

## Inconsistencies with the bundle's README

- The README's layout lists 5 class folders; the bundle has 12.
- The README states no licence; the dataset site gives CC-BY 4.0 (see `DATA.md`).

<!-- Generated by scripts/dataset_report.py: edits below this line are overwritten. -->

## Counts per fold

Vessels or days: distinct vessels (MMSI), or distinct days for background.
Short clips: shorter than the 5 s window, so padded before features.

### Fold 1

| Split | Class | Clips | Vessels or days | Hours | Short clips | Shortest (s) |
|---|---|---:|---:|---:|---:|---:|
| train | background | 2880 | 641 | 4.00 | 0 | 5.0 |
| train | cargo | 2865 | 2017 | 3.98 | 2 | 4.0 |
| train | dredger | 1920 | 4 | 2.67 | 1 | 4.0 |
| train | fishing | 2877 | 190 | 4.00 | 1 | 5.0 |
| train | other | 2878 | 135 | 4.00 | 2 | 4.0 |
| train | passengership | 2879 | 88 | 4.00 | 0 | 5.0 |
| train | pilotvessel | 3565 | 1 | 4.95 | 2 | 4.0 |
| train | pleasurecraft | 2871 | 324 | 3.99 | 0 | 5.0 |
| train | rescue | 2932 | 18 | 4.07 | 0 | 5.0 |
| train | sailing | 2871 | 169 | 3.99 | 1 | 4.0 |
| train | tanker | 2877 | 186 | 4.00 | 2 | 4.0 |
| train | tug | 2871 | 237 | 3.99 | 0 | 5.0 |
| validation | background | 358 | 79 | 0.50 | 0 | 5.0 |
| validation | cargo | 279 | 194 | 0.39 | 0 | 5.0 |
| validation | dredger | 1681 | 2 | 2.33 | 0 | 5.0 |
| validation | fishing | 256 | 17 | 0.36 | 0 | 5.0 |
| validation | other | 298 | 14 | 0.41 | 0 | 5.0 |
| validation | passengership | 289 | 9 | 0.40 | 0 | 5.0 |
| validation | pleasurecraft | 284 | 32 | 0.39 | 0 | 5.0 |
| validation | rescue | 256 | 2 | 0.36 | 0 | 5.0 |
| validation | sailing | 289 | 17 | 0.40 | 1 | 4.0 |
| validation | tanker | 198 | 13 | 0.28 | 0 | 5.0 |
| validation | tug | 257 | 21 | 0.36 | 0 | 5.0 |
| test | background | 362 | 82 | 0.50 | 0 | 5.0 |
| test | cargo | 457 | 311 | 0.63 | 0 | 5.0 |
| test | fishing | 468 | 31 | 0.65 | 0 | 5.0 |
| test | other | 425 | 20 | 0.59 | 0 | 5.0 |
| test | passengership | 432 | 13 | 0.60 | 0 | 5.0 |
| test | pilotvessel | 36 | 1 | 0.05 | 0 | 5.0 |
| test | pleasurecraft | 445 | 50 | 0.62 | 0 | 5.0 |
| test | rescue | 412 | 3 | 0.57 | 0 | 5.0 |
| test | sailing | 441 | 26 | 0.61 | 0 | 5.0 |
| test | tanker | 526 | 34 | 0.73 | 0 | 5.0 |
| test | tug | 473 | 39 | 0.66 | 1 | 4.0 |

### Fold 2

| Split | Class | Clips | Vessels or days | Hours | Short clips | Shortest (s) |
|---|---|---:|---:|---:|---:|---:|
| train | background | 2864 | 641 | 3.98 | 0 | 5.0 |
| train | cargo | 2878 | 2017 | 4.00 | 1 | 4.0 |
| train | dredger | 3397 | 5 | 4.72 | 1 | 4.0 |
| train | fishing | 2876 | 190 | 3.99 | 1 | 5.0 |
| train | other | 2876 | 135 | 3.99 | 0 | 5.0 |
| train | passengership | 2873 | 88 | 3.99 | 0 | 5.0 |
| train | pilotvessel | 36 | 1 | 0.05 | 0 | 5.0 |
| train | pleasurecraft | 2882 | 325 | 4.00 | 0 | 5.0 |
| train | rescue | 2612 | 18 | 3.63 | 0 | 5.0 |
| train | sailing | 2870 | 169 | 3.99 | 2 | 4.0 |
| train | tanker | 2871 | 186 | 3.99 | 1 | 4.0 |
| train | tug | 2878 | 237 | 4.00 | 1 | 4.0 |
| validation | background | 366 | 81 | 0.51 | 0 | 5.0 |
| validation | cargo | 604 | 418 | 0.84 | 1 | 4.0 |
| validation | fishing | 588 | 39 | 0.82 | 0 | 5.0 |
| validation | other | 683 | 32 | 0.95 | 2 | 4.0 |
| validation | passengership | 562 | 17 | 0.78 | 0 | 5.0 |
| validation | pleasurecraft | 603 | 68 | 0.84 | 0 | 5.0 |
| validation | rescue | 464 | 2 | 0.64 | 0 | 5.0 |
| validation | sailing | 612 | 36 | 0.85 | 0 | 5.0 |
| validation | tanker | 576 | 37 | 0.80 | 0 | 5.0 |
| validation | tug | 578 | 48 | 0.80 | 0 | 5.0 |
| test | background | 370 | 80 | 0.51 | 0 | 5.0 |
| test | cargo | 119 | 87 | 0.17 | 0 | 5.0 |
| test | dredger | 204 | 1 | 0.28 | 0 | 5.0 |
| test | fishing | 137 | 9 | 0.19 | 0 | 5.0 |
| test | other | 42 | 2 | 0.06 | 0 | 5.0 |
| test | passengership | 165 | 5 | 0.23 | 0 | 5.0 |
| test | pilotvessel | 3565 | 1 | 4.95 | 2 | 4.0 |
| test | pleasurecraft | 115 | 13 | 0.16 | 0 | 5.0 |
| test | rescue | 524 | 3 | 0.73 | 0 | 5.0 |
| test | sailing | 119 | 7 | 0.17 | 0 | 5.0 |
| test | tanker | 154 | 10 | 0.21 | 1 | 4.0 |
| test | tug | 145 | 12 | 0.20 | 0 | 5.0 |

### Fold 3

| Split | Class | Clips | Vessels or days | Hours | Short clips | Shortest (s) |
|---|---|---:|---:|---:|---:|---:|
| train | background | 2882 | 642 | 4.00 | 0 | 5.0 |
| train | cargo | 2890 | 2018 | 4.01 | 1 | 4.0 |
| train | dredger | 3517 | 5 | 4.88 | 1 | 4.0 |
| train | fishing | 2870 | 190 | 3.99 | 0 | 5.0 |
| train | other | 2878 | 135 | 4.00 | 2 | 4.0 |
| train | passengership | 2882 | 88 | 4.00 | 0 | 5.0 |
| train | pilotvessel | 3601 | 2 | 5.00 | 2 | 4.0 |
| train | pleasurecraft | 2885 | 325 | 4.01 | 0 | 5.0 |
| train | rescue | 2896 | 18 | 4.02 | 0 | 5.0 |
| train | sailing | 2888 | 170 | 4.01 | 1 | 4.0 |
| train | tanker | 2872 | 186 | 3.99 | 2 | 4.0 |
| train | tug | 2887 | 238 | 4.01 | 1 | 4.0 |
| validation | background | 355 | 79 | 0.49 | 0 | 5.0 |
| validation | cargo | 382 | 267 | 0.53 | 0 | 5.0 |
| validation | fishing | 426 | 28 | 0.59 | 1 | 5.0 |
| validation | other | 339 | 16 | 0.47 | 0 | 5.0 |
| validation | passengership | 396 | 12 | 0.55 | 0 | 5.0 |
| validation | pleasurecraft | 374 | 42 | 0.52 | 0 | 5.0 |
| validation | rescue | 216 | 2 | 0.30 | 0 | 5.0 |
| validation | sailing | 340 | 20 | 0.47 | 1 | 4.0 |
| validation | tanker | 420 | 27 | 0.58 | 0 | 5.0 |
| validation | tug | 376 | 31 | 0.52 | 0 | 5.0 |
| test | background | 363 | 81 | 0.50 | 0 | 5.0 |
| test | cargo | 329 | 237 | 0.46 | 1 | 4.0 |
| test | dredger | 84 | 1 | 0.12 | 0 | 5.0 |
| test | fishing | 305 | 20 | 0.42 | 0 | 5.0 |
| test | other | 384 | 18 | 0.53 | 0 | 5.0 |
| test | passengership | 322 | 10 | 0.45 | 0 | 5.0 |
| test | pleasurecraft | 341 | 39 | 0.47 | 0 | 5.0 |
| test | rescue | 488 | 3 | 0.68 | 0 | 5.0 |
| test | sailing | 373 | 22 | 0.52 | 0 | 5.0 |
| test | tanker | 309 | 20 | 0.43 | 0 | 5.0 |
| test | tug | 338 | 28 | 0.47 | 0 | 5.0 |

### Fold 4

| Split | Class | Clips | Vessels or days | Hours | Short clips | Shortest (s) |
|---|---|---:|---:|---:|---:|---:|
| train | background | 2881 | 642 | 4.00 | 0 | 5.0 |
| train | cargo | 2895 | 2018 | 4.02 | 2 | 4.0 |
| train | dredger | 2041 | 5 | 2.83 | 0 | 5.0 |
| train | fishing | 2891 | 191 | 4.02 | 1 | 5.0 |
| train | other | 2877 | 135 | 4.00 | 2 | 4.0 |
| train | passengership | 2874 | 88 | 3.99 | 0 | 5.0 |
| train | pilotvessel | 3601 | 2 | 5.00 | 2 | 4.0 |
| train | pleasurecraft | 2882 | 325 | 4.00 | 0 | 5.0 |
| train | rescue | 3052 | 19 | 4.24 | 0 | 5.0 |
| train | sailing | 2888 | 170 | 4.01 | 2 | 4.0 |
| train | tanker | 2896 | 187 | 4.02 | 1 | 4.0 |
| train | tug | 2888 | 238 | 4.01 | 1 | 4.0 |
| validation | background | 357 | 80 | 0.50 | 0 | 5.0 |
| validation | cargo | 436 | 308 | 0.61 | 0 | 5.0 |
| validation | fishing | 500 | 33 | 0.69 | 0 | 5.0 |
| validation | other | 554 | 26 | 0.77 | 0 | 5.0 |
| validation | passengership | 429 | 13 | 0.60 | 0 | 5.0 |
| validation | pleasurecraft | 532 | 60 | 0.74 | 0 | 5.0 |
| validation | rescue | 268 | 2 | 0.37 | 0 | 5.0 |
| validation | sailing | 374 | 22 | 0.52 | 0 | 5.0 |
| validation | tanker | 414 | 27 | 0.57 | 1 | 4.0 |
| validation | tug | 398 | 33 | 0.55 | 0 | 5.0 |
| test | background | 362 | 80 | 0.50 | 0 | 5.0 |
| test | cargo | 270 | 196 | 0.38 | 0 | 5.0 |
| test | dredger | 1560 | 1 | 2.17 | 1 | 4.0 |
| test | fishing | 210 | 14 | 0.29 | 0 | 5.0 |
| test | other | 170 | 8 | 0.24 | 0 | 5.0 |
| test | passengership | 297 | 9 | 0.41 | 0 | 5.0 |
| test | pleasurecraft | 186 | 21 | 0.26 | 0 | 5.0 |
| test | rescue | 280 | 2 | 0.39 | 0 | 5.0 |
| test | sailing | 339 | 20 | 0.47 | 0 | 5.0 |
| test | tanker | 291 | 19 | 0.40 | 0 | 5.0 |
| test | tug | 315 | 26 | 0.44 | 0 | 5.0 |

### Fold 5

| Split | Class | Clips | Vessels or days | Hours | Short clips | Shortest (s) |
|---|---|---:|---:|---:|---:|---:|
| train | background | 2893 | 642 | 4.02 | 0 | 5.0 |
| train | cargo | 2876 | 2018 | 3.99 | 2 | 4.0 |
| train | dredger | 3529 | 5 | 4.90 | 1 | 4.0 |
| train | fishing | 2890 | 191 | 4.01 | 1 | 5.0 |
| train | other | 2895 | 136 | 4.02 | 2 | 4.0 |
| train | passengership | 2892 | 88 | 4.02 | 0 | 5.0 |
| train | pilotvessel | 3601 | 2 | 5.00 | 2 | 4.0 |
| train | pleasurecraft | 2880 | 325 | 4.00 | 0 | 5.0 |
| train | rescue | 2908 | 19 | 4.04 | 0 | 5.0 |
| train | sailing | 2887 | 170 | 4.01 | 2 | 4.0 |
| train | tanker | 2888 | 187 | 4.01 | 2 | 4.0 |
| train | tug | 2880 | 238 | 4.00 | 1 | 4.0 |
| validation | background | 351 | 81 | 0.49 | 0 | 5.0 |
| validation | cargo | 320 | 218 | 0.44 | 0 | 5.0 |
| validation | dredger | 72 | 1 | 0.10 | 0 | 5.0 |
| validation | fishing | 379 | 25 | 0.53 | 0 | 5.0 |
| validation | other | 277 | 13 | 0.38 | 0 | 5.0 |
| validation | passengership | 378 | 12 | 0.53 | 0 | 5.0 |
| validation | pleasurecraft | 294 | 33 | 0.41 | 0 | 5.0 |
| validation | rescue | 620 | 3 | 0.86 | 0 | 5.0 |
| validation | sailing | 255 | 15 | 0.35 | 0 | 5.0 |
| validation | tanker | 281 | 18 | 0.39 | 0 | 5.0 |
| validation | tug | 357 | 29 | 0.50 | 0 | 5.0 |
| test | background | 356 | 79 | 0.49 | 0 | 5.0 |
| test | cargo | 405 | 286 | 0.56 | 0 | 5.0 |
| test | fishing | 332 | 22 | 0.46 | 0 | 5.0 |
| test | other | 429 | 20 | 0.60 | 0 | 5.0 |
| test | passengership | 330 | 10 | 0.46 | 0 | 5.0 |
| test | pleasurecraft | 426 | 48 | 0.59 | 0 | 5.0 |
| test | rescue | 72 | 1 | 0.10 | 0 | 5.0 |
| test | sailing | 459 | 27 | 0.64 | 0 | 5.0 |
| test | tanker | 432 | 28 | 0.60 | 0 | 5.0 |
| test | tug | 364 | 30 | 0.51 | 0 | 5.0 |

## Classes with no clips

These splits cannot score the listed classes.

| Fold | Split | Classes |
|---:|---|---|
| 1 | validation | pilotvessel |
| 1 | test | dredger |
| 2 | validation | dredger, pilotvessel |
| 3 | validation | dredger, pilotvessel |
| 3 | test | pilotvessel |
| 4 | validation | dredger, pilotvessel |
| 4 | test | pilotvessel |
| 5 | validation | pilotvessel |
| 5 | test | dredger, pilotvessel |
