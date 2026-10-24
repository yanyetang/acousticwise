# Data provenance

No audio is committed to this repository (`data/` is git-ignored). This file records where the data comes from, under which licence, and how to verify it.

## Primary dataset: UniqueShip, "12 Class - 5 Hours Each" (v1.0)

| Item        | Value                                                                                                                                                                            |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Source      | https://uniqueshipdata.org/ (hosted on Google Drive in 10 GB parts)                                                                                                              |
| Bundle      | 12 Class - 5 Hours Each, v1.0 (released September 2026 per the site)                                                                                                             |
| Origin      | Hydrophone recordings from Ocean Networks Canada (Strait of Georgia), labelled by vessel class from co-located AIS                                                               |
| Licence     | CC-BY 4.0, as stated on the dataset site for this bundle. The bundle's own README does not state a licence.                                                                      |
| Attribution | Ocean Networks Canada, and the UniqueShip authors. Cite the paper as directed on the site (the bundle README says "visit website for more information, paper, and how to cite"). |
| Obtained    | 2026-10-09                                                                                                                                                                       |
| Size        | 17.8 GB zipped (two parts), about 26 GB unzipped                                                                                                                                 |

### Checksums (SHA-256, computed 2026-10-09 on the downloaded files)

```
e27e57dddd6071f73625628f41b5d85ae55dee6fe30627be4ad0d6e7858354fa  allclass_5hours.z01  (10,737,418,240 bytes)
292e8c7ede32489a75a7bde7b4e2f72914c194e4311d0374e71778087559b874  allclass_5hours.zip  (7,059,560,361 bytes)
e2971023f1741951d1efda5270520953b296ed5221030ae34348ae47d0ae6b1b  README.md            (2,665 bytes)
```

The README checksum above is for the Google Docs export (2,665 bytes). The original file on Drive is 2,564 bytes with SHA-256 `2219878b52045d5da85454d9c9804613f70b5feaa91a9891e74a373730e2c835`.

### Unzipping

The archive is split. Both parts must be in one folder:

```
zip -s 0 allclass_5hours.zip --out allclass_5hours_joined.zip && unzip -q allclass_5hours_joined.zip -d stage/
```

### Layout and facts verified on 2026-10-09

- `audio/<class>/<start>_<end>.wav` and `fold_1/` … `fold_5/`, each with `metadata_5s_{train,validation,test}.csv`. Mono WAV, 32 kHz, 32-bit PCM, 5.0 s per clip (a few shorter, minimum 4.0 s). About 43,200 clips, roughly 60 hours.
- 12 classes: background, cargo, dredger, fishing, other, passengership, pilotvessel, pleasurecraft, rescue, sailing, tanker, tug. (The bundle README's layout lists only 5; the actual bundle has 12.)
- Splits are grouped by vessel (MMSI), with background split by whole day. In every fold: zero shared MMSI between train/validation, train/test and validation/test; zero shared background days; and zero shared vessels between any two folds' test sets.
- Distinct vessels per class in fold 1's training set: cargo 2,017; pleasurecraft 324; tug 237; fishing 190; tanker 186; sailing 169; other 135; passengership 88; rescue 18; dredger 4; pilotvessel 1. Audio is balanced by hours, not by vessels (see the PRD).

The two checks below ran on Kaggle on 2026-10-09 with `notebooks/kaggle_data_prep.ipynb`. The dataset report ([`docs/dataset-report.md`](docs/dataset-report.md)) regenerates the vessel counts from the package code, and its numbers agree with the table below.

**Distinct vessels per class, all five folds** (train/validation/test; background has no vessels):

| Class         | Fold 1       | Fold 2      | Fold 3       | Fold 4       | Fold 5       |
| ------------- | ------------ | ----------- | ------------ | ------------ | ------------ |
| cargo         | 2017/194/311 | 2017/418/87 | 2018/267/237 | 2018/308/196 | 2018/218/286 |
| pleasurecraft | 324/32/50    | 325/68/13   | 325/42/39    | 325/60/21    | 325/33/48    |
| tug           | 237/21/39    | 237/48/12   | 238/31/28    | 238/33/26    | 238/29/30    |
| fishing       | 190/17/31    | 190/39/9    | 190/28/20    | 191/33/14    | 191/25/22    |
| tanker        | 186/13/34    | 186/37/10   | 187/27/20    | 187/27/19    | 187/18/28    |
| sailing       | 169/17/26    | 169/36/7    | 170/20/22    | 170/22/20    | 170/15/27    |
| other         | 135/14/20    | 135/32/2    | 135/16/18    | 135/26/8     | 136/13/20    |
| passengership | 88/9/13      | 88/17/5     | 88/12/10     | 88/13/9      | 88/12/10     |
| rescue        | 18/2/3       | 18/2/3      | 18/2/3       | 19/2/2       | 19/3/1       |
| dredger       | 4/2/0        | 5/0/1       | 5/0/1        | 5/0/1        | 5/1/0        |
| pilotvessel   | 1/0/1        | 1/0/1       | 2/0/0        | 2/0/0        | 2/0/0        |

Clips per vessel in fold 1 train (min, median, max): cargo 1, 1, 2; pleasurecraft 8, 9, 9; tug 12, 12, 13; fishing 15, 15, 16; tanker 15, 15, 16; sailing 16, 17, 17; other 21, 21, 22; passengership 24, 33, 34; rescue 24, 232, 232; dredger 72, 144, 1,560; pilotvessel 3,565 (one vessel).

**Share of total power per frequency band, %** (fold 1 train, up to 200 clips per class, Welch spectra at 32 kHz):

| Class         | 0-1 kHz | 1-4 kHz | 4-8 kHz | 8-16 kHz |
| ------------- | ------- | ------- | ------- | -------- |
| background    | 98.69   | 0.82    | 0.28    | 0.21     |
| cargo         | 97.48   | 1.73    | 0.44    | 0.34     |
| dredger       | 72.74   | 21.71   | 3.72    | 1.83     |
| fishing       | 72.49   | 13.51   | 6.18    | 7.81     |
| other         | 84.63   | 9.11    | 2.46    | 3.79     |
| passengership | 96.61   | 1.72    | 0.95    | 0.72     |
| pilotvessel   | 58.80   | 36.38   | 2.92    | 1.90     |
| pleasurecraft | 89.29   | 6.80    | 2.18    | 1.73     |
| rescue        | 82.14   | 12.55   | 3.23    | 2.08     |
| sailing       | 79.33   | 13.42   | 4.27    | 2.97     |
| tanker        | 98.02   | 1.26    | 0.40    | 0.31     |
| tug           | 78.99   | 15.12   | 3.33    | 2.56     |

Mean power relative to background: every ship class is 6 to 18 dB above background in the 1-4, 4-8 and 8-16 kHz bands, except pilotvessel (1.2 to 8.2 dB) and sailing (5.6 to 6.3 dB). In the 8-16 kHz band, cargo, dredger, fishing, other, passengership, tanker and tug are 13 to 18 dB above background, which is why the project keeps the native 32 kHz rate. At 0-1 kHz, pilotvessel (-10.6 dB), sailing (-6.8 dB) and rescue (-2.4 dB) are below background.

### Split manifest

`scripts/build_manifest.py` turns the official fold CSVs into `artifacts/splits.json` (clip ids and groups, about 28 MB, not committed) and checks every fold for shared clips or groups between splits. Built on 2026-10-24 from the CSVs exported by `notebooks/metadata_export/`, it passes the check and has SHA-256:

```
84591f54320f34d6872f4b5de769869ba72eada9d8677fa98b37bac8e4916a18  artifacts/splits.json
```

### How the data was handled

The raw bundle was uploaded to a private Kaggle dataset (`yanyetang/uniqueship-12-class-5-hours`) and extracted in a Kaggle notebook, because of limited local disk. Features are cached on Kaggle and only the cache and the split manifest come back to the laptop (see the technical design, section 12).

On 2026-10-09 the Kaggle dataset was deleted by mistake and re-created from a fresh Drive download. The files were checked against the SHA-256 values above, and the dataset has a single version holding both parts (the last part uploaded as `allclass_5hours.zip.part`, because Kaggle rejects it as an invalid zip otherwise). The notebook `notebooks/kaggle_data_prep.ipynb` extracted it to 26 GB. The local copies of the zips were then deleted.

## Secondary dataset (P2): ShipsEar

| Item               | Value                                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Source             | https://underwaternoise.atlanttic.uvigo.es (University of Vigo)                                                                                                    |
| Licence            | None stated; access is on request                                                                                                                                  |
| Permission request | Emailed to the maintainers on 2026-10-09. No reply yet.                                                                                                            |
| Outcome            | _to be filled in when the maintainers reply_                                                                                                                       |
| Local use          | Four public sample WAVs (ids 6, 22, 79, 82), kept outside the repo and never committed. Used only for a qualitative out-of-domain demo, not as a validated result. |
