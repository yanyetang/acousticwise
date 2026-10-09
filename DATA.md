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

### How the data was handled

The raw bundle was uploaded to a private Kaggle dataset and extracted in a Kaggle notebook, because of limited local disk. Features are cached on Kaggle and only the cache and the split manifest come back to the laptop (see the technical design, section 12).

## Secondary dataset (P2): ShipsEar

| Item               | Value                                                                                                                                                              |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Source             | https://underwaternoise.atlanttic.uvigo.es (University of Vigo)                                                                                                    |
| Licence            | None stated; access is on request                                                                                                                                  |
| Permission request | Emailed to the maintainers on 2026-10-09. No reply yet.                                                                                                            |
| Outcome            | _to be filled in when the maintainers reply_                                                                                                                       |
| Local use          | Four public sample WAVs (ids 6, 22, 79, 82), kept outside the repo and never committed. Used only for a qualitative out-of-domain demo, not as a validated result. |
