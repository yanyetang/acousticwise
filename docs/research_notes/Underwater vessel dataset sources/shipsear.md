# ShipsEar underwater vessel noise dataset: suitability as AcousticWise v1.0 primary dataset

Labels used below: **[PAPER]** = stated in the local PDF (Santos-Dominguez et al., Applied Acoustics 113 (2016) 64-69, cited as Paper); **[WEB]** = verified on the web on 2026-10-09; **[INFER]** = my inference.

Paper: /Users/yanyetang/Projects/AcousticWise/docs/superpowers/specs/papers/ShipsEar_An_underwater_vessel_noise_dataset.pdf (DOI 10.1016/j.apacoust.2016.06.008)

## 1. Size, duration, sampling rate, bit depth, channels, format

### Takeaway
The paper says 90 recordings of 15 s to 10 min, 11 vessel types plus natural background noise, 52,734 Hz, 24-bit A/D, wav. It never states a total duration. Its own class table sums to only 81 files and about 170 min, which conflicts with the "90" headline.

### Cited Findings
- [PAPER] "Currently composed of 90 records representing sounds from 11 vessel types"; final database is 90 recordings in wav format lasting from 15 s to 10 min. — Paper, abstract and Sec. 2.2
- [PAPER] Recorder: MarSensing digitalHyd SR-1 autonomous recorder. Hydrophone sensitivity about -193.5 dB re 1 V/1 uPa, flat response 1 Hz-28 kHz. Preamp with 100 Hz high-pass, then programmable gain. 24-bit sigma-delta A/D at 52,734 Hz. Data stored as wav, 5-min files on a 16 GB SD card. — Paper, Sec. 2.1
- [PAPER] Up to 3 hydrophones at different depths and gains were deployed per session. The recording with the highest level before clipping was chosen for the database. So each database record is one channel, i.e. mono. (Mono is my reading of "recording ... selected"; the paper does not say "mono".) — Paper, Sec. 2.1-2.2
- [PAPER] Paper's own classification experiment Table 1 (merged classes, seconds and file counts): A 1729 s (16 files), B 1435 s (17), C 4054 s (27), D 2041 s (9), E 923 s (12). The "Experimental" subset after dropping 12 files for balance: A 16, B 15, C 17, D 9, E 12 files (durations 1729/1228/1098/2041/923 s). — Paper, Table 1
- [WEB] A 2026 arXiv paper says ShipsEar is "approximately 3.1 hours of audio from 90 recording sessions covering 11 vessel types sampled at 64 kHz and then resampled to 16 kHz". That paper contradicts the original on sampling rate (64 kHz vs 52,734 Hz). Trust the original. — [Towards Deployable Underwater Vessel Classification, arXiv 2609.29179](https://arxiv.org/html/2609.29179)
- [WEB] A 3.13 h total and "91 pairs" appears in a secondary comparison table summarised by search, not verified at source. — [Oceanship, arXiv 2401.02099](https://arxiv.org/pdf/2401.02099) (summary via search snippet only; the PDF could not be parsed by my fetch tool)
- [WEB] The official sample zip is live (83.9 MB, 4 wav files: 22__19_07_13_adventure_maniobra.wav, 6__10_07_13_marDeCangas_Entra.wav, 79__23_07_13_H3_zodiac.wav, 82__27_09_13_H3_lluvia.wav). Filenames are `<id>__<DD_MM_YY>_<name>.wav`, so id and date are encoded in the filename. — [samples.zip](https://underwaternoise.atlanttic.uvigo.es/samples.zip) (downloaded and listed)

### Inferences
- [INFER] Class table totals: 81 files, 10,182 s (about 170 min, 2.8 h) vs 90 files in the abstract and about 3.1 h in secondary sources. Roughly 9 files (maybe dredger or unlabelled-type records) are not in Table 1. Implementation must count from the actual files, not from the paper.
- [INFER] Mono, 24-bit source at 52.7 kHz. The paper found only the first 8 kHz useful for its classifier, so resampling to 16 kHz (as most later work does) is defensible. Verify headers after download.

### Gaps
- Exact per-vessel-type file counts and minutes for all 11 types are not in the paper (only merged A-E). They would have to be read from the online database table (indexDB.php) or from the files once obtained.
- Reconciling 90 vs 81 files and 2.8 h vs 3.1 h: not resolvable without the full data.

## 2. Classes and per-class counts; ambient noise

### Takeaway
11 vessel types plus a background-noise category (wind, rain, waves, current). The paper's experiment merges them into 4 size classes (A-D) plus noise (E). The class mix is imbalanced and small.

### Cited Findings
- [PAPER] The 'type' field values: fishing boat, trawler, mussel boat, pilot ship, tugboat, dredger, ro-ro, ocean liner, passenger ferry, sailboat, motorboat, and background noise (wind, rain, waves and current). — Paper, Sec. 2.2 (record fields)
- [PAPER] Ambient noise is included: 4 recordings of wind, rain, waves and current made at the Intecmar meteorological station outside the Ria de Vigo, over several days of heavy weather. — Paper, Sec. 2.2
- [PAPER] Merged classes: A = fishing boats, trawlers, mussel boats, tugboats, dredgers. B = motorboats, pilot boats, sailboats. C = passenger ferries. D = ocean liners and ro-ro. E = background noise. Counts in Table 1 above (C has 27 files and 4054 s, D has 9 files). — Paper, Sec. 3 and Table 1
- [PAPER] Authors' result: leave-one-out CV, GMM on cepstral coefficients, 75.4% overall on the 69-file subset, 100% presence/absence detection. Weakest class D (55.5%, fewest files). — Paper, Sec. 3.4, Table 3
- [WEB] Other papers use five classes (A-E) and report test accuracies above 99% on ShipsEar, described as "a well-defined, yet relatively straightforward benchmark". — [secondary summary via search of arXiv results, e.g. Oceanship](https://arxiv.org/pdf/2401.02099) (snippet only, not verified in full text); macro F1 0.9918 for a two-layer CNN reported in [arXiv 2609.29179](https://arxiv.org/html/2609.29179)

### Inferences
- [INFER] "Closed-set vessel-type classifier" on ShipsEar will in practice be 4-5 classes (A-E) with 9-27 files per class, or 11 fine-grained types with only a handful of files each (some fine types likely 1-5 recordings). Recording-level splits then leave very few test recordings per class, so metrics will have wide variance. Use repeated or stratified grouped CV rather than a single split.
- [INFER] The very high published accuracies (above 99%) are probably inflated by clip-level splits (see Q4), so a correct recording-level result will look much lower. That is a good portfolio story but needs honest framing.

### Gaps
- Per-type minutes for the 11 raw types not stated in the paper.

## 3. Recording method, sites, dates, hydrophones, metadata and split granularity

### Takeaway
Autumn 2012 and summer 2013, mostly in or near Vigo port (Spanish Atlantic coast), bottom-moored hydrophones, rich per-record metadata (date, GPS, depth, distance bin, wind, weather). Recording-level and date/site-level grouping is possible in principle from the metadata, but only once you have the metadata table.

### Cited Findings
- [PAPER] Recorded autumn 2012 and summer 2013; most in or near the port of Vigo (42 deg 14.5' N, 008 deg 43.4' W), Ria de Vigo (drowned river valley, 35 km long, max depth under 45 m, heavy traffic). — Paper, Sec. 2
- [PAPER] Hydrophones bottom-moored with a surface buoy; 3 hydrophones at different depths and gains where possible, 1-2 in very shallow water (under 10 m). — Paper, Sec. 2.1
- [PAPER] Sessions logged: date and time, noise type, GPS, hydrophone heights, approximate horizontal distance, channel depth, gain. Vessels visually identified; most sessions video-documented. Auxiliary "underwater ear" and a horn sound source used to add 2 kHz markers. — Paper, Sec. 2.1
- [PAPER] Scenarios: (a) hydrophones deployed from docks to capture docking and manoeuvring (high background noise from waves on the infrastructure); (b) normal operations at 3 recording sites in the middle of the Ria de Vigo, with an auxiliary vessel deploying hydrophones according to port-authority and AIS movement information; (c) opportunistic recordings, including a suction dredger in La Coruna's outer harbour; (d) natural background noise at Intecmar. — Paper, Sec. 2.2
- [PAPER] Record fields: id, name (often the vessel name), type, pic, location (GPS), date (YYYY-MM-DD HH:MM:SS, 00:00:00 if time unknown), H_G_D, audio, video, channel depth, wind speed (km/h), AIS link, distance (bins <50, 50-100, 100-150, >150 m), atmospheric/oceanographic data, duration, notes. — Paper, Sec. 2.2
- [WEB] The live online database lists these fields per record (checked: first record id 80 "Adricristuy", dredger, Punta Langosteira, 2012-10-04 14:00, channel depth 20 m, distance 350 m). — [indexDB.php](https://underwaternoise.atlanttic.uvigo.es/indexDB.php)
- [WEB] The sample filenames embed the id and recording date. — [samples.zip](https://underwaternoise.atlanttic.uvigo.es/samples.zip)

### Inferences
- [INFER] Vessel `name` is recorded (often the ship name), which allows grouping by vessel identity. This is a stricter and more honest split than by file, because the same ship (for example a ferry on its daily route) can appear in several recordings. Splitting by vessel name, and where possible date, should be considered.
- [INFER] Sites are few (Vigo port, 3 Ria sites, Punta Langosteira / La Coruna, Intecmar). Site is also confounded with class (for example dredger at Langosteira, noise at Intecmar), so a classifier can learn site or background signature instead of vessel signature. Site-held-out splits would be impossible for some classes.
- [INFER] The 4 noise recordings were made on different days in bad weather at a different place, so noise-vs-vessel is probably trivially separable (consistent with the 100% detection result).

### Gaps
- I did not scrape all 90 rows of the online table (about 10 records per page); a full table of date, site and vessel per record was not extracted. Do this once the audio is obtained.

## 4. Pre-segmentation and leakage risk

### Takeaway
The original distribution is one file per event or pass-by (15 s to 10 min), not pre-cut clips, so recording-level splits are possible from the original files. But the widely used derived versions are pre-cut 5-s clips with no recording identifier, which cannot be split safely.

### Cited Findings
- [PAPER] Recordings were "segmented with wide margins" to keep the beginning-to-end of an event or pass-by; each audio file with its metadata forms one record. — Paper, Sec. 2.2
- [PAPER] The authors' own evaluation was leave-one-out at file level, with one file per test utterance. — Paper, Sec. 3.3
- [WEB] A processed distribution used in the literature is a fixed split of pre-segmented 5 s clips (1,778 train, 445 test) where "source-recording identifiers are unavailable", so recording-level independence cannot be verified. The authors of that paper say this prevents verification of recording-independent generalisation and move to DeepShip for that reason. — [arXiv 2609.29179](https://arxiv.org/html/2609.29179)

### Inferences
- [INFER] Use only the original wav files (obtained from the maintainer) and do your own segmentation after assigning each recording to a split. Do not use community 5 s clip mirrors.
- [INFER] Even from the originals, adjacent segments of a long recording are highly correlated, and the 3-hydrophone setup means the same pass-by may exist in several channels. The paper says only the best channel was kept per record, so this duplication should not be in the database, but verify.

### Gaps
- Whether any original record is a continuation of the same event as another (for example consecutive 5-min files of one long pass) is not stated. Check by name, date and time in metadata.

## 5. Licence, access terms, citation, redistribution, hosting status

### Takeaway
No open licence is stated anywhere I could find. Hosting is on the University of Vigo site (live as of 2026-10-09): an 84 MB sample zip is downloadable, while the site says full access requires an email request. Treat audio as not redistributable.

### Cited Findings
- [WEB] The official URL `http://atlanttic.uvigo.es/underwaternoise/` still works: 302 redirect to `https://underwaternoise.atlanttic.uvigo.es`, which returns HTTP 200 (Apache, last-modified 2022-03-22). — verified with curl on 2026-10-09: [landing page](https://underwaternoise.atlanttic.uvigo.es/)
- [WEB] Landing page text: "You can download from here some samples of different types of ships included in the database ... To get full access to the database please send a request to the email contact below." Contact: Universidad de Vigo, Telecommunications Faculty, Campus Universitario Lagoas-Marcosende, 36310 Vigo; dsantos(at)gts.uvigo.es. Footer: "(c) Uvigo-Sonitum 2016". — [landing page](https://underwaternoise.atlanttic.uvigo.es/)
- [WEB] The landing page and the online database page state no licence (no Creative Commons, no terms of use, no citation text). The `indexDB.php` browsing interface is live and gives the metadata table. — [landing page](https://underwaternoise.atlanttic.uvigo.es/), [indexDB.php](https://underwaternoise.atlanttic.uvigo.es/indexDB.php)
- [PAPER] The paper says the database is "made available to the research community", "freely available to researchers, who are invited to contribute to its growth". No licence named, no redistribution terms. Funding: FEDER, project ITC-20113086 Ecodraga, Galician Government CN2011/019. — Paper, abstract, Sec. 4, Acknowledgements
- [PAPER] Citation: Santos-Dominguez D., Torres-Guijarro S., Cardenal-Lopez A., Pena-Gimenez A., "ShipsEar: An underwater vessel noise database", Applied Acoustics 113 (2016) 64-69. The paper contains no explicit "please cite" requirement; citing the paper is the de facto expectation. — Paper, header
- [WEB] I found no official mirror on Zenodo, Kaggle, Hugging Face or GitHub by search; web search returned only papers and the official URL. A Zenodo record 13333019 that appeared in results is unrelated (killer whale Dtag data). — [Zenodo 13333019](https://zenodo.org/records/13333019). The MarineInfo catalogue entry (marineinfo.org/id/dataset/8863) returned HTTP 406 and could not be read.

### Inferences
- [INFER] "Freely available to researchers" plus "request by email" means access is granted by the maintainers on request; it is not an open licence. Redistributing the audio in a public portfolio repo is therefore not permitted by anything I found. Safe approach: repo holds code, split files (ids and filenames only), metadata-derived manifests, and a download script or instructions pointing to the Vigo site and the email contact; do not commit wav files or derived clips. Optionally ask the maintainer in writing for permission and for licence terms.
- [INFER] Many people use it, so the full data has evidently been shared widely, but that does not create a licence. Any Kaggle or GitHub mirror of unknown provenance carries the leakage and licensing problems above.
- [INFER] Single-maintainer, 2016-era PHP site with last-modified 2022: link-rot or access risk is real for a portfolio repo meant to be reproducible. The email contact may be stale (2016); I did not test it.

### Gaps
- No explicit licence or terms found; not verified whether the email request is still answered or what conditions apply.
- Whether the online table's audio links serve full files without the request: not tested (the page text says full access is by email).
- MarineInfo entry unreadable.

## 6. Limitations and caveats

### Takeaway
Small, one-region, imbalanced, partly confounded by site and weather; headline metrics in the literature are saturated and likely leakage-inflated.

### Cited Findings
- [PAPER] Authors say the database needs more recordings, especially of less well represented vessel types, to support classifiers with more classes. — Paper, Sec. 4
- [PAPER] Real conditions in shallow water: high background noise, other vessels pass through the busy area, occasional marine-mammal vocalisations, wave noise on docks, clipping and ambiguous recordings were removed. — Paper, Sec. 1, Sec. 2.2
- [PAPER] Class D (liners/ro-ro) is worst with the fewest files; classes merged by vessel size, with 12 files excluded to balance. — Paper, Sec. 3, Table 3
- [PAPER] Distance is only coarse bins unless exactly known; time is 00:00:00 when unknown. — Paper, Sec. 2.2
- [WEB] Saturated benchmark, with macro F1 above 0.99 for a small CNN and an RBF-SVM; provenance problem for the derived 5 s version. — [arXiv 2609.29179](https://arxiv.org/html/2609.29179)

### Inferences
- [INFER] Single region (Galician Atlantic coast, mostly one estuary), two seasons, 2012-13, one recorder model; no evidence of generalisation to other hydrophones, ports, or deeper water.
- [INFER] Label noise is probably low for named vessels (visual ID, AIS, video) but class membership depends on a coarse merge (for example tug vs fishing vessel). Several "types" probably come from one or two vessels.
- [INFER] The 100 Hz high-pass in the recorder removes the lowest propeller and engine lines, so low-frequency (LOFAR-style) cues are partly missing. Fine for MFCC / log-mel at 16 kHz.
- [INFER] Verdict: ShipsEar is acceptable as a development dataset for a small closed-set classifier with MFCC baseline and log-mel CNN, because the data is real, recording-level grouping is feasible, and metadata is rich. It is weak as the sole "primary" dataset for a public repo: it is tiny, not clearly redistributable, and results will be high-variance. Recommended if used: report grouped (by vessel/recording) CV, frame the results as small-data, and pair it with a second dataset.

### Gaps
- No numeric label-noise estimate exists in what I found.

## 7. Related datasets and second independent source

### Takeaway
The paper itself offers no usable second dataset (it cites DOSITS only as a sound library, and SILENV/SONIC as not or incompletely public). Later literature gives DeepShip as the obvious second source.

### Cited Findings
- [PAPER] Mentioned: DOSITS (www.dosits.org, a sound reference site), SILENV (database of acoustic signatures of 171 vessels, "only recently made public"), and SONIC (incomplete, intended to publish an online database). Prior work used small private datasets (McKenna et al.: 29 freighters from an autonomous recorder in the Santa Barbara Channel; Erbe: 66 jet-ski passes; Roth et al.: icebreaker; Zak: 5 Polish Navy ships; others 6-15 vessels). — Paper, Sec. 1 and refs [8,9,5,21]
- [WEB] DeepShip: 4 classes (Cargo, Passenger, Tanker, Tug), 265 vessels, 609 readable recordings, about 47.2 h; cited as Irfan et al. 2021; the 2026 paper splits at recording level before segmentation. Licence not given in that source. — [arXiv 2609.29179](https://arxiv.org/html/2609.29179)
- [WEB] Other datasets surfaced in search results (not examined in depth): Oceanship (large-scale dataset), QiandaoEar22 (specific ship identification, hosted on IEEE DataPort). — [Oceanship](https://arxiv.org/pdf/2401.02099), [QiandaoEar22](https://arxiv.org/pdf/2406.04353), [IEEE DataPort entry](https://ieee-dataport.org/documents/qiandaoear22)

### Inferences
- [INFER] DeepShip (Strait of Georgia / Canada-recorded Ocean Networks Canada data, per my background knowledge, not verified here) has different geography, recorder, and class taxonomy (cargo, passenger, tanker, tug). Its classes only partly overlap with ShipsEar's, so the two do not give a shared label set; they would serve as an independent second source for the method, not a transfer test. Its licence and hosting must be checked by whoever researches it.

### Gaps
- Licence and hosting of DeepShip, Oceanship, QiandaoEar22 not verified here (outside this assignment's scope, flagged for the other researchers).
- SILENV/SONIC current status unchecked.
