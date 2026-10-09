# DeepShip as an open alternative to ShipsEar (reference date: 9 Oct 2026)

Evidence labels: [P] = verified by me on a primary source today; [S] = secondary paper; [I] = my inference. I could not open the Expert Systems with Applications paper itself (ScienceDirect returned 403; datalearner.com unresolvable), so paper-level facts are via secondary sources or the dataset's own repo.

## Where is it hosted, are links live, is there a request requirement?

### Takeaway
The only official host is the authors' GitHub repo, which holds only a ~63-file sample; the full dataset requires emailing the author. No Zenodo/Kaggle/HF record from the authors was found. A community-made full copy, "DeepShip2", exists on Gitee with no licence.

### Cited Findings
- [P] Official repo https://github.com/irfankamboh/DeepShip is live (HTTP 200, not archived, last push 2023-03-23, GitHub licence field = null). — [GitHub API](https://api.github.com/repos/irfankamboh/DeepShip)
- [P] README says only part of the dataset is in the repo because of GitHub size limits; "The remaining part can be downloaded by sending the email to mirfan@mail.nwpu.edu.cn". — [README.txt](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/README.txt)
- [P] The repo holds folders Cargo, Passengership, Tanker, Tug. Wav files present: Cargo 12, Passengership 20, Tanker 28, Tug 3 (63 total, ~3-26 MB each), plus one "*-metafile" per class. — [Cargo](https://api.github.com/repos/irfankamboh/DeepShip/contents/Cargo), [Passengership](https://api.github.com/repos/irfankamboh/DeepShip/contents/Passengership), [Tanker](https://api.github.com/repos/irfankamboh/DeepShip/contents/Tanker), [Tug](https://api.github.com/repos/irfankamboh/DeepShip/contents/Tug)
- [P] No Zenodo, figshare, Kaggle or Hugging Face record from the authors surfaced in web searches (absence of evidence in search, not proof). — searches run 2026-10-09
- [P] Community copy "DeepShip2": "a complete and pre-processed version" (47 h 04 min, 265 vessels, 4 classes), as DeepShip2.zip.001-.006 on Gitee; the page says no open-source licence file declared. Live (HTTP 200). — [Gitee](https://gitee.com/daiyunchen/DeepShip2)
- [S] Papers describing DeepShip as a public dataset that is downloaded from the GitHub link. — [Oceanship arXiv](https://arxiv.org/pdf/2401.02099), [UniqueShip](https://arxiv.org/abs/2609.13659)

### Inferences
- [I] "Free" access is gated by email to the author (a manual, possibly unreliable step; last repo commit 2023). A public portfolio cannot assume reproducible one-command download. The 63-file GitHub sample is directly downloadable but too small and lacks Tug (3 files).
- [I] The email address is Northwestern Polytechnical University (mail.nwpu.edu.cn); I did not test if it still answers.

### Gaps
- I did not verify the ScienceDirect article page (403) for a data-availability statement. I did not test the email route.

## Licence, redistribution, citation

### Takeaway
DeepShip has no stated licence anywhere I could find, so the default is all rights reserved by the dataset curators. The underlying audio is from Ocean Networks Canada, whose repository is mostly CC-BY 4.0. That helps but does not clearly license the DeepShip derivative. Redistribution of the audio in a public repo should be avoided.

### Cited Findings
- [P] GitHub licence field null; README contains no licence, terms, or citation text. — [GitHub API](https://api.github.com/repos/irfankamboh/DeepShip), [README](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/README.txt)
- [P] DeepShip2 on Gitee: no licence file declared. — [Gitee](https://gitee.com/daiyunchen/DeepShip2)
- [S] ONC: "Most data within the Oceans 3.0 repository are provided under the Creative Commons CC-BY 4.0 license", with some datasets restricted by partner agreements. (Fetched summary of the Frontiers paper; the exact quote came through a summariser, so verify wording before quoting.) — [Frontiers, Oceans 2.0/3.0 system](https://www.frontiersin.org/articles/10.3389/fmars.2022.806452/full)
- [S] UniqueShip, built from the same ONC hydrophones, describes the source as the "open ONC repository" and publishes its own dataset/splits at uniqueshipdata.org (live, HTTP 200; I did not read its licence). — [arXiv 2609.13659](https://arxiv.org/abs/2609.13659), [uniqueshipdata.org](https://uniqueshipdata.org)
- [P] I could not retrieve an ONC terms-of-use page (guessed ONC URLs returned 404-style pages), so the ONC licence is only via the Frontiers paper. 
- Citation to use (bibliographic details from the user's brief and the secondary papers; the DOI was not verified by me): Irfan, M. et al., "DeepShip: An underwater acoustic benchmark dataset and a separable convolution based autoencoder for classification", Expert Systems with Applications, 2021. Note: the 2026 survey [Towards Deployable Underwater Vessel Classification](https://arxiv.org/abs/2609.29179) cites it with arXiv id 2010.11929, which is actually the Vision Transformer paper; UniqueShip does the same. Do not copy that arXiv ID.

### Inferences
- [I] Because the source audio is CC-BY 4.0 (likely), a derivative work is probably permissible with attribution to ONC, but the DeepShip curators' selection/labels carry no licence; the cautious route is: do not redistribute DeepShip wavs; ship only code, split lists (file ids) and derived metrics, and ask the author in writing for permission if audio redistribution matters.
- [I] A cleaner legal alternative is UniqueShip (ONC-derived, openly published) or extracting from ONC directly with AIS, after reading its licence.

### Gaps
- No explicit ONC deployment-level licence and no DeepShip terms were found. No written author permission exists.

## Exact contents

### Takeaway
4 classes (Cargo, Passenger ship, Tanker, Tug), 265 vessels (by MMSI), 47 h 04 min, 32 kHz mono, whole variable-length recordings (seconds to ~25 min). No ambient class.

### Cited Findings
- [P] README: 47 h 04 min, 265 ships, four classes (oil tanker, tug, passenger ship, cargo ship), recorded 02 May 2016 to 04 Oct 2018 at Strait of Georgia delta node. — [README](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/README.txt)
- [S] Vessels (MMSIs) and duration per class: Passenger 46 / 12h22m; Cargo 69 / 10h40m; Tug 17 / 11h17m; Tanker 133 / 12h45m; total 265 / 47h04m; 2 km inclusion / 4 km exclusion. — [UniqueShip Table I](https://arxiv.org/pdf/2609.13659)
- [P] Metafile rows (≈ recordings) per class: Tug 70, Tanker 244, Cargo 109, Passenger 189 = 612. Search snippets of another paper cite 613 recordings, 6 s to 25 min. — metafiles: [tug](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Tug/tug-metafile), [tanker](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Tanker/tanker-metafile), [cargo](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Cargo/cargo-metafile), [passenger](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Passengership/passengership-metafile); [Oceanship](https://arxiv.org/pdf/2401.02099) for 613/32 kHz (secondary)
- [S] A 2026 paper says 609 readable recordings, ~47.2 h, and used recording-level partitioning (26,933/3,266/3,457 5-s segments train/val/test, 16 kHz), macro F1 0.7226 for a 157K-param CNN. — [Towards Deployable Underwater Vessel Classification](https://arxiv.org/abs/2609.29179) (PDF text extracted locally)
- [P] File format from the WAV header of Tug/9.wav (first 44 bytes via range request): RIFF WAVE, format tag 3 (IEEE float), 1 channel, 32000 Hz, 32 bits per sample. Files ~3-26 MB. — [Tug/9.wav](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Tug/9.wav)
- [S] Hydrophone: Ocean Sonics icListen AF, 32 kHz, ~100 m depth. — [UniqueShip](https://arxiv.org/pdf/2609.13659)

### Inferences
- [I] The 32-bit float format likely reflects ONC/icListen export, not a real 32-bit dynamic range. Only 63 files were inspected via listing; the header of one file was read.
- [I] Class sizes by vessel: Tug has only 17 vessels, Passenger 46, so vessel-grouped splits will leave tiny test folds for Tug.

### Gaps
- Bit depth of the original recording (the 32-bit float might be a conversion). Counts 609/612/613 disagree across sources; the authoritative count would need the full data.

## Grouping identifiers and leakage

### Takeaway
Each class has a metafile with ship name, date and time per recording, so grouping by vessel and date is possible. Random clip-level splits are documented as leaky; vessel-disjoint splits reduce accuracy substantially.

### Cited Findings
- [P] Metafile row format, e.g. Tug: `1,52,SEASPAN QUEEN,20171104,200959,636,1315-1922,` i.e. recording index, a second numeric field (52 in all rows seen; meaning not documented), ship name, date YYYYMMDD, time HHMMSS, a duration-like number (636), and a pair of numbers (possibly distance ranges; undocumented). Recording indices match the wav file names. Vessels repeat: in the Tug metafile SEASPAN RAVEN appears 16 times, SEASPAN COMMANDER 13, SEASPAN EAGLE 9. — [tug-metafile](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Tug/tug-metafile)
- [S] "Naively" partitioning continuous records gives falsely optimistic results. On DeepShip with Swin, vessel-disjoint (MMSI) splits give acc 0.583-0.588 (F1 0.548-0.563) vs random-file splits 0.685-0.710 (F1 0.675-0.700): a 13-14 point F1 drop. Also, UniqueShip abstract says 10-48 points across two datasets. — [UniqueShip](https://arxiv.org/abs/2609.13659), [PDF](https://arxiv.org/pdf/2609.13659)
- [S] Data are continuous hydrophone records with persistent signatures for individual vessels (the cause of leakage). — [UniqueShip](https://arxiv.org/pdf/2609.13659)
- [S] The 2026 paper notes ShipsEar lacks source-recording identifiers, so same-recording clips cannot be confirmed to stay in one partition; DeepShip does allow recording-level splitting. — [arXiv 2609.29179](https://arxiv.org/abs/2609.29179)

### Inferences
- [I] A recording-level split (as in the brief) still leaks vessels because each ship has many recordings; the right grouping key is ship name (the metafile name field, with trimming of trailing spaces, as seen in "GLENDALE "). UniqueShip's MMSI grouping is not available in the DeepShip metafiles I saw (only names).
- [I] Ship-name grouping may be imperfect (spelling variants, renamed vessels), so treat as best effort. I only checked the Tug metafile in detail.

### Gaps
- Whether the second metafile column, and the last two columns, are documented in the paper. Whether metafile names cover all 265 vessels consistently.

## Known problems

### Takeaway
Single site, single hydrophone region, many repeats per vessel, MMSI-level leakage in common splits, and unquantified label noise. No corrections or official re-releases found.

### Cited Findings
- [P] Single region: Strait of Georgia delta node (ONC), May 2016 to Oct 2018. README also says recordings include background noise, marine mammal sounds and other human activity noise (i.e. not clean ship-only). — [README](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/README.txt)
- [S] Class labelling relies on a 2 km inclusion radius (AIS-based); another ONC dataset (VTUAD) uses stricter exclusions and background filtering, whereas DeepShip has none. — [UniqueShip](https://arxiv.org/pdf/2609.13659)
- [S] Larger successor datasets from the same hydrophones exist: UniqueShip (2,460 h ship audio, 4,218 vessels, 977 h background, 11 vessel classes + background, MMSI grouping, 20 kHz), VTUAD, Oceanship. — [UniqueShip](https://arxiv.org/pdf/2609.13659)
- [S] Search snippet: "DeepShip2" is a preprocessed full version from a third party. — [Gitee](https://gitee.com/daiyunchen/DeepShip2)
- No errata, corrections or official re-release by the original authors were found (searches 2026-10-09; absence of evidence).

### Inferences
- [I] Label noise is likely moderate (AIS proximity labelling; mixed traffic within 2 km, and vessel-type misassignment), but I found no paper quantifying it.
- [I] Because every recording comes from one location and hydrophone, the model may learn site/season/time cues; the generalisation claims should be limited.

### Gaps
- No quantitative label-noise study found. No check of the original paper for exclusion rules.

## Ambient / background class

### Takeaway
DeepShip has no ambient class. A legitimate open source of ambient audio from the same hydrophones exists (UniqueShip background, VTUAD background) and ONC itself.

### Cited Findings
- [S] DeepShip's Table I entry has no background column ("—"); VTUAD adds background as a fifth class; UniqueShip includes 977 h background (no ship within 8 km, grouped by day). — [UniqueShip](https://arxiv.org/pdf/2609.13659)
- [S] ONC data are mostly CC-BY 4.0. — [Frontiers](https://www.frontiersin.org/articles/10.3389/fmars.2022.806452/full)

### Inferences
- [I] Ambient clips from the same deployment (via UniqueShip or ONC Oceans 3.0 with AIS filter) would match the recording channel and avoid a trivial domain shift; ambient from another source would let a model detect "noise" by channel differences alone.
- [I] Licence of the UniqueShip background was not verified.

### Gaps
- Licence of VTUAD and UniqueShip subsets was not checked.

## Mapping to ShipsEar classes A-E

### Takeaway
Only a loose and partial mapping: DeepShip is by commercial vessel type (cargo, tanker, passenger, tug) while ShipsEar groups small coastal craft. Do not claim it is a drop-in replacement.

### Cited Findings
- [S] DeepShip classes: Cargo, Passenger ship, Tanker, Tug. — [UniqueShip](https://arxiv.org/pdf/2609.13659), [README](https://raw.githubusercontent.com/irfankamboh/DeepShip/main/README.txt)
- [S] ShipsEar: ~3.1 h, 90 recordings, 11 vessel types, commonly grouped into five classes, 64 kHz resampled to 16 kHz. — [arXiv 2609.29179](https://arxiv.org/abs/2609.29179). The A-E definitions are from the user's brief, not verified here.

### Inferences
- [I] Mapping: Tug maps loosely to ShipsEar A (tug); Passenger maps to C (passenger ferry), but DeepShip passengers are largely large ferries/cruise vessels in a ferry corridor; Cargo and Tanker are closest to D (ocean liner / ro-ro) as large merchant ships, although not equivalent; B (motorboat/pilot/sailboat) has no counterpart; E (noise) needs the separate ambient source above.
- [I] Better framing for the project: DeepShip is a separate 4-class benchmark (+ optional ambient class), not a re-creation of A-E.

### Gaps
- No published class mapping between the two datasets was found.
