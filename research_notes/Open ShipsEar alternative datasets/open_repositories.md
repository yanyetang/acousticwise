# Open ShipsEar-alternative vessel-noise datasets (repository search, reference date Oct 2026)

Scope: excludes ShipsEar, DeepShip, Oceanship, QiandaoEar22, Wolfset. About 15 tool calls were used. The search was NOT exhaustive: Kaggle, Hugging Face, Mendeley Data, Dryad, OSF and figshare returned no hits via the search tool. I could not browse their on-site search, so absence of hits is not proof of absence.

## Which open datasets exist and are verified (name, URL, host, licence, access, size, classes, sampling rate, clips vs whole files, identifiers, site)?

### Takeaway
Only two datasets passed all gates (real field audio, vessel-type labels, open licence, no login): UniqueShip (large, ONC Strait of Georgia, vessel-disjoint splits) and the VLIZ "AIS-annotated Hydrophone Recordings" set (Belgian North Sea, 10 s clips). DeuteroNoise is promising but its audio hosting and licence were not verifiable on a host record. VTUAD is behind an IEEE DataPort subscription. The survey landscape found no Zenodo/Kaggle/HF vessel-type set of ShipsEar's kind.

### Cited Findings

**A. UniqueShip (verified, with caveats)**
- Source: ONC hydrophones 2556 and 2523, Strait of Georgia / Fraser River Delta, May 2016 to Nov 2023, Ocean Sonics icListen AF at about 100 m depth, 32 kHz native. Labels come from AIS. — [arXiv 2609.13659](https://arxiv.org/pdf/2609.13659)
- Size: 2,460 h of ship audio from 4,218 unique vessels (MMSI), 977 h of background (3,437 h in total); 11 vessel classes + 1 background; 2,779 days; about 2.5 M 5-s samples; clips resampled to 20 kHz. — [arXiv 2609.13659](https://arxiv.org/pdf/2609.13659)
- All splits are vessel-disjoint (ships appear in only one partition). The paper reports that this exposes a large accuracy drop on DeepShip and VTUAD because of leakage. Published splits are 5-s segments; full-length recordings are "available through the codebase". — [arXiv 2609.13659](https://arxiv.org/pdf/2609.13659), [uniqueshipdata.org](https://uniqueshipdata.org)
- Host: uniqueshipdata.org, with Google Drive download links in 10 GB parts (rclone recommended). Licence stated as CC-BY 4.0 (owned by Ocean Networks Canada). No login stated. Variants: 12-class all data 656 GB (.wv) or 1,400 GB (.wav); 5-class balanced 92 GB unzipped; 12-class 5 h per class 26 GB unzipped. 5-class balanced classes: Tug/Tow, Tanker, Passenger, Cargo, Background. — [uniqueshipdata.org](https://uniqueshipdata.org) (fetched summary)
- Caveats: the host is a project website plus Google Drive, not a DOI repository (less stable than Zenodo). The licence statement was read via a page-summarising fetch, not directly on a licence file. The ONC source data have DOIs (e.g. [10.34943/6fc16205-8e10-498a-96fb-869720d03ab3](https://doi.org/10.34943/6fc16205-8e10-498a-96fb-869720d03ab3)). The paper is an October-2026-era preprint (arXiv ID 2609.*). I did not check it for peer review.

**B. AIS-annotated Hydrophone Recordings for Vessel Classification (VLIZ, verified)**
- Host: Marine Data Archive (VLIZ), DOI [10.14284/723](https://doi.org/10.14284/723), record [marineinfo.org/id/dataset/8773](https://marineinfo.org/doc/dataset/8773?doiid=1154). Licence CC BY 4.0, open access, no login or request stated.
- Content: 27,524 labelled 10-second clips, train/val/test subsets, two hydrophone stations (Gardencity, Grafton) in the Belgian part of the North Sea, 116 days, Jan to Nov 2022. Labels: AIS-derived vessel type, activity, position, speed, distance classes. Sampling rate and total file size are not stated in the record. — same record.
- Pre-cut clips only. Whether MMSI is exposed was not verified; a split by day or station would still be possible if timestamps are present (unverified).

**C. DeuteroNoise (partly verified, not yet safely usable)**
- Open corpus of calibrated 48 kHz / 24-bit hydrophone audio plus AIS, three basins (Catalan coast, Venice Lagoon, western Black Sea / Constanta), 6 campaigns Dec 2023 to Oct 2024; 689.70 h of continuous audio, more than 300 GB raw; about 11.73 h (about 12 h) of labelled single-vessel passages with MMSI, vessel type and distance; six operational categories (leisure, fisheries, commercial cargo, service/workboats, passenger, etc.). Paper is open access (CC BY 4.0). — [Nou-Plana et al., Applied Acoustics 246 (2026) 111239](https://boa.unimib.it/retrieve/d5750c52-16e3-44ac-a933-2316c2997ca8/Nou-Plana%20et%20al-2026-Applied%20Acoustics-VoR.pdf)
- The paper's Table 2 says the dataset has "full open access ... under permissive CC-BY-4.0 / MIT licensing", but the same paper says the WAV files are on RAID storage at La Salle-URL, served by a PostgreSQL/FastAPI web interface. The only public repo I could find is a "preview" with sample data and no licence, stating the full site will be deuteronoise.eu ("coming soon"). — [GitHub preview](https://github.com/Ignasinou/deuteronoise-dataset-preview), [deuteronoise.eu](https://deuteronoise.eu) (no download info found)
- Verdict: NOT verified as an open, stable download in Oct 2026. Re-check before relying on it. Only about 12 h labelled, so it is small even if available.

**D. Underwater noise signatures, Australian waters (IMAS / NESP) — does not qualify**
- CC BY 4.0, open, hosted by IMAS (University of Tasmania). Contents are source spectra and source levels (zipped CSV and XLSX) for mainly cargo ships and some tankers at Perth Canyon and Portland. No raw audio and no vessel-type variety. — [Research Data Australia](https://researchdata.edu.au/underwater-noise-signatures-australian-waters/1712034)

### Inferences
- UniqueShip and the VLIZ set are the only candidates meeting all four gates. Both rely on AIS-derived labels, which are less curated than ShipsEar's hand labels.
- UniqueShip shares its ONC source with DeepShip, VTUAD and Oceanship. It is therefore not a fully independent test source relative to DeepShip or Oceanship, but its vessel-disjoint design makes it a better primary dataset in rigor. The VLIZ set is geographically independent of all ONC-derived sets.

### Gaps
- Sampling rate, file size, whether MMSI is exposed, and the exact vessel-type class list for the VLIZ set were not found; I did not open the data files.
- UniqueShip's 12 class names and per-class counts were not extracted (Fig. 2 of the paper has them). Whether the Google Drive links currently work was not tested (no download attempted).
- Hugging Face, Kaggle, Mendeley Data, Dryad, OSF, figshare (beyond Wolfset) and Papers With Code produced no ship-noise vessel-type dataset in my searches. Not exhaustively browsed.
- SWellEx-96, Salish Sea / ECHO and similar were not individually checked. They are generally source-localisation or vessel-noise-level sets without vessel-type classes, but this is from background knowledge and unverified.

## Candidates discarded (not truly open, or not suitable)

### Takeaway
VTUAD is not freely accessible (subscription), and the DeuteroNoise status is unconfirmed. Others do not provide vessel-type audio.

### Cited Findings
- VTUAD (Vessel Type Underwater Acoustic Data): IEEE DataPort, DOI 10.21227/msg0-ag12, about 13.5 GB, 5 classes (Cargo, Tanker, Tug, Passengership, Background), 3 distance scenarios, WAV+CSV. Page states it "requires an IEEE DataPort Subscription" (Standard Dataset), licence not shown. DISCARDED: login or subscription required, not open. — [IEEE DataPort](https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data). The authors' code at github.com/lucascesarfd/onc_dataset (cited in [arXiv 2609.13659](https://arxiv.org/pdf/2609.13659)) reportedly rebuilds such sets from ONC (not verified here).
- "Processed DeepShip and ShipsEar dataset" on IEEE DataPort: derivative of excluded datasets; not assessed. — [IEEE DataPort](https://ieee-dataport.org/documents/processed-deepships-dataset-and-shipsear-dataset)
- A comparison table in the DeuteroNoise paper lists ShipsEar's licence as "CC-BY-NC via project website". That conflicts with a public-repo use of ShipsEar, a relevant caution for the main project but out of scope here. — [Nou-Plana et al. 2026](https://boa.unimib.it/retrieve/d5750c52-16e3-44ac-a933-2316c2997ca8/Nou-Plana%20et%20al-2026-Applied%20Acoustics-VoR.pdf)
- Preprints.org 202609.0617 (appeared in survey-style searches): fetch returned HTTP 403, content not read. — [link](https://www.preprints.org/manuscript/202609.0617)
- dau.url.edu record for DeuteroNoise: access denied to my fetch (CSUC protection); no metadata read. — [link](https://dau.url.edu/handle/20.500.14342/5963)

### Inferences
- The ONC archive itself (CC-BY via DOI) plus AIS is the common upstream source for DeepShip, VTUAD, Oceanship and UniqueShip.

### Gaps
- Licence of VTUAD not visible on its page.

## Ranking for the project and roles

### Takeaway
1) UniqueShip is the best candidate to replace ShipsEar as the primary dataset, but it is much larger and uses AIS labels. 2) The VLIZ North Sea set is the best independent second-source test (different site, instrument and country), with only coarse labels and 10-s clips. 3) DeuteroNoise is a conditional third option if its release is confirmed.

### Cited Findings
- Rank 1, UniqueShip: 12 vessel classes + background, 4,218 MMSIs, vessel-disjoint splits, CC-BY 4.0 per its site. Fits "recording-level splits" better than any other candidate, since leakage control is built in. A 26 GB or 92 GB subset is manageable for a portfolio project. — [uniqueshipdata.org](https://uniqueshipdata.org), [arXiv](https://arxiv.org/pdf/2609.13659)
- Rank 2, VLIZ AIS-annotated hydrophone set: CC BY 4.0, DOI-backed, 27,524 clips of 10 s, Belgian North Sea. — [Marine Data Archive record](https://marineinfo.org/doc/dataset/8773?doiid=1154)
- Rank 3, DeuteroNoise: Mediterranean/Black Sea 48 kHz, about 12 h labelled, availability unconfirmed. — [paper](https://boa.unimib.it/retrieve/d5750c52-16e3-44ac-a933-2316c2997ca8/Nou-Plana%20et%20al-2026-Applied%20Acoustics-VoR.pdf)

### Inferences
- Class taxonomies differ (UniqueShip: cargo, tanker, tug, passenger style; VLIZ and DeuteroNoise: AIS type or activity categories), so a cross-source test must map to a common coarse label set (e.g. cargo / tanker / tug / passenger or "other"), and the sampling rates differ (20 kHz clips for UniqueShip, unknown for VLIZ, 48 kHz for DeuteroNoise).
- None of these is a drop-in ShipsEar substitute (ShipsEar: 11 hand-labelled types, near-shore, 52.7 kHz). The nearest in spirit is DeuteroNoise (near-port, mixed leisure and workboats) if it is released openly.

### Gaps
- Licence and stability of UniqueShip's Google Drive hosting should be re-confirmed (a Zenodo mirror would be better). Verify the VLIZ data file format and sampling rate by downloading one file (a few MB).
