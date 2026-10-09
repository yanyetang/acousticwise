# Oceanship and QiandaoEar22 as open ShipsEar alternatives

Method note: facts below come from pages I fetched on 2026-10-09 (summarised by the fetch tool, not read raw, except the arXiv PDFs which I converted to text locally). "Verified" means a primary source (paper, official repo, DataPort page) said it. Live link checks were page fetches only; I could not open Baidu Pan or the OneDrive link, and downloaded no data.

## 1. Hosting, access and link status

### Takeaway
Neither dataset is a clean "free, open, reliably hosted" choice. Oceanship is distributed through Baidu Pan links on a GitHub repo with no licence. QiandaoEar22 on IEEE DataPort is subscriber-gated, with an unlicensed OneDrive mirror on GitHub.

### Cited Findings
- Oceanship's official repo is github.com/lizeyujack/oceanship. It was live on 2026-10-09, last pushed 2024-06-11, not archived, 61 stars and 9 open issues. — [GitHub](https://github.com/lizeyujack/oceanship), [GitHub API](https://api.github.com/repos/lizeyujack/oceanship)
- The README gives only Baidu Pan links (with passwords) for the fine-grained (FG) and coarse-grained (CG) versions. "Full" is not uploaded and is obtained as CG + FG. Files are 10 split tar parts (.aa-.aj) holding WAVs. — [README via GitHub API](https://api.github.com/repos/lizeyujack/oceanship/readme)
- I could not test whether the Baidu links are still alive, and Baidu Pan commonly needs an account and its client for large files. This is unverified.
- The Oceanship README also lets users re-crawl audio and AIS themselves from Ocean Networks Canada (ONC) with their own ONC token, using a `jack.sh` script. The paper says the author obtained ONC access via a token. — [README](https://api.github.com/repos/lizeyujack/oceanship/readme), [arXiv PDF](https://arxiv.org/pdf/2401.02099)
- Oceanship paper: arXiv 2401.02099, v3 11 Jun 2024, accepted at ICIC 2024 (Springer chapter, DOI 10.1007/978-981-97-5591-2_40 per OpenAlex). Authors led by Zeyu Li (SJTU). — [arXiv abs](https://arxiv.org/abs/2401.02099), [OpenAlex](https://api.openalex.org/works/doi:10.1007%2F978-981-97-5591-2_40)
- QiandaoEar22 DataPort page: https://ieee-dataport.org/documents/qiandaoear22, DOI 10.2206/QDE22, "Feng HONG, Xiaoyang DU, QiandaoEar22, IEEE Dataport, April 26, 2024". The page shows "LOGIN TO ACCESS DATASET FILES" and was classed subscriber-only (not under the /open-access/ path). The /open-access/qiandaoear22 URL returned 404. — [DataPort](https://ieee-dataport.org/documents/qiandaoear22)
- The DataPort file list is 180+ compressed WAV archives, about 18 GB, in groups: Kaiyuan target/other, Speedboat target/other, UUV target/other and Noise. — [DataPort](https://ieee-dataport.org/documents/qiandaoear22)
- A second copy is advertised on GitHub (xiaoyangdu22/QiandaoEar22). It holds folders KaiYuan, SpeedBoat, UUV and noise_target plus a README. The data sits on a Microsoft SharePoint/OneDrive link because of GitHub size limits. The repo has 6 commits and no licence. I did not open the OneDrive link, so liveness is unverified. — [GitHub](https://github.com/xiaoyangdu22/QiandaoEar22); the repo is cited as the data location in [arXiv 2406.04354 HTML](https://arxiv.org/html/2406.04354v1)
- Papers: arXiv 2406.04353 (short dataset paper, a conference-style 4-page paper) and arXiv 2406.04354 (longer; submitted 15 May 2024; authors Xiaoyang Du and Feng Hong), which is the EURASIP J. Adv. Signal Process. article s13634-024-01181-9 (Nov 2024). The EURASIP link matched on search, but I could not read the Springer page (redirect to a login/cookie gate). — [2406.04353](https://arxiv.org/abs/2406.04353), [2406.04354](https://arxiv.org/abs/2406.04354), [DOAJ](https://doaj.org/article/cdadf279ab554227849724acdbcd3d09)

### Inferences
- A public portfolio project cannot tell users "download X" for QiandaoEar22 through DataPort unless they hold a subscription. The GitHub/OneDrive route is the only free path found and is unverified.
- Baidu Pan hosting is a poor fit for reproducibility. The ONC re-crawl route is more robust but needs an ONC account/token.

### Gaps
- Whether the Baidu and OneDrive links work in Oct 2026.
- Whether DataPort lets a free account download QiandaoEar22 (fetch only showed the login gate and the subscriber classification).

## 2. Licence, redistribution, citation

### Takeaway
Neither dataset states a licence on its own distribution. Oceanship derives from ONC data, which is generally reported as CC BY, but I could not confirm the exact licence text. Redistribution rights for both are unclear, so do not redistribute audio in the portfolio repo.

### Cited Findings
- Oceanship repo: no licence file and no licence in the GitHub API metadata. — [GitHub API](https://api.github.com/repos/lizeyujack/oceanship)
- Oceanship audio comes from ONC Oceans 3.0 (dataset DOI 10.21383/650d90face87-473c-b932-278519062ab5), and the README thanks ONC. A search result indicates ONC data is available under CC BY, but this is a secondary snippet. Oceanship's "Full/CG/FG" clips are derived works, and the AIS enrichment was scraped from shipfinder.com (a third party, terms unknown). — [Oceanship PDF](https://arxiv.org/pdf/2401.02099), [README](https://api.github.com/repos/lizeyujack/oceanship/readme), [search summary](https://data.oceannetworks.ca/DatasetLandingPage?doi-dataset=10.21383/650d90face87-473c-b932-278519062ab5)
- QiandaoEar22: the DataPort page states no licence. The GitHub repo has no licence. The "arXiv.org perpetual non-exclusive license" in the fetch summaries applies to the papers, not the audio. — [DataPort](https://ieee-dataport.org/documents/qiandaoear22), [GitHub](https://github.com/xiaoyangdu22/QiandaoEar22)
- Citation for QiandaoEar22: Hong and Du, DOI 10.2206/QDE22, plus the arXiv paper (2406.04353 or 2406.04354). — [DataPort](https://ieee-dataport.org/documents/qiandaoear22)
- Citation for Oceanship: Li et al., arXiv 2401.02099 / ICIC 2024. — [arXiv](https://arxiv.org/abs/2401.02099)

### Inferences
- Without a licence, default copyright applies to both. Using them privately for training and publishing results is low risk, but committing audio or derived clips to a public repo is not clearly permitted.
- Oceanship's ONC upstream may be CC BY (unverified; I did not read ONC's own terms page, which 404ed at the URL I tried).

### Gaps
- Official ONC licence text; whether the Oceanship authors have authority to redistribute ONC-derived clips; any written permission from Hong/Du.

## 3. Contents

### Takeaway
Oceanship is large, clip-based and AIS-labelled with 15 classes. QiandaoEar22 is small, single-lake, and built for identifying three specific vessels, not vessel types.

### Cited Findings
Oceanship
- 15 categories; 121.17 h in the "Full" set (107,540 audio-text pairs). CG has 53,769 clips (61.96 h) and FG has 53,771 clips (59.21 h; the paper's text and table disagree by two clips on which has 53,771). Only the FG half carries heading and speed. — [arXiv PDF](https://arxiv.org/pdf/2401.02099)
- The README says "65 hours", inconsistent with the paper's 121 h (likely because each repo version covers only part). — [README](https://api.github.com/repos/lizeyujack/oceanship/readme)
- Pre-cut clips in WAV, no whole recordings. Model input in the paper is 16 kHz audio (a model setting, not necessarily the stored rate). Stored sampling rate, bit depth, channels and clip length were not stated in anything I read. — [arXiv PDF](https://arxiv.org/pdf/2401.02099)
- Largest classes are Cargo and Tug; smallest are "Diving ship" and "Spare". Data span 15 July 2020 to 18 Feb 2021 per Sec 2.1 (the abstract/README say 2021-2022, an internal inconsistency). — [arXiv PDF](https://arxiv.org/pdf/2401.02099)
- Labels come from decoded AIS messages (position, sog, cog, heading, timestamp, MMSI) matched to audio by timestamp. MMSI is masked in the paper's example. The paper's text query set of 26 names includes Fishing, Motorboat, Port Tender, Spare, Trawler, Diving ship, Dredging, Towing, Search and Rescue, Cargo, Pilot Vessel, Tanker, Pleasure Craft, Passenger, RORO, Sailboat, Military ship, Tug, Ocean liner, Mussel boat, Law Enforcement, Anti-pollution equipment, Medical Transport, Natural ambient noise, Sailing. This is a query vocabulary built to match ShipsEar names, not the 15 dataset classes. I did not find a definitive list of the 15 dataset classes. — [arXiv PDF](https://arxiv.org/pdf/2401.02099)
- Distinct vessel count: not stated. Hours per class: only in a figure, not extracted.

QiandaoEar22
- Ship-radiated noise 9 h 28 min plus background 21 h 58 min (about 31 h). 10,611 ship records and about 25,900 background records, all 3 s WAV segments, cut from 143 recordings. — [arXiv 2406.04353 PDF](https://arxiv.org/pdf/2406.04353), [arXiv 2406.04354 HTML](https://arxiv.org/html/2406.04354v1)
- 20 "categories of targets" in the short paper, including large tour boats, cargo ships, sanitation vessels, small boats, research ships, plus helicopters and cars whose airborne noise reaches the hydrophone. — [2406.04353 PDF](https://arxiv.org/pdf/2406.04353)
- Spec: 52,734 Hz, 16 bit, single channel, 1-25.8 kHz band, WAV (from the fetch summary of the longer paper; I did not see the sentence directly). — [2406.04354 HTML](https://arxiv.org/html/2406.04354v1)
- Benchmark tasks use three named targets only: SpeedBoat, KaiYuan (tour boat) and UUV. Train samples quoted: 4,848 speedboat, 1,363 KaiYuan, 631 UUV. The paper's 2406.04354 also does ship-vs-background detection. — [2406.04354 HTML](https://arxiv.org/html/2406.04354v1)
- Labels: "auditory sensation (strong/middle/weak), distance, and type". The DataPort file groups are "target" vs "other" per ship (Kaiyuan, Speedboat, UUV) plus Noise. — [2406.04353 PDF](https://arxiv.org/pdf/2406.04353), [DataPort](https://ieee-dataport.org/documents/qiandaoear22)
- A background/noise class exists (night and idle periods, 17 Noise archives). — [2406.04353 PDF](https://arxiv.org/pdf/2406.04353), [DataPort](https://ieee-dataport.org/documents/qiandaoear22)

### Inferences
- QiandaoEar22 has specific-ship targets (one-vs-rest), not a vessel-type taxonomy. Multi-target mixtures are allowed in a segment, so single-label classification is awkward.
- Oceanship classes follow AIS ship-type naming, so they are closer to ShipsEar-style types than QiandaoEar22 is.

### Gaps
- The exact 15 Oceanship class names and per-class hours (figure only); stored sampling rate/clip length; number of distinct vessels in either dataset; the number of distinct ships behind QiandaoEar22's 20 categories.

## 4. Leakage-safe grouping identifiers

### Takeaway
Oceanship has timestamps and (masked/enriched) MMSI information allowing grouped splits in principle, but recording-level IDs are not documented. QiandaoEar22 has only 143 parent recordings, with no documented per-segment recording ID.

### Cited Findings
- Oceanship: clips carry timestamps, location and AIS MMSI; the README ships .txt enhanced files with time, MMSI, callsign, IMO, heading, course, speed, ship type, position, dimensions. FG version only has heading/speed. — [README](https://api.github.com/repos/lizeyujack/oceanship/readme), [arXiv PDF](https://arxiv.org/pdf/2401.02099)
- QiandaoEar22: 10,611 3 s segments from 143 recordings; paper splits 70/15/15 by task, with no stated recording-level grouping. — [2406.04354 HTML](https://arxiv.org/html/2406.04354v1)
- UniqueShip (arXiv 2609.13659) reports that random partitioning in two prominent UATR datasets inflates test accuracy by 10-48 points versus careful partitioning. It is built from ONC with 4,218 unique vessels. The abstract does not name the two datasets (fetch summary said so); reading the full paper for whether it covers Oceanship or QiandaoEar22 is still to do. — [arXiv](https://arxiv.org/abs/2609.13659)

### Inferences
- For Oceanship, group by MMSI (where unmasked) or by contiguous timestamp blocks; many clips from one vessel pass will be near-duplicates, so naive clip-level splits will leak.
- For QiandaoEar22, filenames inside the archives may encode the recording; unverified because the archives are not downloaded.

### Gaps
- Actual filename conventions in both datasets; whether MMSI is present in released Oceanship files or only the masked paper example.

## 5. Recording setup and independence from ShipsEar

### Takeaway
Both are independent of ShipsEar (different sites, hardware, regions).

### Cited Findings
- ShipsEar: Spanish Atlantic coast (NW Spain), autumn 2012 and summer 2013, 90 recordings, 11 vessel types, about 3 h. — [Oceanship PDF describing ShipsEar](https://arxiv.org/pdf/2401.02099)
- Oceanship: ONC cabled hydrophone observatory audio (the paper's coordinates in the AIS example are about 123.45 W, 48.77 N, i.e. off British Columbia, Canada, the Strait of Georgia area). Hardware not stated in what I read. DeepShip is also ONC-derived, so Oceanship and DeepShip share a source and may overlap. — [Oceanship PDF](https://arxiv.org/pdf/2401.02099)
- QiandaoEar22: Qiandao Lake (freshwater), Zhejiang, 24-28 June 2022, at the Xin'anjiang Experimental Site of IOA, CAS; DigitalHyd SR-1 hydrophone anchored via counterweight and buoy, water depth 30-50 m, hydrophone at 10-15 m, near a channel. Tourist season, so mostly sightseeing boats and speedboats. — [2406.04353 PDF](https://arxiv.org/pdf/2406.04353), [2406.04354 HTML](https://arxiv.org/html/2406.04354v1)

### Inferences
- A lake near-shore setup is a large domain shift from ShipsEar's harbour/coastal sea. Good for cross-dataset generalisation tests, poor as a drop-in ShipsEar replacement.

### Gaps
- Oceanship hydrophone model/depth/sampling rate.

## 6. Known limitations and critiques

### Takeaway
No independent critiques of either dataset were found. Limitations come from the authors' papers and my reading of them.

### Cited Findings
- Oceanship: heavy class imbalance (Cargo, Tug large; Diving ship, Spare tiny); labels depend on AIS-to-timestamp matching, with some vessel types indeterminate when MMSI lookup fails; the paper's own text has duration/date inconsistencies (121 h vs 65 h; 2020-21 vs 2021-22). — [arXiv PDF](https://arxiv.org/pdf/2401.02099), [README](https://api.github.com/repos/lizeyujack/oceanship/readme)
- Oceanship authors state ZSL and retrieval baselines only (Oceannet R@1 24.38% on ShipsEar), with no supervised closed-set benchmark. — [arXiv PDF](https://arxiv.org/pdf/2401.02099)
- QiandaoEar22: one lake, five days, tourist season; mixed multi-target segments; a subjective "auditory sensation" label; benchmark splits not documented as recording-level. — [2406.04353 PDF](https://arxiv.org/pdf/2406.04353), [2406.04354 HTML](https://arxiv.org/html/2406.04354v1)
- Related leakage warning for the field: [UniqueShip](https://arxiv.org/abs/2609.13659).

### Gaps
- Third-party re-use or critique papers for either dataset (web searches returned none).

## 7. Mapping to ShipsEar classes A-E

### Takeaway
Oceanship can plausibly map to A-D (plus E only if ambient clips exist); QiandaoEar22 can map only to a handful of coarse classes plus E and is a poor fit.

### Cited Findings
- ShipsEar's 5-class grouping is described in the Oceanship paper; the Oceanship text queries explicitly include ShipsEar names (Fishing, Trawler, Mussel boat, Tug, Dredging, Motorboat, Pilot Vessel, Sailboat, Passenger, Ocean liner, RORO, Natural ambient noise). — [arXiv PDF](https://arxiv.org/pdf/2401.02099)
- Oceanship states that it covers "approximately 50%" of ShipsEar's categories and all of DeepShip's (cargo, tanker, tug, passenger). — [arXiv PDF](https://arxiv.org/pdf/2401.02099)

### Inferences (proposal, unverified because the 15 class names are not confirmed)
- Oceanship (AIS ship types): A = Fishing, Tug, Dredging, Towing, Port Tender (if present); B = Pleasure Craft, Sailing, Pilot (if present); C = Passenger (large coastal ferries will be mixed with others, so check AIS length); D = Cargo, Tanker are the nearest to "large ships" but ShipsEar D (ocean liner, ro-ro) is not equivalent, so treat as a different class, not a mapping; E = no ambient class identified, would need AIS-free background clips from ONC.
- QiandaoEar22: speedboat ~ B; KaiYuan tour boat ~ C (passenger) at best; UUV and cargo/sanitation/research have no ShipsEar equivalent; Noise ~ E. No class A or D equivalent.
- Taxonomy mismatch means any "mapped" model is a cross-domain proxy, not like-for-like with ShipsEar's 5 classes.

### Gaps
- Final Oceanship class list (needed to firm up the mapping); recordings per class for either dataset.

## Recommendation summary (for the report writer)
- Neither dataset should be redistributed in a public repo. Oceanship (AIS-labelled, ONC-derived, unlicensed repo, Baidu hosting) is the more useful for a type classifier but needs verification of the class list, IDs and licence. QiandaoEar22 is subscriber-gated on DataPort, has no vessel-type taxonomy, and is best used only as an out-of-domain background/speedboat/tour-boat check.
- A more defensible alternative surfaced during research: UniqueShip (ONC-based, with leakage-safe splits, uniqueshipdata.org), not examined here — [arXiv 2609.13659](https://arxiv.org/abs/2609.13659).
