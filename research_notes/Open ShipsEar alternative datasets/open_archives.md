# Open hydrophone archives for vessel-type-labelled ship noise (as of Oct 2026)

Method note: about 20 tool calls (search plus fetch of landing pages and abstracts). Nothing was downloaded. Several primary pages blocked fetching (ONC data portal licence page 404, marineinfo.org 406, preprints.org 403, arXiv PDF too large). Those claims rest on search snippets and are marked as such.

## 1. Ocean Networks Canada (Strait of Georgia, ONC): licence, access, and labelled/AIS-linked datasets

### Takeaway
ONC is the best source. Ready-made, AIS-labelled, openly licensed vessel datasets already exist on ONC Strait of Georgia data (UniqueShip, 2,460 h, CC-BY 4.0, direct download). ONC's own licence wording could not be confirmed on ONC pages. The licence rests on the dataset authors' statement that ONC owns the data and releases it as CC-BY 4.0.

### Cited Findings
- UniqueShip: 2,460 h of ship-radiated noise from 4,218 unique vessels, ONC Strait of Georgia hydrophones, May 2016 to Nov 2023, paired with AIS. Data is "owned by ONC and available under CC-BY 4.0". Classes are 5-class balanced (Tug/Tow, Tanker, Passenger Ship, Cargo, Background) and a 12-class full set. The audio is 20 kHz, in .wv (compressed) and .wav. Each 5 s clip carries 17 AIS fields (MMSI, distance, speed, course, length, beam, draught, navigation status and more). Vessel-disjoint splits are provided. The 5-class subset is 92 GB unzipped, the 12-class 5 h subset is 26 GB, and the full set is 656 GB (.wv) or 1.4 TB (.wav). Download is by Google Drive links in 10 GB chunks (rclone recommended) — [uniqueshipdata.org](https://uniqueshipdata.org) (fetched page); paper: [UniqueShip, arXiv 2609.13659](https://arxiv.org/abs/2609.13659)
- The UniqueShip paper reports that random partitioning in two prominent UATR datasets inflates test accuracy by 10 to 48 percentage points, so vessel-disjoint splitting matters — [arXiv 2609.13659](https://arxiv.org/abs/2609.13659). The abstract does not name the two datasets (I did not confirm they are DeepShip or ShipsEar).
- DeepShip (cargo, passenger, tanker, tug; 47.07 h; 265 vessels) is described as ONC icListen recordings from the Strait of Georgia delta node, 2 May 2016 to 4 Oct 2018, 32 kHz hydrophone sample rate — [AquaSignal, arXiv 2505.14285](https://arxiv.org/pdf/2505.14285) and the search snippets. I did not find a licence statement for DeepShip itself: unverified. I did not find a licence on DeepShip's own distribution page either. Since it derives from ONC data, UniqueShip is the safer, explicitly licensed alternative.
- A "VTUAD" is described in a 2026 preprint as 525 single-vessel clips (about 56 h, 32 kHz mono WAV) in 12 categories (11 AIS vessel types plus background), all labels generated automatically by aligning AIS with ONC icListen audio. The preprint says the ONC-owned data are CC-BY 4.0 and that the WAV clips are redistributable. Source is the search snippet of [preprints.org 202609.0617](https://www.preprints.org/manuscript/202609.0617). The page returned 403 to me, so this is unverified at the primary page.
- A separate, older "VTUAD: Vessel Type Underwater Acoustic Data" (created 8 Sep 2022, DOI 10.21227/msg0-ag12) is on IEEE DataPort. It has 5 classes (Cargo, Tanker, Tug, Passenger ship, Background), three distance scenarios (2-4, 3-5, 4-6 km), WAV plus CSV, and CTD metadata. It is a subscriber-only "Standard Dataset", with no licence stated on the page — [IEEE DataPort](https://ieee-dataport.org/documents/vtuad-vessel-type-underwater-acoustic-data). This is a poor fit for an openly licensed project.
- A 2026 preprint describes an AIS-aligned labelling workflow on ONC Strait of Georgia data. It uses an indexed spatiotemporal join of hydrophone windows to AIS and separates no-contact, single-contact and two-contact windows (about 9.5e5 windows) — [arXiv 2607.13840](https://arxiv.org/abs/2607.13840). Only the abstract was read.
- ONC says its hydrophone audio can be downloaded as wav/mp3/flac-type files via an Oceans 3.0 account and API. The search summary says audio downloads come as mp3, wav, png or mat. The Python `onc` client wraps the Oceans 3.0 API; registration is at https://data.oceannetworks.ca/Registration (search results, not checked on the portal).
- ONC's Legal Notices page restricts reuse of the website and its content without written permission. It does not state a data licence and points to separate terms of use — [ONC Legal Notices](https://www.oceannetworks.ca/legal-notices/). Treat the licence on raw Oceans 3.0 downloads as unverified on ONC's own pages. If raw data is pulled directly, check the Oceans 3.0 terms of use and cite ONC.

### Inferences
- Realistic in three weeks: download the 5-class balanced UniqueShip subset (92 GB; fewer hours is possible via the 26 GB 12-class subset) and use its vessel-disjoint splits. No labelling work is needed.
- Mapping to the ShipsEar scheme is only partial. UniqueShip's cargo, tanker, passenger and tug classes do not map onto ShipsEar A-E. Tug maps to A, passenger to C, and cargo/tanker are closest to D, but only loosely. Motorboat/sailboat (B) is missing from the 5-class set. The 12-class set may add more categories, but its class list was not retrieved (gap).
- Building new labels from raw ONC plus AIS is a heavier job (AIS retrieval, spatiotemporal join, closest-point-of-approach filtering). It is plausible only if the project needs classes UniqueShip lacks.

### Gaps
- The exact ONC data licence text on an ONC-hosted page was not located (portal page 404; Legal Notices silent on data).
- The full class list of the 12-class UniqueShip set and the licence of DeepShip itself were not retrieved.
- Whether the 2026 VTUAD preprint and the 2022 IEEE DataPort VTUAD share a name only (they appear to differ in size and classes) was not resolved.
- Vancouver Port ECHO program data (Burrard Inlet, Haro Strait) with labelled passages was not found in this pass.

## 2. NOAA NCEI passive acoustic archive and NOAA/NPS sanctuary products (SanctSound, Glacier Bay, Stellwagen)

### Takeaway
Access is open and cloud-hosted. The vessel products are binary "ship present" event detections, not vessel-type labels. A type-labelled set would need an AIS join you build yourself.

### Cited Findings
- NCEI archives 12+ long-term projects (SanctSound, ADEON, Ocean Noise Reference Station Network and others). Access is via an interactive map, a public Google Cloud bucket, the SanctSound portal, or email to pad.info@noaa.gov. Formats are WAV, FLAC, netCDF and CSV, and each dataset has its own DOI citation. The archive DOI is 10.25921/PF0H-SQ72. The documentation shows no vessel/AIS products — [NCEI Passive Acoustic Data](https://www.ncei.noaa.gov/products/passive-acoustic-data)
- SanctSound vessel-detection products on ERDDAP (for example Olympic Coast OC02_02) hold 1,805 events for 11 Jul to 30 Oct 2019. They were detected with Triton's Ship-Detector Remora on a 48 kHz LTSA, flagged, and manually reviewed as vessel or ambient. The variables are start/end time and a binary ship-present flag. No AIS or vessel type is in the metadata, and the licence text says the data "may contain inaccuracies" with no warranty — [ERDDAP OC02_02_ships](https://coastwatch.pfeg.noaa.gov/erddap/info/noaaSanctSound_OC02_02_ships/index.html)
- Similar `_ships` datasets exist for other sites such as FK01, SB01, SB03 and GR02 (search results: [ERDDAP SB01_11](https://coastwatch.pfeg.noaa.gov/erddap/griddap/noaaSanctSound_SB01_11_ships.html)).

### Inferences
- The event catalogue gives free "ship vs ambient" timestamps, which is useful for the ambient class (E) and for locating passages. Vessel type would need AIS (for example NOAA/BOEM MarineCadastre AIS, not checked here) joined to the timestamps. Several sites are low-frequency or decimated, and I did not verify per-site sampling rates.
- A good candidate for later work, not for a three-week plan.

### Gaps
- NCEI/NOAA licence wording (US government works are normally public domain) was not verified on a NOAA page.
- Glacier Bay and Stellwagen vessel-labelled products, and per-site sampling rates, were not checked.

## 3. Other archives

### Takeaway
Two European AIS-labelled corpora (DeuteroNoise, Belgian North Sea) are open and could supplement ONC, especially for small craft. Most cabled and bioacoustic archives have audio but no vessel labels.

### Cited Findings
- DeuteroNoise: three European sites (Catalan coast, Venice Lagoon, western Black Sea). It has nearly 700 h of public continuous audio, with more than 11 h labelled. The licence is CC BY 4.0. Labels include cargo vessel traffic, workboats, leisure craft and natural background, and events carry ship identity, position and speed metadata. It is hosted at the URV DAU repository — [DAU record](https://dau.url.edu/handle/20.500.14342/5963) (licence and content read via [merit.url.edu](https://merit.url.edu/en/publications/the-deuteronoise-dataset-an-open-calibrated-multi-basin-resource-/)). Sampling rate not stated in what I read. Small craft and workboats make it a good fit for ShipsEar-style classes B and A.
- Belgian North Sea "AIS-annotated Hydrophone Recordings for Vessel Classification" (LifeWatch Broadband Acoustic Network): 10 s segments, 27,524 labelled segments from 116 days at two stations (Gardencity, Grafton), FLAC files of about 10 min, labelled with AIS position, speed, type and activity — [marineinfo.org dataset 8863](https://marineinfo.org/id/dataset/8863) and [8773](https://marineinfo.org/id/dataset/8773). Pages blocked fetching, so this rests on search snippets. Licence and sampling rate: unverified.
- OOI Regional Cabled Array hydrophones: OOI says its data has no specific licence beyond acknowledgement and citation guidelines. Hydrophone data is delivered as miniSEED through the API, with WAV/FLAC under development at the time of the post — [OOI forum](https://discourse.oceanobservatories.org/t/license-type-for-mseed-hydrophone-recordings/753). No vessel labels found.
- MBARI Pacific Sound (MARS hydrophone): open on the AWS Registry of Open Data under CC BY 4.0, full-resolution 256 kHz plus daily 16 kHz and 2 kHz files, about 2 TB per month, recording since July 2015 — [MBARI passive acoustic data](https://www.mbari.org/data/passive-acoustic-data/), [AWS registry](https://registry.opendata.aws/pacific-sound) (licence and rates from search summary). No vessel labels found. The site is at 900 m depth in Monterey Bay.
- Orcasound: public S3 buckets, FLAC archive plus HLS live streams, 2018 to present. The licence is CC BY-NC-SA 4.0 (non-commercial, share-alike). The registry entry has no vessel labels. Ship noise is contextualised with AIS in project work — [AWS Open Data Orcasound](https://registry.opendata.aws/orcasound). The NC clause may conflict with an openly licensed or commercial deliverable.
- IMOS (Australia): "Underwater noise signatures of ships in Australian waters" gives ship source spectra from AIS plus five IMOS deployments (Perth Canyon, Portland; 1,463 days, 2012-2016) under CC BY 4.0. The release is zipped CSV spectra and Excel source levels, not audio, and it is mostly cargo vessels and some tankers — [Research Data Australia](https://researchdata.edu.au/underwater-noise-signatures-australian-waters/1712034). The IMOS passive acoustic observatory raw data is discoverable via AODN ([data.gov.au record](https://data.gov.au/data/dataset/4f0b036a-d7a4-4f78-821e-b9461ccb5b6d), blocked 403, so its licence and formats are unverified).
- SWellEx-96: UCSD Library Digital Collections, a May 1996 experiment off Point Loma with towed sources at 50 to 400 Hz and vertical-array data. The page says the work is under US copyright and use beyond fair use needs UC Regents permission — [UCSD Library](https://library.ucsd.edu/dc/collection/bb3312136z). It is a localization dataset, not a vessel-type classification set, and the licence is not open.

### Inferences
- OOI, MBARI, Orcasound and the Watkins archive are not good starting points. There is no vessel-type labelling, and the AIS join would be a separate project. Watkins is a marine-mammal archive, so I did not search for it.
- DeuteroNoise and the Belgian set are the most useful supplements, for small-craft and workboat classes.

### Gaps
- Not covered: EMODnet, JERICO, JOMOPANS, ICES/SONS vessel-noise datasets, Watkins, US Navy/ONR releases other than SWellEx, and the Glacier Bay/Stellwagen detail. Time/tool-call budget ran out.
- Belgian dataset licence and sampling rate, DeuteroNoise sampling rate, and IMOS audio licence are unverified.

## 4. Feasibility within three weeks

### Takeaway
Realistic: UniqueShip (primary), optionally DeuteroNoise and the Belgian North Sea set as supplements. Future work: building labels from raw ONC/NOAA plus AIS, and the other archives.

### Cited Findings
- See section 1 for UniqueShip's size, licence and splits. Section 3 covers DeuteroNoise and the Belgian set.

### Inferences
- Plan: week 1, download the subset (92 GB) and check the licence text on uniqueshipdata.org and the ONC attribution wording. Week 2, train with the provided vessel-disjoint splits. Week 3, evaluate, and check whether DeuteroNoise adds the motor/leisure/workboat classes ShipsEar covers.
- The label taxonomy differs from ShipsEar, so frame the project as a "ShipsEar-like" classifier on AIS-type labels (cargo, tanker, passenger, tug). Rewriting the taxonomy to A-E needs a clear justification.
- Sampling-rate caveat: UniqueShip is 20 kHz, DeepShip 32 kHz, ShipsEar 52.7 kHz originally, so resample to a common rate before mixing.

### Gaps
- Class-balance, SNR and distance statistics for UniqueShip beyond what the site states were not examined.
