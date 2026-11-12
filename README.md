# AcousticWise

Classifies vessel type from underwater sound, on the [UniqueShip](https://uniqueshipdata.org/) dataset (12 classes). A log-mel CNN beats an MFCC baseline on vessels never seen in training, and runs in a local Streamlit app.

![The app classifying an upload](docs/images/app-result.png)

*A public ShipsEar sample: out-of-domain audio, not a validated result.*

## Run the app

```
git clone https://github.com/yanyetang/acousticwise.git && cd acousticwise
uv sync
curl -L -o artifacts/runs/r2-cnn/cnn_fold1_20261025-145517/model.pt \
  https://github.com/yanyetang/acousticwise/releases/download/v1.0/model.pt
echo "4b073b8921d13968e65a62ddb98139bf85afd3ce135ef3ceb2d592163d7bc54d  artifacts/runs/r2-cnn/cnn_fold1_20261025-145517/model.pt" | shasum -a 256 -c
uv run acousticwise-app
```

The model is the CNN trained on fold 1 (1 MB; the `shasum` line checks the download). Upload a WAV of at least 4 s; audio over 300 s is classified on its first 300 s. Set `ACOUSTICWISE_RUN_DIR` to serve another run.

![A too-short upload gets a clear message](docs/images/app-error.png)

## How it works

```mermaid
flowchart LR
    subgraph Kaggle["Kaggle (the 26 GB audio stays here)"]
        A[UniqueShip bundle] --> B[cache_features.py<br/>log-mel per 5 s clip]
        B --> C[(Feature cache<br/>3.2 GB)]
        C --> T2[CNN / CRNN training<br/>on a T4 GPU]
    end
    subgraph Laptop
        C --> T1[Baseline training<br/>MFCC + random forest]
        M[(Split manifest<br/>official folds)] --> T1
        M --> T2
        T1 --> R[(Runs: model, config,<br/>labels, metrics)]
        T2 --> R
        R --> E[evaluate / compare<br/>test split, once]
        R --> P[Predictor]
        P --> S[Streamlit app]
    end
```

Splits are the dataset's five official folds, disjoint by vessel. Design: [`docs/technical-design.md`](docs/technical-design.md).

## Results

Test macro-F1, mean ± std over five folds. "9 classes" leaves out the three with 1 to 19 training vessels.

| Model | 9 classes | 9 classes, folds 1–5 | 12 classes | Accuracy |
|---|---|---|---|---|
| Random forest (MFCC) | 0.259 ± 0.045 | 0.264, 0.189, 0.268, 0.260, 0.315 | 0.218 ± 0.052 | 0.231 ± 0.073 |
| **CNN (log-mel)** | **0.322 ± 0.044** | 0.344, 0.260, 0.332, 0.301, 0.375 | **0.272 ± 0.053** | **0.275 ± 0.092** |
| CRNN (log-mel) | 0.315 ± 0.055 | 0.320, 0.236, 0.337, 0.298, 0.386 | 0.270 ± 0.070 | 0.273 ± 0.095 |

The CNN wins on all five folds; chance is about 0.11. Details: [`docs/results.md`](docs/results.md).

## Limitations

- One hydrophone network: it generalizes to unseen vessels at the same sites, not to other places or hardware.
- Dredger, pilotvessel and rescue have too few vessels to learn the type.
- Labels come from AIS, so clips can contain other sounds; `other` is a mixed class.

## Reproduce

- **Tier 1, no data:** `uv run pytest` (includes an end-to-end run on synthetic audio).
- **Tier 2, with data:** retraining matches the reported scores within ±1 point (baseline) and ±2 (CNN); checked in [`docs/results.md`](docs/results.md).

Data: download the bundle from [uniqueshipdata.org](https://uniqueshipdata.org/) (about 62 GB free while unzipping: 18 GB of zip parts, an 18 GB joined copy and 26 GB unzipped; 26 GB once the zips are deleted), or build the cache on Kaggle (technical design, section 12). Then:

```
# Plain path only: join the two zip parts, unzip to data/uniqueship/, build the cache
zip -s 0 allclass_5hours.zip --out joined.zip && unzip -q joined.zip -d data/ && rm joined.zip
mv data/allclass_5hours data/uniqueship
uv run python scripts/cache_features.py
# Both paths
uv run python scripts/build_manifest.py
uv run python -m acousticwise.train --fold 1 --model cnn
uv run python -m acousticwise.evaluate artifacts/runs/<run>
```

## Data and credits

Ocean Networks Canada hydrophone data via UniqueShip, CC-BY 4.0. Provenance, checksums and citation: [`DATA.md`](DATA.md).
