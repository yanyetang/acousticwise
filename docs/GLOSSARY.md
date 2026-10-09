# Glossary

**Clip** — one 5-second hydrophone recording from the dataset. The unit that is classified and the unit that appears in the split manifests.

**Vessel** — a single ship, identified by its MMSI. The unit of splitting: all clips from one vessel stay in one split.

**Background** — clips with no vessel present, labelled with MMSI 0. Split by whole day instead of by vessel.

**Class** — a vessel type label (cargo, tug, …) or background. The dataset assigns each vessel one dominant class.

**Fold** — one of the dataset's five official vessel-disjoint train/validation/test partitions.

**Official folds** — the dataset's own folds. The project uses them as shipped and does not re-split.

**Leakage** — a vessel (or background day) appearing in more than one split of the same fold.

**12-class score** — macro-F1 over all twelve classes; comparable with the dataset authors' published baseline.

**9-class score** — macro-F1 over the classes with enough distinct vessels to be trusted; excludes dredger, pilotvessel and rescue (final rule pending).

**Model confidence** — the model's softmax score per class, not a calibrated probability.

**Feature cache** — precomputed features for every clip, built once so training does not need the raw audio.

**Cross-source test** — evaluating on a different dataset (for example ShipsEar). Reported separately; out of scope for P0.
