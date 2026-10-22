# Glossary

**Clip** — one 5-second hydrophone recording from the dataset. The unit that is classified and the unit that appears in the split manifests.

**Vessel** — a single ship, identified by its MMSI. The unit of splitting: all clips from one vessel stay in one split.

**Background** — clips with no vessel present, labelled with MMSI 0. Split by whole day instead of by vessel.

**Class** — a vessel type label (cargo, tug, …) or background. The dataset assigns each vessel one dominant class.

**Fold** — one of the dataset's five official vessel-disjoint train/validation/test partitions.

**Official folds** — the dataset's own folds. The project uses them as shipped and does not re-split.

**Leakage** — a vessel (or background day) appearing in more than one split of the same fold.

**12-class score** — macro-F1 over all twelve classes (those with test clips in the fold); shown beside the headline for completeness (the dataset authors publish no 12-class result).

**9-class score** — the headline: macro-F1 over background and every class with at least 50 distinct training vessels in every fold; excludes dredger, pilotvessel and rescue.

**Model confidence** — the model's softmax score per class, not a calibrated probability.

**Feature cache** — precomputed log-mel features for every clip, one per sample rate, built once so training does not need the raw audio.

**Clip id** — a clip's identity everywhere (manifests, cache, error lists): its path under `audio/` without `.wav`, such as `cargo/<start>_<end>`.

**Stale cache** — a feature cache whose recorded feature settings differ from the current config; training refuses it.

**Cross-source test** — evaluating on a different dataset (for example ShipsEar). Reported separately; out of scope for P0.
