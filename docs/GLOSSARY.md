# Glossary

## Project terms

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

## Audio and features

**Sample rate** — how many times per second the sound is measured. UniqueShip is recorded at 32 kHz (32,000 samples a second). The highest frequency a rate can hold, the **Nyquist frequency**, is half of it: 16 kHz at 32 kHz.

**Resampling** — converting audio to another sample rate, so uploads at any rate match what the model was trained on.

**Window** — the fixed 5-second piece of audio the model looks at. A longer upload is cut into windows that overlap by half (a 2.5 s **hop**); the model scores each one and the scores are averaged.

**Padding** — adding silence to the end of a clip that is a little shorter than one window (4.0 to 5 s), so every window has the same length.

**Decibel (dB)** — a logarithmic scale for loudness: +6 dB is twice the amplitude, and each 10 dB is ten times the power. Spectrograms are stored in dB because hearing, and the useful differences between sounds, are closer to logarithmic.

**Peak and RMS level** — two measures of how loud a clip is: the peak is its single loudest sample, the RMS (root mean square) its average loudness.

**Normalization** — removing loudness differences between clips (a near ship is louder than a far one), so the model learns the shape of a sound rather than its volume. In a dB spectrogram it is just a constant added to every value, which is why the cache stores each clip's peak and RMS level and normalization is applied on load.

**Spectrogram** — a picture of sound: time left to right, frequency bottom to top, and the colour is how loud each frequency is at each moment.

**Mel scale and log-mel spectrogram** — the mel scale spaces frequency bands the way hearing does: narrow at low frequencies, wide at high ones. A log-mel spectrogram groups the spectrogram into 64 such bands and converts it to dB; for a 5 s window it is a 64 × 313 grid. It is the CNN's input.

**MFCC** (mel-frequency cepstral coefficients) — 20 numbers per moment that summarize the overall shape of the log-mel spectrum. The baseline uses the mean and spread of each over time: 40 numbers per clip.

**Feature** — a number or array computed from the raw audio for the model to learn from, such as the log-mel spectrogram or the MFCC summary.

**float16** — numbers stored with half the usual precision (about 3 significant digits). It halves the cache size; the rounding (under 0.1 dB) is far smaller than the differences the model learns from.

**Memory mapping** — opening a large file so that only the parts being read are loaded into memory. Training reads batches from the 1.7 GB cache this way.

## Training

**Training, validation and test splits** — the model learns from the training split; the validation split is used to choose settings and the best moment to stop; the test split is used once at the end to report results.

**Epoch** — one pass of training over every training clip. Training usually runs for many epochs.

**Batch** — the small group of clips (for example 32) the model sees at once before it adjusts its weights.

**Loss** — a number that measures how wrong the model's predictions are on a batch; training adjusts the weights to make it smaller. The project uses **cross-entropy**, the standard loss for choosing one class out of several.

**Class weights** — extra weight in the loss for rare classes, so the model cannot score well by ignoring them.

**Seed** — the starting number for the random choices in training (shuffling, initial weights). The same seed gives the same run; different seeds show how much results vary by chance (**run-to-run noise**, about ±1 point here).

**Overfitting** — when a model memorizes its training clips instead of learning what makes a class, so it scores well on training data and worse on new data.

**Train/serve skew** — when the features used in training differ from those computed when the model is used (in the app). The model then sees inputs unlike anything it learned from, and gets quietly worse without any error. The project avoids it with one preprocessing path and the stale-cache check.

**Data leakage** — when information from the test data reaches training, so the score looks better than the model really is. Here the main risk is the same vessel in both training and test: the model can recognize that ship rather than its type. See **Leakage** above.

## Models

**Baseline** — a simple model that sets the score to beat: here, MFCC summaries into a classical classifier (SVM or random forest).

**CNN** (convolutional neural network) — a model that learns patterns in images by sliding small filters over them; here the image is the log-mel spectrogram.

**LSTM** (long short-term memory) — a model that reads a sequence step by step and remembers what came before; used here to read the spectrogram over time.

**CRNN** — a CNN whose output is read over time by an LSTM, instead of being averaged over time.

**Pretrained model and fine-tuning** — a model already trained on another large task (for example **ImageNet**, about a million photos) and then trained further on this task. **MobileNetV3-Small** is one such small image model; in the spike it gave no gain over training from scratch.

**Logits** — the raw scores a model gives each class before they are turned into probabilities.

**Softmax** — turns the logits into scores between 0 and 1 that sum to 1 (the **model confidence**). The predicted class is the one with the highest score (the **argmax**).

## Evaluation

**Accuracy** — the share of clips classified correctly. It can look good on unbalanced data even when rare classes are always wrong.

**Precision and recall** — for one class: precision is the share of clips predicted as that class that really are; recall is the share of that class's clips that are found.

**F1** — one score per class that combines precision and recall (their harmonic mean); it is high only when both are.

**Macro-F1** — the average of the per-class F1 scores, so a rare class counts as much as a common one. The project's headline metric (see **9-class score**).

**Balanced accuracy** — the average of the per-class recalls; like macro-F1, it gives every class equal weight.

**Confusion matrix** — a table of true class against predicted class; the off-diagonal cells show which classes are mistaken for which (for example tanker called cargo).

**5-class rescore** — scoring a 12-class model on the dataset paper's 5 classes (background, cargo, passenger ship, tanker, tug) by taking the highest of only those 5 logits, so the result can be compared with the paper.

**Chance level** — the score of guessing at random: about 8% accuracy for 12 classes and 20% for 5.
