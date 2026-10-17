# Spike: CNN vs CRNN vs LSTM on fold 1 (preliminary)

Run on Kaggle (T4 GPU) on 2026-10-16 with notebook `spike_cnn_crnn_lstm.ipynb`. Raw outputs are in `results/`.

**These numbers are preliminary.** They come from one fold, a capped training set and 25 epochs. The real comparison runs on all 5 folds in S15.

New to machine learning? Read the [plain-language guide](#plain-language-guide) at the end first.

## Question

The CNN averages its features over time. Does a model that reads the clip as a sequence (an LSTM) classify vessel types better?

## Setup

| | |
|---|---|
| Data | UniqueShip 12-class, official fold 1, vessel-disjoint |
| Train | 7,200 clips (600 per class, spread round-robin over vessels or days) |
| Validation | 1,650 clips (up to 150 per class) |
| Test | 4,477 clips, the full fold 1 test split (no dredger, so 11 classes) |
| Input | 5 s log-mel, 32 kHz, 64 mels × 313 frames, standardized |
| Training | AdamW lr 1e-3, cosine decay, batch 64, 25 epochs, 3 seeds per model |
| Model choice | best epoch by validation 12-class macro-F1; test scored once |

The three models:

- **CNN** (242 k params): 4 conv blocks, then the mean over time.
- **CRNN** (341 k params): the same conv blocks, then a bidirectional LSTM over 78 time steps.
- **LSTM** (202 k params): a bidirectional LSTM directly on the 313 log-mel frames, with no conv layers.

## Results

Mean ± std over 3 seeds, on the fold 1 test split.

| Model | Macro-F1 (12-class) | Macro-F1 (9-class) | Accuracy | Train time / seed |
|---|---|---|---|---|
| CNN | 0.258 ± 0.006 | 0.303 ± 0.006 | 0.293 ± 0.005 | 185 s |
| **CRNN** | **0.265 ± 0.002** | **0.314 ± 0.005** | **0.298 ± 0.001** | 189 s |
| LSTM | 0.221 ± 0.021 | 0.263 ± 0.021 | 0.249 ± 0.022 | 61 s |

Random guessing over the 11 test classes would score about 0.09.

### Per-class F1 (mean of 3 seeds)

| Class | Test vessels / days | Test clips | CNN | CRNN | LSTM | CRNN − CNN |
|---|---|---|---|---|---|---|
| background | 82 days | 362 | 0.601 | 0.601 | 0.496 | 0.000 |
| cargo | 311 | 457 | 0.420 | 0.408 | 0.293 | −0.012 |
| tanker | 34 | 526 | 0.409 | 0.480 | 0.433 | **+0.071** |
| pleasurecraft | 50 | 445 | 0.294 | 0.299 | 0.254 | +0.005 |
| tug | 39 | 473 | 0.266 | 0.287 | 0.254 | +0.021 |
| fishing | 31 | 468 | 0.262 | 0.235 | 0.146 | −0.027 |
| sailing | 26 | 441 | 0.204 | 0.245 | 0.254 | **+0.041** |
| passengership | 13 | 432 | 0.211 | 0.221 | 0.180 | +0.010 |
| rescue | 3 | 412 | 0.109 | 0.085 | 0.057 | −0.024 |
| other | 20 | 425 | 0.064 | 0.049 | 0.060 | −0.015 |
| pilotvessel | 1 | 36 | 0.000 | 0.007 | 0.000 | +0.007 |

### Figures

![Confusion matrices, seed 0](results/confusion.png)

![Validation macro-F1 per epoch](results/val_curves.png)

## What it means

1. **The CRNN is slightly ahead of the CNN, but the gap is not yet proven.** On 12 classes the gap (+0.007) is about the size of the CNN's seed spread (0.006), and the seeds overlap: the best CNN seed (0.265) beats the worst CRNN seed (0.264). On 9 classes the gap (+0.011) is about twice the spread, which is suggestive but comes from only one fold. The CRNN is also more stable across seeds.
2. **The LSTM alone is clearly worse.** It scores about 0.04 below the CNN, well outside the seed spread, and it varies most between seeds (one seed peaked at epoch 3). Raw frames are a long sequence (313 steps) with no frequency-pattern learning. This supports putting conv layers in front of the LSTM rather than using the LSTM alone. It is about 3× faster to train.
3. **The CRNN may not have finished learning.** Its best epochs were 19, 23 and 24, while the CNN peaked at epochs 13–18 (see the validation curves). More epochs could help the CRNN more than the CNN.
4. **Where the CRNN helps:** tanker (+0.07) and sailing (+0.04) improve the most. The biggest error in every model is cargo vs tanker. Both are large merchant ships with similar engine sound.
5. **Some classes cannot be judged on this fold.** Pilotvessel has 1 test vessel (36 clips) and rescue has 3, so their F1 mostly reflects those few vessels.
6. **Validation scores (about 0.32) are higher than test (about 0.26).** New vessels in the test split are harder than the validation vessels. This is expected with vessel-disjoint splits.

## Limits

- One fold out of 5; fold-to-fold differences are likely larger than seed differences.
- Training capped at 600 clips per class and 25 epochs; no augmentation and no tuning.
- Absolute scores are low and are not v1.0 numbers.
- The LSTM-only model is a standard, untuned version (final state only, no gradient clipping). Tuning might narrow its gap, though probably not close it.
- Audio is limited to 16 kHz by the dataset's 32 kHz sample rate; any ship sound above 16 kHz is not available to the models.

## Next steps

- Keep the CRNN in S14 as a config option next to the CNN (mean pool or bi-LSTM).
- In S15, compare baseline, CNN and CRNN on all 5 folds. Count a CRNN gain only if it is larger than the fold-to-fold spread.
- Consider training the CRNN longer (for example 40 epochs) and check if it keeps improving.
- Drop the LSTM-only model from v1.0.

## Plain-language guide

### The task

Each example is a 5-second recording from an underwater microphone, labeled with what was nearby: a ship type or background (no ship). There are 12 labels, called **classes**. A **model** learns to guess the class of a new recording from many labeled examples. This is **supervised learning**: every training example comes with the right answer.

### Train, validation and test

- **Train** (7,200 clips): the model learns from these.
- **Validation** (1,650 clips): checked after each pass to choose the best version of the model.
- **Test** (4,477 clips): the final exam, used once.

A ship in the test set never appears in training (**vessel-disjoint**). Otherwise a model could memorize individual ships instead of learning what a ship type sounds like.

### Sound as a picture

Each clip becomes a **log-mel spectrogram**: left to right is time (313 steps), bottom to top is pitch (64 bands), and brightness is loudness. The recordings are sampled at 32 kHz (32,000 measurements per second), so they hold sound up to 16 kHz, half the sample rate. Higher sounds are not in the data.

### The three models

| Model | How it reads the picture |
|---|---|
| CNN | scans small windows for patterns (such as an engine hum at a certain pitch), then averages them over time |
| LSTM | reads the picture one time step after another, with a memory of what came before ("long short-term memory") |
| CRNN | CNN pattern-finding first, then an LSTM reads those patterns in order |

### How training works

A model is a large set of numbers called **weights** (about 242,000 for the CNN). They start random and are adjusted step by step:

1. **Guess** the class for a **batch** of 64 clips.
2. **Measure the mistake** as one number, the **loss**.
3. **Work out** which way to nudge each weight to lower the loss (**backpropagation**).
4. **Nudge** every weight a little (**gradient descent**). The size of the nudge is the **learning rate**.

One **epoch** is one pass through all 7,200 training clips, about 113 batches. 25 epochs is about 2,800 learning steps. After each epoch the model is scored on validation, and the best epoch is kept.

The recipe line "AdamW lr 1e-3, cosine decay, batch 64, 25 epochs" means: use the **AdamW** optimizer (it adapts the step size per weight and keeps weights small), start with a learning rate of 0.001, shrink it smoothly to zero (**cosine decay**), learn from 64 clips at a time, and pass through the data 25 times. All three models used the same recipe, so only the architecture differs.

### Seeds and the ± numbers

Training has random parts: the starting weights, the shuffle order and dropout. A **seed** is the number that fixes those random choices. The same seed repeats a run exactly, and a different seed gives a different random start. Each model was trained with seeds 0, 1 and 2, so 3 runs.

The **±** is how much the 3 runs differed (standard deviation). If one model beats another by less than the ±, the win may be luck. For example, CRNN vs CNN is +0.007 against a spread of about 0.006, so it is not yet proven. LSTM vs CNN is −0.037, and even the best LSTM run (0.239) is below the worst CNN run (0.253), so that gap is real.

### The scores

- **F1** scores one class from 0 to 1. It combines "when the model said tug, was it a tug?" with "of all the tugs, how many did it find?"
- **Macro-F1** averages F1 over the classes, each class counting equally, so rare classes matter as much as common ones. **Accuracy** (the share of correct guesses) can look good while rare classes are ignored.
- **12-class** uses all classes; **9-class** leaves out rescue, dredger and pilotvessel, which have very few ships.
- Random guessing would score about 0.09.

### How to read the figures

**Confusion matrix** (`confusion.png`): rows are the true class, columns are the guess, and each row adds up to 1.0. The diagonal is correct guesses; anything else is a mix-up. For example, in the CNN panel, the tanker row shows 0.53 under cargo: 53% of tankers were called cargo. A dark vertical stripe shows a model's "default" guess when unsure (pleasurecraft for the CNN, sailing for the CRNN).

**Validation curves** (`val_curves.png`): the validation macro-F1 after each epoch for every run. Rising lines mean the model is still learning; the CRNN lines are still rising at epoch 25.

### Why the LSTM alone does worse (likely reasons, not tested)

1. It has no built-in way to find a pattern at a slightly different pitch, which the CNN's sliding windows do naturally.
2. Remembering across 313 steps is hard; the CRNN's LSTM reads only 78 steps of patterns the CNN already found.
3. Ship sound is mostly steady over 5 seconds, so the order in time adds little.
4. It was the least stable to train, and it was not tuned.

It trains about 3× faster, but training takes minutes either way, and the app only makes predictions, which take milliseconds for any of the three.

### What comes next: the baseline

The full comparison (S15) adds an **MFCC baseline**: a simple, classical method to compare against. MFCC boils each clip down to 40 hand-designed numbers (**features**), which go into a **random forest** (many decision trees voting) or an **SVM** (the widest boundary between classes), whichever scores better on validation. The question it answers: does a model that learns its own features from the full picture beat expert hand-designed features?
