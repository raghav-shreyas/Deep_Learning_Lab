# Experiment 6 — End-to-End Study of RNN, LSTM and GRU for Sequence Learning

**CS3807 – Deep Learning Laboratory**
Shiv Nadar University Chennai | B.Tech AI & Data Science | Semester V

## Overview

This experiment implements and compares three recurrent architectures — **Vanilla RNN**,
**LSTM**, and **GRU** — for human activity classification from raw smartphone inertial
sensor signals, then extends the study to the effect of sequence length and to
sequence-to-sequence learning with an encoder–decoder LSTM.

## Contents

| File | Description |
|---|---|
| `Experiment_6_RNN_LSTM_GRU.ipynb` | Main notebook: data pipeline, model training, evaluation, sequence-length study, seq2seq task |
| `Deep_Learning_Experiment_6_Report.pdf` | Full lab report with plots, tables and discussion |

## Dataset

**UCI Human Activity Recognition (HAR) Using Smartphones** — raw inertial signals
(not the pre-computed 561-feature vectors).

- Input shape: `(N, 128, 9)` — 128 time steps × 9 channels (3-axis body acceleration,
  3-axis gyroscope, 3-axis total acceleration)
- Classes (6): `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`,
  `STANDING`, `LAYING`
- Split: 70% train / 15% validation / 15% test (test set untouched during model
  selection)
- Source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Pipeline

```
Raw Sensor Signals → Windowing (128×9) → Normalization → RNN / LSTM / GRU → Dense → Softmax → Activity Class
```

Common classifier skeleton (identical across all three models except the recurrent layer):

```
Input(128, 9) → [SimpleRNN | LSTM | GRU](32 units) → Dropout(0.2) → Dense(16, ReLU) → Dense(6, Softmax)
```

Training config: Adam optimizer (lr = 1e-3), sparse categorical cross-entropy loss,
batch size 32.

## Experiments

1. **Temporal visualization** — sensor signal patterns across activity classes
2. **RNN vs. LSTM vs. GRU** — trained under identical conditions, evaluated on
   accuracy, macro precision/recall/F1, confusion matrix, parameter count, training time
3. **Effect of sequence length** — same models retrained with T = 32, 64, 128
4. **Sequence-to-sequence learning** — encoder–decoder LSTM trained on a synthetic
   integer-sequence reversal task (e.g. `[1,4,7,2] → [2,7,4,1]`), evaluated on
   token-level and sequence-level accuracy

## Results Summary

| Model | Accuracy (%) | Macro F1 (%) | Parameters | Training Time (s) |
|---|---|---|---|---|
| RNN | 69.67 | 68.82 | 1,974 | 32.12 |
| LSTM | 89.05 | 89.07 | 6,006 | 68.06 |
| GRU | 89.17 | 89.17 | 4,758 | 113.02 |
| Seq2Seq (token / seq. acc.) | 99.68 / 98.70 | — | 52,874 | — |

**Key finding:** LSTM and GRU substantially outperform the Vanilla RNN and converge far
more smoothly; GRU matches LSTM's accuracy with ~21% fewer parameters. The Vanilla RNN's
performance degrades at longer sequence lengths (T = 128), consistent with the
vanishing-gradient limitation, while LSTM and GRU both improve with more temporal context.

See the full report (`Deep_Learning_Experiment_6_Report.pdf`) for plots, per-class
metrics, confusion matrices, and discussion.

## Requirements

```
tensorflow / keras
numpy
pandas
matplotlib
scikit-learn
```

## References

1. Goodfellow, Bengio & Courville, *Deep Learning*, MIT Press, 2016.
2. Hochreiter & Schmidhuber, "Long Short-Term Memory," *Neural Computation*, 1997.
3. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical
   Machine Translation," EMNLP, 2014.
4. Anguita et al., "A Public Domain Dataset for Human Activity Recognition Using
   Smartphones," ESANN, 2013.
