# Oxford-IIIT Pet Classification — CNN Hyperparameter & Transfer Learning Study

A systematic study of neural network design choices for image classification, using **MobileNetV2** transfer learning on the **Oxford-IIIT Pet** dataset (37 breeds of cats and dogs). The notebook empirically compares weight initialization schemes, regularization strategies, optimizers, learning rates, batch sizes, feature extraction vs. fine-tuning, and validates the best configuration with k-fold cross-validation before final testing.

## Overview

The notebook (`Lab-5.ipynb`) is organized as a sequence of controlled experiments, each isolating one design choice:

1. **Setup** — load `tensorflow`, `tensorflow_datasets`, and set global seeds/config (`IMG_SIZE=224`, 37 classes, batch size 32)
2. **Data pipeline** — load the Oxford-IIIT Pet dataset via `tensorflow_datasets`, split 85/15 into train/validation, preprocess (resize, MobileNetV2 `[-1, 1]` scaling, random flip + brightness augmentation on the training set)
3. **Model builder** — a configurable function that builds a MobileNetV2-based classifier with adjustable weight initialization, dropout, L2 regularization, batch normalization, frozen/unfrozen base, and optimizer/learning rate
4. **Experiment 1 — Weight initialization:** compares `zeros`, `random`, `glorot`, and `he` initialization (training loss & validation accuracy curves)
5. **Experiment 2 — Regularization:** compares no regularization, L2, dropout, and batch normalization
6. **Manual batch-norm walkthrough:** a small worked numerical example of the batch normalization formula
7. **Experiment 3 — Optimizers:** compares SGD, SGD+momentum, RMSprop, and Adam (loss/accuracy curves + a summary table with best validation accuracy, convergence epoch, and training time)
8. **Experiment 4 — Hyperparameter sweeps:** learning rate (`1e-3` vs `1e-4`), batch size (`16`/`32`/`64`), and dropout rate, each plotted against validation accuracy
9. **Experiment 5 — Feature extraction vs. fine-tuning:** trains with the MobileNetV2 base frozen, then continues training with the last ~30 layers unfrozen at a lower learning rate
10. **5-fold cross-validation:** evaluates four candidate configurations (baseline, dropout+BN, L2, RMSprop) over 5 folds on a subset of the data, selecting the best by mean validation accuracy
11. **Final model training:** retrains the best configuration on the full train/validation split for more epochs, then evaluates on the held-out test set
12. **Final evaluation:** test accuracy, macro precision/recall/F1, parameter count, confusion matrix, most-confused class pairs, and a sample of misclassified images
13. **Summary table:** a consolidated comparison of CV accuracy, test accuracy, and training time across all major configurations tried

## Dataset

- **Dataset:** [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) via `tensorflow_datasets` (`oxford_iiit_pet`)
- **Classes:** 37 cat and dog breeds
- **Preprocessing:** images resized to 224×224 and scaled to `[-1, 1]` (MobileNetV2 convention); training images augmented with random horizontal flip and brightness jitter
- **Split:** 85/15 train/validation from the dataset's training split, plus the dataset's original test split held out for final evaluation
- **Cross-validation subset:** a capped sample (1,500 examples by default, configurable) used for the 5-fold CV stage to keep runtime manageable

## Model Architecture

A configurable builder function producing:

`Input(224, 224, 3) → MobileNetV2 (ImageNet weights, global-avg pooling) → [optional BatchNorm] → Dropout → Dense(37, Softmax)`

Configurable dimensions explored in the notebook:

| Dimension | Options tested |
|---|---|
| Weight initialization | zeros, random normal, Glorot uniform, He normal |
| Regularization | none, L2, dropout, batch normalization |
| Optimizer | SGD, SGD+momentum, RMSprop, Adam |
| Learning rate | 1e-3, 1e-4 (and 1e-5 for fine-tuning) |
| Batch size | 16, 32, 64 |
| Transfer learning mode | frozen base (feature extraction) vs. last ~30 layers unfrozen (fine-tuning) |

## Evaluation Methodology

- Exploratory sweeps (initialization, regularization, optimizer, learning rate, batch size, dropout) are run for a small number of epochs (`EPOCHS_SMALL = 5`) to compare configurations quickly.
- The most promising configurations are validated more rigorously with **5-fold cross-validation** (mean ± standard deviation of validation accuracy).
- The overall best configuration is retrained for more epochs (`EPOCHS_FULL = 15`) on the full data and evaluated once on the untouched test set.
- Final metrics: test accuracy, macro-averaged precision/recall/F1, confusion matrix, top confused class pairs, and qualitative inspection of misclassified images.

Since this is an empirical hyperparameter study, exact numeric results (best initialization, best optimizer, final test accuracy, etc.) are produced by running the notebook rather than fixed here — the final summary table at the end of the notebook is designed to be filled in with those results.

## Visualizations

The notebook produces 14–15 plots, including:

- Training loss and validation accuracy vs. epoch, by weight initialization
- Training/validation accuracy and loss, by regularization scheme
- With vs. without batch normalization
- Training loss and validation accuracy vs. epoch, by optimizer
- Learning rate, batch size, and dropout rate vs. validation accuracy
- Feature extraction vs. fine-tuning accuracy/loss curves
- 5-fold cross-validation accuracy with error bars
- Final confusion matrix and a sample of misclassified test images

## Requirements

```
tensorflow
tensorflow-datasets
numpy
pandas
matplotlib
seaborn
scikit-learn
```

Install with:

```bash
pip install tensorflow tensorflow-datasets numpy pandas matplotlib seaborn scikit-learn
```

A GPU is strongly recommended — the notebook trains dozens of model variants (initialization × regularization × optimizer × hyperparameter sweeps × 5-fold CV × final model).

## Usage

1. Open `Lab-5.ipynb` in Google Colab or Jupyter (Colab is recommended; the first cell installs dependencies for a Colab session).
2. Enable a GPU runtime if using Colab (`Runtime > Change runtime type > GPU`).
3. Run the cells top to bottom. Each experiment section builds on shared helper functions (`build_model`, `train_and_time`, `make_split`) defined earlier in the notebook.
4. The 5-fold CV and final training stages are the most time-consuming — adjust `SUBSET`, `EPOCHS_SMALL`, and `EPOCHS_FULL` to trade off runtime vs. thoroughness.
5. Review the final metrics table, confusion matrix, and summary comparison table at the end.

## Project Structure

```
.
├── Lab-5.ipynb   # Main notebook: data pipeline, model builder, hyperparameter experiments, CV, final evaluation
└── README.md
```

## Author

Sanjay — B.Tech AI & Data Science, Shiv Nadar University Chennai
