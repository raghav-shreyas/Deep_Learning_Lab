# CIFAR-10 Image Classification with Transfer Learning (MobileNetV2)

An image classification pipeline that fine-tunes a pretrained **MobileNetV2** (ImageNet weights) on the **CIFAR-10** dataset using TensorFlow/Keras. The notebook covers data loading, preprocessing, a two-stage transfer-learning workflow (frozen base → fine-tuning), evaluation, and result visualization.

## Overview

The notebook (`Lab-4.ipynb`) implements the following pipeline:

1. Load CIFAR-10 — either directly via `tensorflow.keras.datasets` or from a Kaggle-hosted pickled version
2. Subsample the dataset (20,000 train / 4,000 test) and normalize pixel values to `[0, 1]`
3. Visualize sample images from each class
4. Build a transfer-learning model: `Resizing → MobileNetV2 (frozen, ImageNet weights) → GlobalAveragePooling2D → Dense(128, ReLU) → Dense(10, Softmax)`
5. **Stage 1:** Train the classification head with the MobileNetV2 base frozen
6. **Stage 2 (fine-tuning):** Unfreeze the last 20 layers of MobileNetV2 and continue training at a lower learning rate
7. Evaluate on the test set: accuracy, precision, recall, F1-score, confusion matrix, classification report
8. Visualize training/validation accuracy and loss curves across both stages
9. Visualize the confusion matrix and a sample of misclassified images
10. Summarize frozen-base vs. fine-tuned performance and training time

## Dataset

- **Dataset:** CIFAR-10 (10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
- **Full size:** 50,000 train / 10,000 test, 32×32 RGB images
- **Used subset:** 20,000 train / 4,000 test (random subsample, seeded for reproducibility) to keep training time manageable
- **Loading options:**
  - **Option A:** Download via Kaggle API (requires a `kaggle.json` API token) — pickled CIFAR-10 batches
  - **Option B:** Load directly via `tensorflow.keras.datasets.cifar10` (simpler, no external credentials needed)

## Model Architecture

| Stage | Component | Trainable |
|-------|-----------|-----------|
| Input | `Input(32, 32, 3)` → `Resizing(96, 96)` | — |
| Base | MobileNetV2 (ImageNet weights, no top) | Frozen (Stage 1) → last 20 layers unfrozen (Stage 2) |
| Head | `GlobalAveragePooling2D → Dense(128, ReLU) → Dense(10, Softmax)` | Yes |

- Input images are resized from 32×32 to 96×96 to preserve more detail for the MobileNetV2 backbone.
- **Stage 1:** Adam optimizer, learning rate `0.001`, 8 epochs, base frozen.
- **Stage 2 (fine-tuning):** Adam optimizer, learning rate `0.0001`, 5 epochs, last 20 layers of the base unfrozen.

## Evaluation

The notebook computes, on the held-out test subset:

- Accuracy, weighted Precision, Recall, and F1-score
- Full per-class classification report
- Confusion matrix (heatmap)
- A comparison table of training accuracy, validation accuracy, and validation loss between the frozen-base and fine-tuned stages, along with training time per stage and total parameter count

Since results depend on the random subsample, number of epochs, and hardware, exact metric values are produced when the notebook is run rather than hardcoded here.

## Visualizations

The notebook generates and saves:

- `sample_images.png` — sample images from the training subset with class labels
- `accuracy_plot.png` — training vs. validation accuracy across both training stages
- `loss_plot.png` — training vs. validation loss across both training stages
- `confusion_matrix.png` — confusion matrix heatmap over all 10 classes
- `misclassified.png` — a sample of misclassified test images with true/predicted labels

## Requirements

```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
kaggle          # only needed if using the Kaggle download option
```

Install with:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kaggle
```

> **Note:** The notebook is written for Google Colab (uses `google.colab.files` for the Kaggle token upload). If running locally, replace that cell with a manual copy of your `kaggle.json` into `~/.kaggle/`, or skip Option A entirely and use Option B (Keras' built-in CIFAR-10 loader).

A GPU is strongly recommended for training MobileNetV2 in reasonable time.

## Usage

1. Open `Lab-4.ipynb` in Google Colab (or Jupyter, with the local-setup note above).
2. Run the setup cell to confirm TensorFlow and GPU availability.
3. Choose **one** data-loading option:
   - Option A: upload your `kaggle.json` and download the Kaggle CIFAR-10 batches, or
   - Option B: load CIFAR-10 directly through Keras (recommended for simplicity).
4. Run the remaining cells top to bottom to preprocess data, build the model, train (frozen base, then fine-tune), and generate evaluation metrics and plots.

## Project Structure

```
.
├── Lab-4.ipynb              # Main notebook: data loading, model, training, evaluation
├── sample_images.png        # Generated on run
├── accuracy_plot.png        # Generated on run
├── loss_plot.png            # Generated on run
├── confusion_matrix.png     # Generated on run
├── misclassified.png        # Generated on run
└── README.md
```

## Author

Sanjay — B.Tech AI & Data Science, Shiv Nadar University Chennai
