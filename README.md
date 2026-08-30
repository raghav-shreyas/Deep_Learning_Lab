# Experiment 3 — CNN for Multi-Class Image Classification (CIFAR-10)

**Course:** CS3807 – Deep Learning Laboratory
**Framework:** TensorFlow / Keras
**Dataset:** CIFAR-10

## Overview

This experiment implements a Convolutional Neural Network (CNN) from architecture design through training, evaluation, and analysis, using the CIFAR-10 dataset (60,000 32×32 RGB images across 10 classes). It covers:

- Dataset exploration (class distribution, sample images)
- Preprocessing (normalization, one-hot encoding, train/validation split)
- CNN architecture design (Conv2D, BatchNorm, MaxPooling/AveragePooling, Dropout, Dense layers)
- Trainable parameter analysis per layer
- Model training with early stopping
- Evaluation (accuracy, precision, recall, F1-score, confusion matrix, classification report)
- Intermediate feature map visualization
- Pooling strategy comparison (Max vs. Average)
- Hyperparameter exploration (kernel size, filter count, optimizer)

## Repository Contents

```
.
├── CNN_CIFAR10_Image_Classification.ipynb   # Full notebook (code + outputs)
├── Experiment_3_Report.tex                  # LaTeX lab report
├── images/                                  # EPS figures used in the report
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

Open `CNN_CIFAR10_Image_Classification.ipynb` in Jupyter or Google Colab and run all cells top to bottom. CIFAR-10 downloads automatically via `tensorflow.keras.datasets.cifar10`. A GPU runtime is recommended (training the main model for 25 epochs is significantly slower on CPU).

## Model Architecture

```
Input (32, 32, 3)
├─ Block 1: Conv2D(32, 3x3) → BatchNorm → Conv2D(32, 3x3) → MaxPool(2x2) → Dropout(0.25)
├─ Block 2: Conv2D(64, 3x3) → BatchNorm → Conv2D(64, 3x3) → MaxPool(2x2) → Dropout(0.25)
├─ Block 3: Conv2D(128, 3x3) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
├─ Flatten
├─ Dense(256, ReLU) → Dropout
└─ Dense(10, Softmax)
```

Total parameters: **667,434** (666,986 trainable / 448 non-trainable — from BatchNorm).

## Results (25 epochs, Adam optimizer, batch size 64, early stopping)

| Metric              | Value  |
|---------------------|--------|
| Test Accuracy       | 0.8281 |
| Macro Precision     | 0.8294 |
| Macro Recall        | 0.8281 |
| Macro F1-score      | 0.8272 |

Strongest classes: frog (92.7%), ship (92.3%), automobile (87.5%). Weakest classes: cat (66.1%), bird (70.2%) — cat/dog is the single largest confusion pair.

### Hyperparameter Exploration (5-epoch quick runs)

| Configuration                        | Val. Accuracy | Parameters |
|---------------------------------------|:---:|---:|
| 3×3 kernel, Adam                      | ~0.71 | 667,434 |
| 5×5 kernel, Adam                      | ~0.71 | (larger) |
| 3×3 kernel, SGD                       | ~0.48 | 667,434 |
| 3×3 kernel, fewer filters (16,32,64)  | 0.6904 | 300,570 |

### Pooling Comparison (10-epoch controlled run)

| Pooling Strategy | Val. Accuracy |
|---|:---:|
| Max Pooling      | ~0.775 |
| Average Pooling  | ~0.81  |

## Report

The full lab report, including background theory, mandatory plots with interpretation, hyperparameter analysis, and discussion, is in [`Experiment_3_Report.tex`](./Experiment_3_Report.tex).

## References

1. I. Goodfellow, Y. Bengio and A. Courville, *Deep Learning*, MIT Press, 2016.
2. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006.
3. S. Haykin, *Neural Networks and Learning Machines*, Pearson, 2009.
4. Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-Based Learning Applied to Document Recognition," *Proceedings of the IEEE*, 1998.
5. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," CIFAR-10 Dataset, 2009.
6. TensorFlow/Keras Documentation.
