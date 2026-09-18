# Banknote Authentication with a Perceptron (From Scratch)

A from-scratch implementation of the single-layer Perceptron algorithm, applied to the [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication). The perceptron is built using only NumPy (no ML framework), trained with the classic online update rule, and benchmarked against scikit-learn's `Perceptron`.

## Overview

The notebook (`Experiment_1.ipynb`) walks through a complete, small-scale ML experiment:

1. Load the banknote authentication dataset via `ucimlrepo`
2. Explore and visualize the data (histograms, correlation heatmap, scatter plots, boxplots)
3. Preprocess: standardize features and split into train/test sets
4. Implement a Perceptron classifier from scratch
5. Train it and track per-epoch error, weights, and bias
6. Evaluate on the test set (accuracy, precision, recall, F1, confusion matrix)
7. Compare performance across different learning rates
8. Visualize the learned decision boundary in 2D
9. Benchmark against scikit-learn's `Perceptron`
10. Summarize results in tabular form

## Dataset

- **Source:** UCI Machine Learning Repository — Banknote Authentication (id `267`)
- **Samples:** 1,372
- **Features:** `variance`, `skewness`, `curtosis`, `entropy` (extracted from wavelet-transformed banknote images)
- **Target:** `class` — 0 (authentic) or 1 (forged)
- **No missing values**

## Perceptron Implementation

A minimal Perceptron class implementing:

- Weight initialization from a small random normal distribution
- Step activation function
- Online (per-sample) weight/bias updates using the perceptron learning rule
- Per-epoch history tracking (misclassified count, weights, bias) for later visualization

```python
class Perceptron:
    def __init__(self, n_features, learning_rate=0.01, n_epochs=50, random_state=42):
        ...
    def fit(self, X, y, verbose=True):
        ...
    def predict(self, X):
        ...
```

## Results

Trained with `learning_rate=0.01`, `n_epochs=50`, on an 80/20 stratified train/test split (1,097 train / 275 test):

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.9891 |
| Precision | 0.9837 |
| Recall    | 0.9918 |
| F1-score  | 0.9878 |

### Learning Rate Comparison

| Learning Rate | Accuracy | F1-score |
|---------------|----------|----------|
| 0.001         | 0.9782   | 0.9758   |
| 0.01          | 0.9891   | 0.9878   |
| 0.1           | 0.9818   | 0.9799   |

### Scratch vs. scikit-learn

| Implementation | Accuracy | F1-score |
|-----------------|----------|----------|
| From scratch     | 0.9891   | 0.9878   |
| scikit-learn     | 0.9745   | 0.9721   |

The from-scratch implementation slightly outperforms scikit-learn's default `Perceptron` on this run, likely due to differences in initialization and learning-rate scheduling.

## Visualizations

The notebook produces:

- Feature distribution histograms
- Correlation heatmap
- Scatter plot of `variance` vs. `skewness` by class
- Feature boxplots
- Training error (misclassifications) vs. epoch
- Weight evolution over epochs
- Bias evolution over epochs
- Confusion matrix
- Learning-rate comparison curves
- 2D decision boundary plot (`variance` vs. `skewness`)

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
ucimlrepo
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn ucimlrepo
```

## Usage

1. Clone the repository and open `Experiment_1.ipynb` in Jupyter or Google Colab.
2. Run all cells top to bottom — the dataset is fetched automatically via `ucimlrepo`.
3. Review the printed metrics and generated plots.

## Project Structure

```
.
├── Experiment_1.ipynb   # Main notebook: data loading, EDA, perceptron implementation, evaluation
└── README.md
```

## Author

Sanjay — B.Tech AI & Data Science, Shiv Nadar University Chennai
