# CS3807 – Experiment 2: MLP for Multi-Class Image Classification (Fashion-MNIST)

Implementation of a Multi-Layer Perceptron (MLP) in TensorFlow/Keras for multi-class
classification on the Fashion-MNIST dataset, including baseline training/evaluation and
automated hyperparameter optimization with `RandomizedSearchCV` + SciKeras.

**Course:** CS3807 – Deep Learning Laboratory
**Program:** B.Tech Artificial Intelligence & Data Science, Semester V
**Institution:** Shiv Nadar University Chennai

## Overview

The notebook builds a baseline MLP (`784 → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)`),
trains and evaluates it on Fashion-MNIST, then searches a hyperparameter space (hidden layers,
neurons, learning rate, batch size, epochs, optimizer, activation, dropout) to find and retrain
an optimized configuration, and compares the two models.

## Dataset

- **Fashion-MNIST** — 60,000 training images, 10,000 test images, 10 classes, 28×28 grayscale.
- Loaded directly via `keras.datasets.fashion_mnist.load_data()` — no manual download needed.

## Repository Contents

```
.
├── CS3807_Experiment2_MLP_FashionMNIST.ipynb   # main notebook (all tasks)
├── report.tex                                   # LaTeX lab report
├── report.pdf                                   # compiled report
├── eps_figures/                                 # 600 DPI EPS versions of all plots
│   ├── SampleImages.eps
│   ├── ClassDistribution.eps
│   ├── TrainValAccuracy.eps
│   ├── TrainValLoss.eps
│   ├── ConfMatrixBaseline.eps
│   ├── ConfMatrixOptimized.eps
│   ├── HPSearch.eps
│   └── BaselineVsOptimized.eps
└── README.md
```

## Requirements

- Python 3.10+
- `tensorflow`
- `scikeras==0.13.0`
- `scikit-learn==1.5.2`
- `numpy`, `pandas`, `matplotlib`

Install:
```bash
pip install tensorflow scikeras==0.13.0 scikit-learn==1.5.2 numpy pandas matplotlib
```

> **Colab note:** If running in Google Colab, install `scikeras`/`scikit-learn` first, then
> **Runtime → Restart session**, then re-run all cells from the top. SciKeras 0.13.0 is not
> compatible with the newer scikit-learn (≥1.6) that Colab ships by default; without a restart
> you'll hit `AttributeError: 'super' object has no attribute '__sklearn_tags__'`.

## How to Run

1. Open `CS3807_Experiment2_MLP_FashionMNIST.ipynb` in Jupyter/Colab.
2. Run cells top to bottom in order:
   - **Setup** — installs, imports, seeds
   - **Data loading & exploration** — Task 1 (sample images, class distribution)
   - **Preprocessing** — Task 2 (flatten, normalize, one-hot, train/val split)
   - **Baseline model** — Tasks 3–5 (build, train 20 epochs, evaluate)
   - **Hyperparameter search** — `RandomizedSearchCV`, 25 iterations, 5-fold CV, on a
     10,000-image training subset (kept small to stay tractable in Colab)
   - **Optimized model** — retrained on the full training set using the best found params
   - **Comparison** — baseline vs. optimized metrics and plot
3. All plots are saved as `.png` files in the working directory as each cell runs.

## Outputs

Running the notebook produces:
- Console output: dataset shapes, `model.summary()`, per-epoch training logs, best
  hyperparameters, best CV accuracy, test-set metrics, training times for both models
- 9 saved plots (`.png`) covering dataset exploration, training curves, confusion matrices,
  the hyperparameter search results, and the baseline-vs-optimized comparison

## Results Summary

| Metric | Baseline | Optimized |
|---|---|---|
| Accuracy | 0.8715 | 0.8760 |
| Precision (macro) | 0.8755 | 0.8788 |
| Recall (macro) | 0.8715 | 0.8760 |
| F1-score (macro) | 0.8721 | 0.8766 |

The optimized model gives a modest improvement over the baseline. Most residual errors on
both models are concentrated among visually similar upper-body classes (Shirt, T-shirt/top,
Pullover, Coat) rather than being spread evenly across all classes — see `report.pdf` for
the full confusion-matrix analysis and discussion.

## Notes / Design Decisions

- The hyperparameter search space (3,888 total combinations) is too large for exhaustive
  `GridSearchCV` with 5-fold CV on 60k images, so `RandomizedSearchCV` (`n_iter=25`) is used
  instead, on a 10,000-image subset — this trade-off is documented in the report.
- `n_jobs=1` is used in the search for Keras/GPU stability in Colab; expect it to be slow.
- Random seeds (`numpy`, `tensorflow`) are fixed at 42 for reproducibility, though exact
  results can still vary slightly across hardware/TF versions.

## Report

`report.pdf` / `report.tex` contains the full write-up: objective, theory, dataset,
experimental procedure, source code, hyperparameter optimization methodology, results
tables, all 9 plots with inference, discussion, and conclusion.

## References

1. Goodfellow, I., Bengio, Y., and Courville, A., *Deep Learning*, MIT Press.
2. Bishop, C. M., *Pattern Recognition and Machine Learning*, Springer.
3. Haykin, S., *Neural Networks and Learning Machines*, Pearson.
4. [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist), Zalando Research.
5. [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
