[README.md](https://github.com/user-attachments/files/33415417/README.md)
# Eruption_Prediction_Team12# Predicting Eruptive Events at Kīlauea from Earthquake Data

**Replication and extension of:** Mullet, Singh & Wong, *Predicting Eruptive Events at Volcanoes from Earthquake Data* (Stanford CS229, Fall 2019).  
**Notebook:** `mlminiproj.ipynb`  
**Original repository:** https://github.com/bmullet/PEEVED

## Project overview

This project reproduces the paper's machine-learning experiments for Kīlauea's East Rift Zone and adds an experimental extension. It uses a historical earthquake catalogue to investigate two tasks:

1. **Eruption-state classification:** predict whether a quake occurred during a labelled eruptive period.
2. **Time-to-eruption regression:** estimate the time until the next listed eruption start (zero for quakes labelled as occurring during an eruption).

This is a research replication exercise, **not an operational volcano early-warning system**.

## Dataset

The notebook clones the authors' public `PEEVED` repository at runtime and reads the earthquake catalogue and eruption episode table from it. The East Rift Zone filter produces **3,764 earthquakes**, spanning **1 January 1983 to 30 December 1986**, and the notebook reads **49 eruption episodes** from `PuuOo.csv`.

Baseline input features:

- Longitude and latitude
- Earthquake depth and magnitude
- Counts of earlier earthquakes within the previous 1, 7 and 30 days

Additional Part B features:

- `log_energy_7d`: base-10 log of cumulative estimated radiated seismic energy over the previous 7 days
- `maxmag_7d`: maximum magnitude over the previous 7 days
- `meandepth_7d`: mean depth over the previous 7 days
- `rate_ratio_d_m`: daily-to-monthly earthquake-count ratio, with `+1` smoothing in numerator and denominator
- `log_gap_prev`: base-10 log of one plus the time gap (hours) since the previous earthquake

The erupting label follows the notebook's interpretation of the episode table: earthquakes within the episode duration plus a 24-hour buffer after an eruption start are labelled erupting. This label construction is a simplification and should be kept in mind when interpreting results.

## Models

### Part A — replication

- Logistic regression
- K-means clustering
- Random forest
- Four-hidden-layer neural network (1,000 units per hidden layer)

Part A uses the authors' train/dev/test index files (2,633 / 752 / 379 earthquakes), the original seven features, and a scaler fitted across the full dataset to mirror the paper's described setup.

### Part B — extension

Part B keeps the four baseline model families and adds **HistGradientBoosting** for classification and regression. It also changes the evaluation/preprocessing setup:

- Assigns whole **14-day blocks** to train/dev/test (approximately 70/15/15 by block, with 2,848 / 429 / 487 earthquakes in the saved run)
- Fits feature and target scalers on training data only
- Adds the five seismic-summary features above
- Uses class weighting for logistic regression, random forest and HistGradientBoosting classification; weighted binary cross-entropy for the neural-network classifier
- Tunes classification thresholds on the development set to maximise Cohen's kappa

The blocks in Part B are shuffled before assignment; this is a block-based split, **not a strictly chronological future-only test**. The notebook explains that a fully chronological split would make the test set all-positive because of the eruption labels late in this short catalogue.

## Run in Google Colab

1. Upload `mlminiproj.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run **Runtime → Run all**.
4. The first data cell clones `https://github.com/bmullet/PEEVED` into the Colab working directory. The notebook imports NumPy, pandas, Matplotlib, scikit-learn and PyTorch; Colab generally provides these packages already.
5. At the end, inspect the printed test-set tables and generated plots.

A compatible local environment can use Python 3.10+ and install the main dependencies with:

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

Then start Jupyter and open the notebook. Git must be installed and internet access must be available so the notebook can clone the source repository. A GPU is strongly recommended when `RUN_NN = True`.

## Runtime

The notebook's introductory estimate is **about 10–25 minutes** on a Colab GPU, depending on hardware and runtime conditions. The saved execution log reports about **3.9 minutes for Part A** and **4.3 minutes for Part B** on the GPU used for that run. CPU-only execution is slower, especially for the four 1,000-epoch neural-network training runs. Set `RUN_NN = False` near the top of the notebook to skip neural networks if you need a quicker run; the other models still execute.

## Outputs

The notebook saves:

- `all_results.csv` — train, development and test metrics for both parts
- `summary_test_scores.png` — comparison chart for test Cohen's kappa and regression R²
- Inline plots for model tuning, feature importance, neural-network loss and observed-vs-predicted regression values

In Colab, these files are created in the working directory (typically under `/content/PEEVED` after the clone cell changes directory). Download them from the Files panel if needed.

## Saved test-set results

### Classification

The notebook reports Cohen's kappa, an AUROC calculated from **hard class predictions**, and a separate AUROC calculated from **predicted probabilities** for models that provide probabilities. Probability-based AUROC is the conventional ranking measure; do not mix the two columns in a paper comparison.

| Part | Model | Kappa | AUROC (hard labels) | AUROC (probabilities) |
|---|---|---:|---:|---:|
| A | Neural Network | 0.394 | 0.652 | 0.730 |
| A | Random Forest | 0.393 | 0.646 | 0.801 |
| A | Logistic Regression | 0.335 | 0.631 | 0.695 |
| A | K-means | 0.185 | 0.590 | — |
| B | Random Forest | 0.104 | 0.581 | 0.578 |
| B | Logistic Regression | 0.098 | 0.554 | 0.575 |
| B | HistGradientBoosting | 0.094 | 0.567 | 0.586 |
| B | K-means | -0.058 | 0.461 | — |
| B | Neural Network | -0.123 | 0.394 | 0.344 |

### Time-to-eruption regression

The target is standardised, so the RMSE values below are in **scaled target units**, not hours or days.

| Part | Model | Test RMSE | Test R² |
|---|---|---:|---:|
| A | Random Forest | 0.797 | 0.390 |
| A | Neural Network | 0.880 | 0.256 |
| A | K-means | 1.125 | -0.216 |
| B | Neural Network | 0.708 | 0.034 |
| B | Random Forest | 0.720 | approximately 0.000 |
| B | HistGradientBoosting | 0.782 | -0.180 |
| B | K-means | 1.122 | -1.427 |

## Interpretation

- In Part A, the saved run's Random Forest reaches **κ = 0.393** and **R² = 0.390** on the test set. This is directionally consistent with the paper's finding that random forests were competitive, though the metrics are not identical.
- Under the Part B split and training changes, the best classification kappa falls to **0.104** (Random Forest); HistGradientBoosting does not outperform the Random Forest in this saved run.
- In Part B regression, the neural network has the highest R² (**0.034**) among the tested models, while Random Forest is approximately zero. These values indicate little predictive value beyond a mean-based reference on that test set.
- The extension changes several factors simultaneously. The score drop cannot be attributed to block splitting, the additional features, class weighting or threshold tuning alone.

## Important metric note

The notebook's `cls_scores` function calculates `auroc` from binary predictions (`yhat`) and separately calculates `auroc_proba` from model scores/probabilities. Standard AUROC is usually computed from a continuous score or probability, so the two columns answer different questions. When reporting results, identify the column explicitly and compare against the original paper cautiously. The paper reports Random Forest test AUROC around 0.71; the saved replication has probability-based AUROC 0.801, while its hard-label AUROC is 0.646.

## Limitations

- One volcano and a short catalogue ending in 1986
- Episode-derived labels with a 24-hour buffer
- Random assignment of 14-day blocks in Part B rather than strict chronological validation
- Multiple Part B modifications at once, preventing clean attribution of performance changes
- Neural-network overfitting appears in the saved loss curves, especially in Part B
- Time-to-eruption RMSE uses standardised values and must not be interpreted as physical time without inverse-transforming predictions

## Reference

Mullet, B., Singh, A., & Wong, Y.-Q. *Predicting Eruptive Events at Volcanoes from Earthquake Data*. Stanford CS229 Machine Learning Project Report, Fall 2019. Code and data: [bmullet/PEEVED](https://github.com/bmullet/PEEVED).
