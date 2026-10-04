# Freight Rate Prediction — Spotter ML Engineer Assessment

Predicts `posted_rate` (USD) for truck loads.

All of the work is in **`freight_model_two.ipynb`**. It compares 10 models, including XGBoost, LightGBM and a Ridge + LightGBM hybrid, in these steps:

1. **Load the data**
2. **EDA:** data types, summary statistics, duplicates, missing values, negative values, clipped values, the target, market signals, the time trend, extreme labels, and differences between train and validation, with evidence for each cleaning decision
3. **Cleaning & feature engineering**
4. **Modelling:** 10 models, grid search with expanding-window time-series CV, MAE / RMSE / R² metrics, and keep vs drop of extreme labels
5. **Evaluation** on an untouched Sep–Oct holdout
6. **Final predictions:** re-train the chosen model on Jan–Oct and write `validation_predictions.csv` and `data/december_chart_inputs.csv`

## Run

```bash
python -m pip install -r requirements.txt
jupyter notebook freight_model_two.ipynb       # or: jupyter nbconvert --to notebook --execute freight_model_two.ipynb
```

Running the notebook writes both prediction files. Then validate them and create the December chart with the provided scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

The chart is saved to `scorer_results/candidate_december.png`.

## Layout

```
data/
  train_test.csv                       labelled loads, Jan–Oct 2025
  validation.csv                       loads to predict, Nov–Dec 2025
  validation_predictions_template.csv
  december_chart_inputs.csv            the 31 December chart rows (predicted_rate filled in)
freight_model_two.ipynb                analysis, models and final predictions
validation_predictions.csv             final predictions for the 12,000 validation loads
score.py                               Spotter's provided scorer
requirements.txt
```

## Current result (holdout Sep–Oct 2025)

| Model | MAE, all rows | MAE, normal rows |
|---|---|---|
| XGBoost | $106 | $50 |
| **Hybrid: Ridge + LightGBM on residuals** | **$93** | **$37** |

"Normal rows" excludes the ~1.4% of labels flagged as extreme price errors (see EDA §2.9).
