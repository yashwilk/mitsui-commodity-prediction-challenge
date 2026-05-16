## Feature 1 (Log Return)

Percentage Return = ((Pt - Pt-1) / Pt-1) × 100
Example: ((110 - 100) / 100) × 100 = 10%

Log Return = ln(Pt / Pt-1)
Example: ln(110 / 100) = 0.0953

Log returns are preferred because they are additive over time and work better for financial analysis and forecasting.

log_return_1d  = log(price today) - log(price 1 day ago)
log_return_3d  = log(price today) - log(price 3 days ago)
log_return_5d  = log(price today) - log(price 5 days ago)
log_return_10d = log(price today) - log(price 10 days ago)

All of these are shifted by 1 day before being used as features. So the feature value on date_id=5 is actually the log return computed up to date_id=4.

---

## Feature 2 (Rolling Mean)

A rolling mean is the average of the log return over the past N days. It smooths out daily noise and captures the overall trend direction.

rolling_mean_5d = average of last 5 daily log returns

date_id=4: average of returns on days 0,1,2,3,4
         = average of (NaN, +0.00621, -0.00867, +0.00467, +0.01770)
         = +0.00498  ← copper has been trending slightly up

rolling_mean_5d   = average daily return over past 5 days
rolling_mean_10d  = average daily return over past 10 days
rolling_mean_20d  = average daily return over past 20 days

Again shifted by 1 day to prevent leakage.

---

## Feature 3 (Rolling Standard Deviation / Volatility)

Standard deviation measures how much the daily returns have been varying. High std means volatile market. Low std means calm market.

rolling_std_5d = std of last 5 daily log returns

date_id=4: std of (NaN, +0.00621, -0.00867, +0.00467, +0.01770)
         = 0.00893  ← moderate volatility

date_id=9: if all returns were +0.00100 every day
         std = 0.00000  ← no volatility at all

rolling_std_5d   = volatility over past 5 days
rolling_std_10d  = volatility over past 10 days
rolling_std_20d  = volatility over past 20 days

---

## Feature 4 (Price Difference)

A price difference is simply today's price minus the price N days ago in raw terms.

diff_1d = price today - price 1 day ago
diff_3d = price today - price 3 days ago
diff_5d = price today - price 5 days ago

---

## Feature 5 (Lagged Price Levels)

This is simply the raw price from N days ago used directly as a feature.

price_lag1 = price 1 day ago
price_lag3 = price 3 days ago
price_lag5 = price 5 days ago

date_id=5:
  price_lag1 = LME_CA_Close on date_id=4 = 9180.00
  price_lag3 = LME_CA_Close on date_id=2 = 8978.00
  price_lag5 = LME_CA_Close on date_id=0 = 9000.00

---

## Feature 6 (Spread Return — for spread targets only)

For targets that are asset A minus asset B, we create all the above features for both assets separately. But we also create one additional feature — the spread between their recent returns:

spread_return_1d = log_return_1d of asset A - log_return_1d of asset B
spread_return_5d = log_return_5d of asset A - log_return_5d of asset B

---

## Preprocessing Pipeline

**log1p transform** — applies log(1 + x) to compress extreme values. A feature value of 10,000 becomes about 9.2. A value of 0.001 stays near 0.001. This reduces the influence of outliers.

**Replace infinities with NaN** — after log transforms and divisions, infinite values can appear. They are replaced with NaN before imputation.

**Median imputation** — all remaining NaN values are replaced with the median of that feature column. Median is more robust than mean because it is not affected by extreme outliers.

**StandardScaler** — scales all features to zero mean and unit variance. After scaling, a value of +1 means "1 standard deviation above average" regardless of what the original unit was. This helps the model treat all features equally.

---

## Evaluation Metric — Spearman Sharpe

It measures how consistently your model correctly ranks assets from best to worst performance, every single day.

date_id=0:
  actual ranks    : [1, 2, 3, ...]  (across targets: target_0, target_1, ...)
  predicted ranks : [1, 2, 3, ...]
  daily_corrs list → corr = 0.85

Final grade = mean / std of all daily correlation scores

---

## Model 1 
— Baseline1:Predict Zero for Everything

Predicting zero is the simplest possible prediction — it requires no data, no features, no training. It sets the absolute floor. Any real model must beat this score.

### Baseline 2: Predict Yesterday's Return

Baseline 2 predicts that tomorrow's return will equal today's return. This captures the simplest possible momentum signal — if copper rose today, predict it will rise tomorrow. No features, no training, just one shift of the actual labels. This is the smarter baseline — it uses real information from the data.

---

train.csv row date_id=5:
"On day 5, copper was 9000, gold was 1820, USD/JPY was 131..."

train_labels.csv row date_id=5:
"On day 5, copper's next-day return was +0.62%, aluminium's was +0.28%..."

target_pairs.csv says:
  target_408 | lag=4 | LME_AH_Close - FX_ZARCHF

This means train_labels.csv computed target_408 as:
  [log(LME_AH_Close day d+4) - log(LME_AH_Close day d)]
  MINUS
  [log(FX_ZARCHF day d+4) - log(FX_ZARCHF day d)]

And train.csv provides:
  LME_AH_Close on day d → plain raw price, belongs to that day only
  FX_ZARCHF on day d    → plain raw price, belongs to that day only

---

## Model Scores (Spearman Sharpe)

| Model                          |               Score |
|--------------------------------|---------------------|
| Random predictions             |               -0.01 |
| Baseline 1 — predict zero      |                 NaN |
| Baseline 2 — predict yesterday |              2.3343 |
| LightGBM CV (single target)    |                1.22 |
| LightGBM lag 1                 |              5.4711 |
| LightGBM lag 2                 |              4.3529 |
| LightGBM lag 3                 |              4.9086 |
| LightGBM lag 4                 |              4.6129 |
| LightGBM overall               |              4.8364 |
| Stacking ensemble lag 1        |              5.7926 |
| Stacking ensemble lag 2        |              4.6785 |
| Stacking ensemble lag 3        |              4.9842 |
| Stacking ensemble lag 4        |              4.8601 |
| Stacking ensemble overall      |              5.0789 |

---

## Stacking Ensemble Architecture

```
Original features (34 columns)
       ↓
  ┌────────────────────────────┐
  │ LightGBM  → pred_lgbm     │  Level 0
  │ RandomForest → pred_rf    │
  │ XGBoost   → pred_xgb      │
  └────────────────────────────┘
       ↓
  [pred_lgbm, pred_rf, pred_xgb]  ← 3 columns
       ↓
  XGBoost meta-model              Level 1
       ↓
  final prediction
```

### Level 0 — Base Learners

**LightGBM** (`pred_lgbm`)
A gradient boosting framework that builds trees leaf-wise rather than depth-wise. It is fast and memory-efficient on tabular data. It handles the 34 engineered features well and captures non-linear interactions between rolling returns, volatility, and price lags. `num_leaves=31`, `learning_rate=0.05`, `n_estimators=100`.

**Random Forest** (`pred_rf`)
An ensemble of independently trained decision trees, each trained on a random subset of features and bootstrap samples of data. Unlike boosting, trees are not corrected iteratively — diversity comes from randomness. It provides a structurally different signal to the meta-model, reducing overfitting risk in the stack. `n_estimators=100`, `max_depth=6`, `min_samples_leaf=5`.

**XGBoost** (`pred_xgb`)
A regularised gradient boosting implementation that builds trees depth-wise. It adds L1/L2 regularisation terms to the loss function, making it more robust to outliers than LightGBM on small datasets. Configured shallower (`max_depth=4`) than LightGBM to complement it rather than duplicate it. `n_estimators=100`, `learning_rate=0.05`.

### Level 1 — Meta-Model

**XGBoost meta-model**
Takes the three base-learner predictions (`pred_lgbm`, `pred_rf`, `pred_xgb`) as its only input features and learns how to optimally blend them. Because the meta-model sees only 3 columns, it is kept deliberately small (`n_estimators=50`, `max_depth=3`) to avoid overfitting. It learns when to trust each base learner — for example, upweighting LightGBM when volatility is high, or Random Forest when the signal is noisy.