# Housing Price Predictor

Predicts residential home sale prices using the [Kaggle House Prices dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) (Ames, Iowa). Trains and compares three regression models — Linear Regression, Random Forest, and XGBoost — with full exploratory data analysis, feature engineering, and hyperparameter tuning.

**Kaggle leaderboard score: 0.14146** (RMSLE) out of 4,500+ teams. Ranked 2731 out of 5023 on first submission. 

---

## Results

| Model | RMSE | R² |
|---|---|---|
| Linear Regression | 0.1400 | 0.87 |
| Random Forest | ~0.1400 | ~0.87 |
| XGBoost (tuned) | 0.1202 | 0.9167 |

XGBoost outperformed both baselines after hyperparameter tuning (learning rate, tree depth, estimator count via 5-fold cross-validation).

---

## What's in this repo

```
housing-price-predictor/
├── data/                  ← not tracked (Kaggle terms)
├── notebooks/
│   └── 01_eda.ipynb       ← full pipeline: EDA → cleaning → modeling
└── outputs/
    └── submission.csv     ← Kaggle submission file
```

---

## Approach

### 1. Exploratory data analysis
- Identified right-skewed `SalePrice` distribution — applied log transform (`np.log1p`) for modeling
- Found and visualized 19 columns with missing data

### 2. Data cleaning
- Filled `NaN` values in 15 categorical columns with `"None"` (no pool, no garage, etc. — missing meant absent, not unknown)
- Used **group imputation** for `LotFrontage` — filled with the median of each neighborhood rather than the global median
- Removed two extreme outliers: large high-quality homes that sold for unusually low prices (likely estate/family sales), which were distorting model training

### 3. Feature engineering
Created 11 new features including:
- `TotalSF` — combined basement + first + second floor square footage
- `QualityXSize` — interaction term multiplying `OverallQual` × `TotalSF` (captures disproportionate premium for large, high-quality homes)
- `HouseAge` and `RemodAge` — derived from sale year and build/remodel year
- `TotalBath`, `TotalPorch`, and binary flags for pool, garage, fireplace, basement

### 4. Modeling
- Label-encoded all categorical features
- 80/20 train/test split with `random_state=42`
- Tuned XGBoost hyperparameters individually using 5-fold cross-validation:
  - `learning_rate`: tested 0.01, 0.05, 0.1 → **0.05**
  - `max_depth`: tested 3, 4, 5, 6 → **3**
  - `n_estimators`: tested 500, 1000, 2000, 3000 → **500**

### 5. Key findings
- `OverallQual` (overall material and finish quality) was by far the most predictive feature — it acts as a single expert summary score that buyers price heavily
- `TotalSF` was second — more space means more money
- `CentralAir` ranked third — in Ames, Iowa, absence of central air signals an older unmodernized home, acting as a proxy for overall update status

---

## Setup

```bash
git clone https://github.com/edub966/housing-price-predictor
cd housing-price-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyterlab
```

Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) and place them in `data/`.

Then open the notebook:
```bash
jupyter lab notebooks/01_eda.ipynb
```

---

## Next Steps

- Better categorical encoding — label encoding is lossy, target encoding would be stronger
- Blending models — averaging XGBoost + Lasso predictions
- More feature engineering — neighborhood median prices, interaction terms


## Stack
Python · pandas · scikit-learn · XGBoost · matplotlib · seaborn
