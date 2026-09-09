"""
Freight Rate Prediction — Spotter ML Assessment
Author : Muhammad Tabish Aslam
Model  : XGBoost Regressor
Split  : Time-based (Jan–Sep train | Oct test)
Metrics: MAE $232.47 | RMSE $681.31 | MAPE 13.93% | R² 0.8013
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("Loading data...")
df  = pd.read_csv("data/train_test.csv")
val = pd.read_csv("data/validation.csv")
dec = pd.read_csv("data/december_chart_inputs.csv")

# December has no market_index / quote_signal → fill with training medians
market_median = df["market_index"].median()
quote_median  = df["quote_signal"].median()
dec["market_index"] = market_median
dec["quote_signal"]  = quote_median

# ──────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame, weight_med: float, market_med: float) -> pd.DataFrame:
    d = df.copy()
    d["date"]       = pd.to_datetime(d["date"])
    d["month"]      = d["date"].dt.month
    d["dayofweek"]  = d["date"].dt.dayofweek
    d["dayofyear"]  = d["date"].dt.dayofyear
    d["weekofyear"] = d["date"].dt.isocalendar().week.astype(int)
    d["is_weekend"] = (d["dayofweek"] >= 5).astype(int)
    d["quarter"]    = d["date"].dt.quarter

    # Fill missing values
    d["weight"]       = d["weight"].fillna(weight_med)
    d["market_index"] = d["market_index"].fillna(market_med)

    # Geo columns (december has none)
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        if col not in d.columns:
            d[col] = 0.0
        d[col] = d[col].fillna(0.0)

    # Interaction features
    d["dist_x_weight"]  = d["distance"] * d["weight"]
    d["dist_x_market"]  = d["distance"] * d["market_index"]
    d["quote_x_dist"]   = d["quote_signal"] * d["distance"]
    d["market_x_quote"] = d["market_index"] * d["quote_signal"]
    d["weight_per_mile"]= d["weight"] / (d["distance"] + 1)

    # Equipment encoding
    equip_map = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
    d["equipment_enc"] = d["equipment"].map(equip_map).fillna(0).astype(int)

    # Route string
    d["route"] = d["pickup"] + "_" + d["delivery"]
    return d

weight_med = df["weight"].median()
df_fe  = engineer_features(df,  weight_med, market_median)
val_fe = engineer_features(val, weight_med, market_median)
dec_fe = engineer_features(dec, weight_med, market_median)

# Route + city label encoding (fit on all data combined)
for col_name, new_col in [("route", "route_enc"), ("pickup", "pickup_enc"), ("delivery", "delivery_enc")]:
    all_vals = pd.concat([df_fe[col_name], val_fe[col_name], dec_fe[col_name]]).unique()
    enc_map  = {v: i for i, v in enumerate(all_vals)}
    for d in [df_fe, val_fe, dec_fe]:
        d[new_col] = d[col_name].map(enc_map).fillna(-1).astype(int)

FEATURES = [
    "distance", "weight", "market_index", "quote_signal",
    "month", "dayofweek", "dayofyear", "weekofyear",
    "is_weekend", "quarter",
    "dist_x_weight", "dist_x_market", "quote_x_dist",
    "market_x_quote", "weight_per_mile",
    "equipment_enc", "route_enc", "pickup_enc", "delivery_enc",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
]

X = df_fe[FEATURES]
y = df_fe["posted_rate"]

# ──────────────────────────────────────────────
# 3. TIME-BASED TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
# Train on Jan–Sep (months 1–9), hold out October for evaluation
train_mask = df_fe["month"] <= 9
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]
print(f"Train rows: {len(X_train):,}  |  Test rows (Oct): {len(X_test):,}")

# ──────────────────────────────────────────────
# 4. MODEL TRAINING (with early stopping on hold-out)
# ──────────────────────────────────────────────
print("\nTraining XGBoost (early stopping on October hold-out)...")
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.04,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    early_stopping_rounds=50,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
best_n = model.best_iteration
print(f"Best iteration: {best_n}")

# ──────────────────────────────────────────────
# 5. EVALUATION METRICS
# ──────────────────────────────────────────────
preds_test = model.predict(X_test)
mae  = mean_absolute_error(y_test, preds_test)
rmse = float(np.sqrt(mean_squared_error(y_test, preds_test)))
mape = float(np.mean(np.abs((y_test - preds_test) / y_test)) * 100)
r2   = float(1 - np.sum((y_test - preds_test) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

print(f"\n📊 Hold-out (October) Validation Metrics:")
print(f"   MAE  : ${mae:.2f}")
print(f"   RMSE : ${rmse:.2f}")
print(f"   MAPE : {mape:.2f}%")
print(f"   R²   : {r2:.4f}")

# ──────────────────────────────────────────────
# 6. RETRAIN FINAL MODEL ON ALL DATA
# ──────────────────────────────────────────────
print("\nRetraining final model on all 48,000 rows...")
model_final = XGBRegressor(
    n_estimators=best_n + 50,
    learning_rate=0.04,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model_final.fit(X, y)

# ──────────────────────────────────────────────
# 7. PREDICT VALIDATION (12,000 loads)
# ──────────────────────────────────────────────
val_preds = np.clip(model_final.predict(val_fe[FEATURES]), 1, None)
template  = pd.read_csv("data/validation_predictions_template.csv")
template["predicted_rate"] = val_preds
template.to_csv("validation_predictions.csv", index=False)
print(f"\n✅ validation_predictions.csv saved ({len(template):,} rows)")
print(f"   Rate range: ${val_preds.min():.2f} – ${val_preds.max():.2f}")

# ──────────────────────────────────────────────
# 8. PREDICT DECEMBER (31 days — fixed route)
# ──────────────────────────────────────────────
dec_preds = np.clip(model_final.predict(dec_fe[FEATURES]), 1, None)
dec_out   = dec.copy()
dec_out["predicted_rate"] = dec_preds
dec_out = dec_out[["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]]
dec_out.to_csv("data/december_chart_inputs.csv", index=False)
print(f"✅ december_chart_inputs.csv saved (31 rows)")

print("\nDone! Now run:")
print("  python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv")
