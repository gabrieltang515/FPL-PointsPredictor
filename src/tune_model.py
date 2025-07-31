# src/tune_model.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump
from lightgbm import LGBMRegressor
from features import load_processed, make_features, get_X_y


def main():
    # 1) Load & featurize
    df   = load_processed("data/processed/player_gw_stats.csv")
    feat = make_features(df, window=3)
    X, y = get_X_y(feat)

    # 2) Time-based split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # 3) Parameter grid
    param_dist = {
        "n_estimators":    [50, 100, 200, 300],
        "max_depth":       [None, 5, 10, 20],
        "min_samples_leaf":[1, 2, 5],
        "max_features":    ["sqrt", "log2", None]
    }

    tscv   = TimeSeriesSplit(n_splits=5)

    # 2) Define LightGBM and its hyperparameter grid
    lgbm = LGBMRegressor(random_state=42)
    param_dist = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [31, 50, 100],
        'max_depth': [-1, 10, 20, 30],
        'min_child_samples': [20, 50, 100]
    }

    # 3) TimeSeries CV and randomized search
    tscv = TimeSeriesSplit(n_splits=5)
    random_search = RandomizedSearchCV(
        estimator=lgbm,
        param_distributions=param_dist,
        n_iter=50,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    best_lgbm = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_mae = -random_search.best_score_
    print("🔎 Best LightGBM hyperparameters:", best_params)
    print("🔎 Best cross‐val MAE:", best_cv_mae)

    # 4) Evaluate on hold‐out set
    preds_test = best_lgbm.predict(X_test)
    test_mae = mean_absolute_error(y_test, preds_test)
    print("📊 LightGBM Test MAE:", test_mae)

    # 5) Save tuned LightGBM model
    os.makedirs("models", exist_ok=True)
    dump(best_lgbm, "models/lgbm_tuned.joblib")
    print("💾 Saved tuned LightGBM to models/lgbm_tuned.joblib")

    print("💾 Saved tuned RF to models/rf_tuned.joblib")

if __name__ == "__main__":
    main()
