# src/model.py
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from src.features import load_processed, make_features, get_X_y

from lightgbm import LGBMRegressor


def train_and_evaluate(model_name: str = "baseline"):
    # 1) Load & featurize
    df = load_processed("data/processed/player_gw_stats.csv")
    feat = make_features(df, window=3)
    X, y = get_X_y(feat)
    
    X_train, X_test, y_train, y_test = train_test_split(
          X, y, test_size=0.2, random_state=42, shuffle=False
      )

      # 3) Instantiate model based on choice
    if model_name == "baseline":
      model = LinearRegression()
    elif model_name == "rf":
        model = RandomForestRegressor(random_state=42)
    elif model_name == "lgbm":
        model = LGBMRegressor(random_state=42)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

      # 4) Train & predict
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    
    # 5) Evaluate
    mae = mean_absolute_error(y_test, preds)
    print(f"{model_name} MAE: {mae:.3f}")
    
    # 6) Save for later
    joblib.dump(model, f"models/{model_name}.joblib")
    print(f"Model saved to models/{model_name}.joblib")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    # run both baseline and a stronger model
    train_and_evaluate("baseline")
    train_and_evaluate("rf")
    train_and_evaluate("lgbm")
