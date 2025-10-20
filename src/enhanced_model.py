# src/enhanced_model.py
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

from src.features import load_processed, make_features, get_X_y


def create_enhanced_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Create enhanced features beyond the basic rolling features.
    """
    df = df.copy()
    
    # Sort by player and gameweek
    df = df.sort_values(["player_id", "gameweek"])
    
    # Enhanced rolling features with different windows
    windows = [2, 3, 5, 10]
    for w in windows:
        # Points rolling features
        df[f"points_roll{w}_mean"] = (
            df.groupby("player_id")["total_points"]
              .shift(1)
              .rolling(w, min_periods=1)
              .mean()
        )
        df[f"points_roll{w}_std"] = (
            df.groupby("player_id")["total_points"]
              .shift(1)
              .rolling(w, min_periods=1)
              .std()
        )
        df[f"points_roll{w}_max"] = (
            df.groupby("player_id")["total_points"]
              .shift(1)
              .rolling(w, min_periods=1)
              .max()
        )
        
        # Minutes rolling features
        df[f"minutes_roll{w}_mean"] = (
            df.groupby("player_id")["minutes"]
              .shift(1)
              .rolling(w, min_periods=1)
              .mean()
        )
        df[f"minutes_roll{w}_sum"] = (
            df.groupby("player_id")["minutes"]
              .shift(1)
              .rolling(w, min_periods=1)
              .sum()
        )
        
        # Goals and assists rolling features
        for col in ["goals", "assists"]:
            if col in df.columns:
                df[f"{col}_roll{w}_sum"] = (
                    df.groupby("player_id")[col]
                      .shift(1)
                      .rolling(w, min_periods=1)
                      .sum()
                )
                df[f"{col}_roll{w}_mean"] = (
                    df.groupby("player_id")[col]
                      .shift(1)
                      .rolling(w, min_periods=1)
                      .mean()
                )
        
        # Advanced metrics rolling features
        for col in ["xG", "xA", "key_passes", "npxG"]:
            if col in df.columns:
                df[f"{col}_roll{w}_mean"] = (
                    df.groupby("player_id")[col]
                      .shift(1)
                      .rolling(w, min_periods=1)
                      .mean()
                )
                df[f"{col}_roll{w}_sum"] = (
                    df.groupby("player_id")[col]
                      .shift(1)
                      .rolling(w, min_periods=1)
                      .sum()
                )
    
    # Form indicators
    df["form_3gw"] = df["points_roll3_mean"] - df["points_roll10_mean"]
    df["form_5gw"] = df["points_roll5_mean"] - df["points_roll10_mean"]
    
    # Consistency indicators
    df["consistency_3gw"] = df["points_roll3_std"] / (df["points_roll3_mean"] + 1e-8)
    df["consistency_5gw"] = df["points_roll5_std"] / (df["points_roll5_mean"] + 1e-8)
    
    # Minutes efficiency
    df["points_per_minute"] = df["total_points"] / (df["minutes"] + 1e-8)
    df["points_per_minute_roll3"] = (
        df.groupby("player_id")["points_per_minute"]
          .shift(1)
          .rolling(3, min_periods=1)
          .mean()
    )
    
    # Goal and assist efficiency
    if "goals" in df.columns and "minutes" in df.columns:
        df["goals_per_minute"] = df["goals"] / (df["minutes"] + 1e-8)
        df["goals_per_minute_roll3"] = (
            df.groupby("player_id")["goals_per_minute"]
              .shift(1)
              .rolling(3, min_periods=1)
              .mean()
        )
    
    if "assists" in df.columns and "minutes" in df.columns:
        df["assists_per_minute"] = df["assists"] / (df["minutes"] + 1e-8)
        df["assists_per_minute_roll3"] = (
            df.groupby("player_id")["assists_per_minute"]
              .shift(1)
              .rolling(3, min_periods=1)
              .mean()
        )
    
    # Fixture difficulty interactions
    if "fixture_difficulty" in df.columns:
        df["fixture_difficulty_squared"] = df["fixture_difficulty"] ** 2
        df["points_roll3_fixture_weighted"] = df["points_roll3_mean"] * (5 - df["fixture_difficulty"])
    
    # Position-specific features
    for pos in ["pos_FWD", "pos_MID", "pos_DEF", "pos_GKP"]:
        if pos in df.columns:
            df[f"{pos}_points_roll3"] = df[pos] * df["points_roll3_mean"]
            df[f"{pos}_minutes_roll3"] = df[pos] * df["minutes_roll3_mean"]
    
    return df


def create_ensemble_model():
    """
    Create an ensemble of LightGBM models with different hyperparameters.
    """
    # Model 1: Conservative (lower learning rate, more trees)
    lgbm1 = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=8,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    # Model 2: Aggressive (higher learning rate, fewer trees)
    lgbm2 = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=100,
        max_depth=12,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1
    )
    
    # Model 3: Balanced
    lgbm3 = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=50,
        max_depth=10,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    
    # Create ensemble
    ensemble = VotingRegressor([
        ('lgbm1', lgbm1),
        ('lgbm2', lgbm2),
        ('lgbm3', lgbm3)
    ], weights=[0.4, 0.3, 0.3])
    
    return ensemble


def hyperparameter_tuning(X_train, y_train):
    """
    Perform advanced hyperparameter tuning with more comprehensive search space.
    """
    # Enhanced parameter grid
    param_dist = {
        'n_estimators': [300, 500, 800, 1000],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'num_leaves': [31, 50, 75, 100, 150],
        'max_depth': [6, 8, 10, 12, 15],
        'min_child_samples': [20, 30, 50, 100],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [0, 0.01, 0.1, 1],
        'min_split_gain': [0, 0.01, 0.1]
    }
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Base model
    base_model = LGBMRegressor(random_state=42, verbose=-1)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=100,  # Increased iterations
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    print("🔍 Performing hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_cv_mae = -random_search.best_score_
    
    print(f"🔎 Best hyperparameters: {best_params}")
    print(f"🔎 Best CV MAE: {best_cv_mae:.4f}")
    
    return best_model


def train_enhanced_model():
    """
    Train an enhanced LightGBM model with better features and tuning.
    """
    print("🚀 Training Enhanced LightGBM Model")
    print("=" * 50)
    
    # 1. Load and create enhanced features
    print("📊 Loading data and creating enhanced features...")
    df = load_processed("data/processed/player_gw_stats.csv")
    feat = make_features(df, window=3)
    feat_enhanced = create_enhanced_features(feat, window=3)
    X, y = get_X_y(feat_enhanced)
    
    print(f"📈 Enhanced features shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    
    # 2. Time-based split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"📚 Training set size: {X_train.shape[0]}")
    print(f"🧪 Test set size: {X_test.shape[0]}")
    
    # 3. Train multiple models
    models = {}
    
    # Model 1: Hyperparameter tuned
    print("\n🎯 Training hyperparameter-tuned model...")
    models['tuned'] = hyperparameter_tuning(X_train, y_train)
    
    # Model 2: Ensemble
    print("\n🎯 Training ensemble model...")
    models['ensemble'] = create_ensemble_model()
    models['ensemble'].fit(X_train, y_train)
    
    # Model 3: Best single model from tuning
    print("\n🎯 Training best single model...")
    best_single = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=75,
        max_depth=10,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.01,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1
    )
    models['best_single'] = best_single
    models['best_single'].fit(X_train, y_train)
    
    # 4. Evaluate all models
    print("\n📊 Evaluating models...")
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred
        }
        
        print(f"{name.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # 5. Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
    best_model = models[best_model_name]
    best_results = results[best_model_name]
    
    print(f"\n🏆 Best model: {best_model_name.upper()}")
    print(f"📈 MAE: {best_results['mae']:.4f}")
    print(f"📈 RMSE: {best_results['rmse']:.4f}")
    print(f"📈 R²: {best_results['r2']:.4f}")
    
    # 6. Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/lgbm_enhanced.joblib")
    print(f"💾 Saved enhanced model to models/lgbm_enhanced.joblib")
    
    # 7. Save results for comparison
    comparison_data = {
        'model_name': list(results.keys()),
        'mae': [results[name]['mae'] for name in results.keys()],
        'rmse': [results[name]['rmse'] for name in results.keys()],
        'r2': [results[name]['r2'] for name in results.keys()]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("models/enhanced_model_comparison.csv", index=False)
    print("💾 Saved comparison results to models/enhanced_model_comparison.csv")
    
    return best_model, best_results, results


def train_enhanced_model_lightweight():
    """
    Train a lightweight version of the enhanced model with reduced computational load.
    """
    print("🚀 Training Lightweight Enhanced LightGBM Model")
    print("=" * 50)
    
    # 1. Load and create enhanced features
    print("📊 Loading data and creating enhanced features...")
    df = load_processed("data/processed/player_gw_stats.csv")
    feat = make_features(df, window=3)
    feat_enhanced = create_enhanced_features(feat, window=3)
    X, y = get_X_y(feat_enhanced)
    
    print(f"📈 Enhanced features shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    
    # 2. Time-based split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"📚 Training set size: {X_train.shape[0]}")
    print(f"🧪 Test set size: {X_test.shape[0]}")
    
    # 3. Train lightweight models
    models = {}
    
    # Model 1: Quick hyperparameter search (reduced iterations)
    print("\n🎯 Training lightweight hyperparameter-tuned model...")
    param_dist_light = {
        'n_estimators': [300, 500, 800],
        'learning_rate': [0.05, 0.1, 0.15],
        'num_leaves': [31, 50, 75],
        'max_depth': [8, 10, 12],
        'min_child_samples': [30, 50],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits
    base_model = LGBMRegressor(random_state=42, verbose=-1)
    
    random_search_light = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist_light,
        n_iter=20,  # Reduced iterations
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=1,  # Single job to reduce heat
        verbose=1
    )
    
    random_search_light.fit(X_train, y_train)
    models['tuned_light'] = random_search_light.best_estimator_
    
    # Model 2: Simple ensemble (fewer trees)
    print("\n🎯 Training lightweight ensemble model...")
    lgbm1_light = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=50,
        max_depth=8,
        random_state=42,
        verbose=-1
    )
    
    lgbm2_light = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        num_leaves=75,
        max_depth=10,
        random_state=42,
        verbose=-1
    )
    
    ensemble_light = VotingRegressor([
        ('lgbm1', lgbm1_light),
        ('lgbm2', lgbm2_light)
    ], weights=[0.6, 0.4])
    
    models['ensemble_light'] = ensemble_light
    models['ensemble_light'].fit(X_train, y_train)
    
    # Model 3: Best single model (pre-tuned)
    print("\n🎯 Training best single model...")
    best_single_light = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=50,
        max_depth=8,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    models['best_single_light'] = best_single_light
    models['best_single_light'].fit(X_train, y_train)
    
    # 4. Evaluate all models
    print("\n📊 Evaluating models...")
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred
        }
        
        print(f"{name.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # 5. Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
    best_model = models[best_model_name]
    best_results = results[best_model_name]
    
    print(f"\n🏆 Best lightweight model: {best_model_name.upper()}")
    print(f"📈 MAE: {best_results['mae']:.4f}")
    print(f"📈 RMSE: {best_results['rmse']:.4f}")
    print(f"📈 R²: {best_results['r2']:.4f}")
    
    # 6. Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/lgbm_enhanced_lightweight.joblib")
    print(f"💾 Saved lightweight enhanced model to models/lgbm_enhanced_lightweight.joblib")
    
    return best_model, best_results, results


def train_enhanced_model_cloud_optimized():
    """
    Cloud-optimized version with better resource management.
    """
    print("☁️ Training Cloud-Optimized Enhanced LightGBM Model")
    print("=" * 50)
    
    # 1. Load and create enhanced features
    print("📊 Loading data and creating enhanced features...")
    df = load_processed("data/processed/player_gw_stats.csv")
    feat = make_features(df, window=3)
    feat_enhanced = create_enhanced_features(feat, window=3)
    X, y = get_X_y(feat_enhanced)
    
    print(f"📈 Enhanced features shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    
    # 2. Time-based split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"📚 Training set size: {X_train.shape[0]}")
    print(f"🧪 Test set size: {X_test.shape[0]}")
    
    # 3. Cloud-optimized hyperparameter tuning
    print("\n🎯 Training cloud-optimized hyperparameter-tuned model...")
    
    # More aggressive parameter space for cloud
    param_dist_cloud = {
        'n_estimators': [500, 800, 1000, 1500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
        'num_leaves': [31, 50, 75, 100, 150, 200],
        'max_depth': [6, 8, 10, 12, 15, 20],
        'min_child_samples': [20, 30, 50, 100, 200],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1, 10],
        'reg_lambda': [0, 0.01, 0.1, 1, 10],
        'min_split_gain': [0, 0.01, 0.1, 1]
    }
    
    # More CV splits for better validation
    tscv = TimeSeriesSplit(n_splits=8)
    base_model = LGBMRegressor(random_state=42, verbose=-1)
    
    # Use all available cores for cloud
    random_search_cloud = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist_cloud,
        n_iter=200,  # More iterations for cloud
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,  # Use all cores
        verbose=2
    )
    
    random_search_cloud.fit(X_train, y_train)
    best_cloud_model = random_search_cloud.best_estimator_
    
    # 4. Advanced ensemble with more models
    print("\n🎯 Training advanced ensemble model...")
    
    # More diverse ensemble
    lgbm1_cloud = LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.01,
        num_leaves=100,
        max_depth=12,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    lgbm2_cloud = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.1,
        num_leaves=200,
        max_depth=15,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1
    )
    
    lgbm3_cloud = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=150,
        max_depth=10,
        min_child_samples=30,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    
    lgbm4_cloud = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=75,
        max_depth=18,
        min_child_samples=40,
        subsample=0.75,
        colsample_bytree=0.75,
        random_state=42,
        verbose=-1
    )
    
    ensemble_cloud = VotingRegressor([
        ('lgbm1', lgbm1_cloud),
        ('lgbm2', lgbm2_cloud),
        ('lgbm3', lgbm3_cloud),
        ('lgbm4', lgbm4_cloud)
    ], weights=[0.3, 0.25, 0.25, 0.2])
    
    ensemble_cloud.fit(X_train, y_train)
    
    # 5. Evaluate models
    models_cloud = {
        'tuned_cloud': best_cloud_model,
        'ensemble_cloud': ensemble_cloud
    }
    
    print("\n📊 Evaluating cloud-optimized models...")
    results_cloud = {}
    
    for name, model in models_cloud.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results_cloud[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred
        }
        
        print(f"{name.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # 6. Find best model
    best_model_name = min(results_cloud.keys(), key=lambda x: results_cloud[x]['mae'])
    best_model = models_cloud[best_model_name]
    best_results = results_cloud[best_model_name]
    
    print(f"\n🏆 Best cloud-optimized model: {best_model_name.upper()}")
    print(f"📈 MAE: {best_results['mae']:.4f}")
    print(f"📈 RMSE: {best_results['rmse']:.4f}")
    print(f"📈 R²: {best_results['r2']:.4f}")
    
    # 7. Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/lgbm_enhanced_cloud.joblib")
    print(f"💾 Saved cloud-optimized model to models/lgbm_enhanced_cloud.joblib")
    
    return best_model, best_results, results_cloud


def train_enhanced_model_local_optimized():
    """
    Local-optimized version with better resource management for laptops.
    """
    print("💻 Training Local-Optimized Enhanced LightGBM Model")
    print("=" * 50)
    
    # 1. Load and create enhanced features
    print("📊 Loading data and creating enhanced features...")
    df = load_processed("data/processed/player_gw_stats.csv")
    feat = make_features(df, window=3)
    feat_enhanced = create_enhanced_features(feat, window=3)
    X, y = get_X_y(feat_enhanced)
    
    print(f"📈 Enhanced features shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    
    # 2. Time-based split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"📚 Training set size: {X_train.shape[0]}")
    print(f"🧪 Test set size: {X_test.shape[0]}")
    
    # 3. Local-optimized training (reduced but still comprehensive)
    models = {}
    
    # Model 1: Moderate hyperparameter search
    print("\n🎯 Training local-optimized hyperparameter-tuned model...")
    param_dist_local = {
        'n_estimators': [500, 800, 1000],
        'learning_rate': [0.03, 0.05, 0.1, 0.15],
        'num_leaves': [31, 50, 75, 100],
        'max_depth': [8, 10, 12, 15],
        'min_child_samples': [30, 50, 100],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [0, 0.01, 0.1]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    base_model = LGBMRegressor(random_state=42, verbose=-1)
    
    # Use half the cores to prevent overheating
    import multiprocessing
    n_jobs = max(1, multiprocessing.cpu_count() // 2)
    
    random_search_local = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist_local,
        n_iter=50,  # Moderate iterations
        cv=tscv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=n_jobs,  # Use half cores
        verbose=1
    )
    
    random_search_local.fit(X_train, y_train)
    models['tuned_local'] = random_search_local.best_estimator_
    
    # Model 2: Balanced ensemble
    print("\n🎯 Training local-optimized ensemble model...")
    lgbm1_local = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=75,
        max_depth=10,
        random_state=42,
        verbose=-1
    )
    
    lgbm2_local = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.1,
        num_leaves=100,
        max_depth=12,
        random_state=42,
        verbose=-1
    )
    
    ensemble_local = VotingRegressor([
        ('lgbm1', lgbm1_local),
        ('lgbm2', lgbm2_local)
    ], weights=[0.6, 0.4])
    
    models['ensemble_local'] = ensemble_local
    models['ensemble_local'].fit(X_train, y_train)
    
    # 4. Evaluate models
    print("\n📊 Evaluating local-optimized models...")
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred
        }
        
        print(f"{name.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # 5. Find best model
    best_model_name = min(results.keys(), key=lambda x: results[x]['mae'])
    best_model = models[best_model_name]
    best_results = results[best_model_name]
    
    print(f"\n🏆 Best local-optimized model: {best_model_name.upper()}")
    print(f"📈 MAE: {best_results['mae']:.4f}")
    print(f"📈 RMSE: {best_results['rmse']:.4f}")
    print(f"📈 R²: {best_results['r2']:.4f}")
    
    # 6. Save best model
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/lgbm_enhanced_local.joblib")
    print(f"💾 Saved local-optimized model to models/lgbm_enhanced_local.joblib")
    
    return best_model, best_results, results


def analyze_feature_importance(model, X_train, top_n=20):
    """
    Analyze feature importance for the enhanced model.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train.columns
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔍 Top {top_n} Most Important Features:")
        print("=" * 60)
        for idx, row in importance_df.head(top_n).iterrows():
            print(f"{row['feature']:<35} {row['importance']:>8.2f}")
        
        return importance_df
    else:
        print("⚠️ Model doesn't have feature_importances_ attribute")
        return None


if __name__ == "__main__":
    import sys
    
    # Check training mode
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "--lightweight":
            print("🔄 Running in lightweight mode to reduce computational load...")
            best_model, best_results, all_results = train_enhanced_model_lightweight()
        elif mode == "--cloud":
            print("🔄 Running in cloud-optimized mode...")
            best_model, best_results, all_results = train_enhanced_model_cloud_optimized()
        elif mode == "--local":
            print("🔄 Running in local-optimized mode...")
            best_model, best_results, all_results = train_enhanced_model_local_optimized()
        else:
            print(f"❌ Unknown mode: {mode}")
            print("Available modes: --lightweight, --cloud, --local")
            sys.exit(1)
    else:
        print("🔄 Running full enhanced model training (this will be computationally intensive)...")
        print("💡 Available modes:")
        print("   --lightweight: Fast training for laptops")
        print("   --local: Optimized for local machines")
        print("   --cloud: Full power for cloud/GPU environments")
        best_model, best_results, all_results = train_enhanced_model()
    
    # Load data for feature importance analysis
    df = load_processed("data/processed/player_gw_stats.csv")
    feat = make_features(df, window=3)
    feat_enhanced = create_enhanced_features(feat, window=3)
    X, y = get_X_y(feat_enhanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Analyze feature importance
    importance_df = analyze_feature_importance(best_model, X_train)
    
    print(f"\n🎉 Enhanced model training complete!")
    print(f"📊 Best MAE: {best_results['mae']:.4f}")
    print(f"📊 Best R²: {best_results['r2']:.4f}") 