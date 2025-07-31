# src/visualize_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features import load_processed, make_features, get_X_y

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_model_and_data(model_name="lgbm"):
    """Load the trained model and prepare data for visualization."""
    # Load model
    model = joblib.load(f"models/{model_name}.joblib")
    
    # Load and prepare data
    df = load_processed()
    feat = make_features(df, window=3)
    X, y = get_X_y(feat)
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred, X_train, y_train

def plot_prediction_vs_actual(y_test, y_pred, model_name="LightGBM"):
    """Plot predicted vs actual FPL points."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    ax1.scatter(y_test, y_pred, alpha=0.6, s=20)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual FPL Points')
    ax1.set_ylabel('Predicted FPL Points')
    ax1.set_title(f'{model_name}: Predicted vs Actual FPL Points')
    ax1.grid(True, alpha=0.3)
    
    # Add R² and MAE to plot
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}\nMAE = {mae:.3f}', 
              transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Residual plot
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.6, s=20)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted FPL Points')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name.lower()}_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_distribution(y_test, y_pred, model_name="LightGBM"):
    """Plot the distribution of prediction errors."""
    errors = y_test - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of errors
    ax1.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Prediction Error (Actual - Predicted)')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{model_name}: Distribution of Prediction Errors')
    ax1.grid(True, alpha=0.3)
    
    # Box plot of errors
    ax2.boxplot(errors, vert=False)
    ax2.set_xlabel('Prediction Error')
    ax2.set_title('Error Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name.lower()}_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print error statistics
    print(f"\n{model_name} Error Statistics:")
    print(f"Mean Error: {errors.mean():.3f}")
    print(f"Std Error: {errors.std():.3f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")

def plot_feature_importance(model, X_train, model_name="LightGBM"):
    """Plot feature importance for the model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_train.columns
        
        # Create DataFrame for easier plotting
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True)
        
        # Plot top 15 features
        top_features = importance_df.tail(15)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name}: Top 15 Most Important Features')
        plt.grid(True, alpha=0.3)
        
        # Color bars based on importance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.lower()}_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print(f"\n{model_name} Top 10 Most Important Features:")
        for idx, row in importance_df.tail(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

def plot_performance_by_points_range(y_test, y_pred, model_name="LightGBM"):
    """Plot how well the model performs across different ranges of FPL points."""
    # Create bins for different point ranges
    bins = [0, 2, 4, 6, 8, 10, 15, 20, 30, 50, 100]
    labels = ['0-2', '2-4', '4-6', '6-8', '8-10', '10-15', '15-20', '20-30', '30-50', '50+']
    
    y_test_binned = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)
    
    # Calculate MAE for each bin
    mae_by_bin = []
    bin_counts = []
    
    for label in labels:
        mask = y_test_binned == label
        if mask.sum() > 0:
            mae = mean_absolute_error(y_test[mask], y_pred[mask])
            mae_by_bin.append(mae)
            bin_counts.append(mask.sum())
        else:
            mae_by_bin.append(0)
            bin_counts.append(0)
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE by point range
    bars1 = ax1.bar(labels, mae_by_bin, alpha=0.7)
    ax1.set_xlabel('FPL Points Range')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title(f'{model_name}: Prediction Error by Points Range')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Sample count by point range
    bars2 = ax2.bar(labels, bin_counts, alpha=0.7, color='orange')
    ax2.set_xlabel('FPL Points Range')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Distribution by Points Range')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name.lower()}_performance_by_range.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(y_test, y_pred, model_name="LightGBM"):
    """Create a comprehensive summary report."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name} MODEL PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"Root Mean Square Error (RMSE): {rmse:.3f}")
    print(f"R² Score: {r2:.3f}")
    print(f"Number of predictions: {len(y_test):,}")
    print(f"Actual points range: {y_test.min():.1f} to {y_test.max():.1f}")
    print(f"Predicted points range: {y_pred.min():.1f} to {y_pred.max():.1f}")
    
    # Calculate percentage of predictions within different error ranges
    errors = np.abs(y_test - y_pred)
    within_1 = (errors <= 1).mean() * 100
    within_2 = (errors <= 2).mean() * 100
    within_3 = (errors <= 3).mean() * 100
    
    print(f"\nPrediction Accuracy:")
    print(f"Within ±1 point: {within_1:.1f}%")
    print(f"Within ±2 points: {within_2:.1f}%")
    print(f"Within ±3 points: {within_3:.1f}%")

def main():
    """Run all visualizations for the LightGBM model."""
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Loading model and data...")
    model, X_test, y_test, y_pred, X_train, y_train = load_model_and_data("lgbm")
    
    print("Creating visualizations...")
    
    # Create all plots
    plot_prediction_vs_actual(y_test, y_pred, "LightGBM")
    plot_error_distribution(y_test, y_pred, "LightGBM")
    plot_feature_importance(model, X_train, "LightGBM")
    plot_performance_by_points_range(y_test, y_pred, "LightGBM")
    
    # Create summary report
    create_summary_report(y_test, y_pred, "LightGBM")
    
    print(f"\nAll plots saved to the 'plots' directory!")

if __name__ == "__main__":
    main() 