# src/compare_models.py
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

def evaluate_model(model_name):
    """Evaluate a single model and return metrics."""
    try:
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
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model': model_name.upper(),
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_test': y_test,
            'y_pred': y_pred
        }
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        return None

def plot_model_comparison(results):
    """Create comparison plots for all models."""
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to plot!")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. MAE Comparison
    models = [r['model'] for r in valid_results]
    maes = [r['mae'] for r in valid_results]
    
    bars1 = axes[0,0].bar(models, maes, alpha=0.7)
    axes[0,0].set_ylabel('Mean Absolute Error (Lower is Better)')
    axes[0,0].set_title('Model Comparison: MAE')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mae in zip(bars1, maes):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mae:.3f}', ha='center', va='bottom')
    
    # 2. R² Comparison
    r2s = [r['r2'] for r in valid_results]
    
    bars2 = axes[0,1].bar(models, r2s, alpha=0.7, color='orange')
    axes[0,1].set_ylabel('R² Score (Higher is Better)')
    axes[0,1].set_title('Model Comparison: R² Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, r2 in zip(bars2, r2s):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{r2:.3f}', ha='center', va='bottom')
    
    # 3. Prediction vs Actual for all models
    for i, result in enumerate(valid_results):
        axes[1,0].scatter(result['y_test'], result['y_pred'], 
                          alpha=0.4, s=10, label=result['model'])
    
    # Add perfect prediction line
    all_y_test = np.concatenate([r['y_test'] for r in valid_results])
    axes[1,0].plot([all_y_test.min(), all_y_test.max()], 
                   [all_y_test.min(), all_y_test.max()], 'r--', lw=2)
    axes[1,0].set_xlabel('Actual FPL Points')
    axes[1,0].set_ylabel('Predicted FPL Points')
    axes[1,0].set_title('Predicted vs Actual (All Models)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Error distribution comparison
    for result in valid_results:
        errors = result['y_test'] - result['y_pred']
        axes[1,1].hist(errors, alpha=0.5, bins=30, label=result['model'])
    
    axes[1,1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Prediction Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Error Distribution Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_comparison_table(results):
    """Print a formatted comparison table."""
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid results to display!")
        return
    
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
    print("-" * 40)
    
    for result in valid_results:
        print(f"{result['model']:<12} {result['mae']:<8.3f} {result['rmse']:<8.3f} {result['r2']:<8.3f}")
    
    # Find best model for each metric
    best_mae = min(valid_results, key=lambda x: x['mae'])
    best_r2 = max(valid_results, key=lambda x: x['r2'])
    
    print(f"\nBest MAE: {best_mae['model']} ({best_mae['mae']:.3f})")
    print(f"Best R²: {best_r2['model']} ({best_r2['r2']:.3f})")

def main():
    """Compare all trained models."""
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Evaluating all models...")
    
    # Evaluate all models
    model_names = ["baseline", "rf", "lgbm"]
    results = []
    
    for model_name in model_names:
        print(f"Evaluating {model_name}...")
        result = evaluate_model(model_name)
        results.append(result)
    
    # Create comparison plots
    print("Creating comparison plots...")
    plot_model_comparison(results)
    
    # Print comparison table
    print_comparison_table(results)
    
    print(f"\nComparison plot saved to 'plots/model_comparison.png'")

if __name__ == "__main__":
    main() 