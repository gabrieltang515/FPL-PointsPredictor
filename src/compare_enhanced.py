# src/compare_enhanced.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.features import load_processed, make_features, get_X_y
from src.enhanced_model import create_enhanced_features

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def evaluate_enhanced_model():
    """Evaluate the enhanced model and compare with existing models."""
    print("🔍 Evaluating Enhanced Model vs Current Models")
    print("=" * 60)
    
    # Load data
    df = load_processed("data/processed/player_gw_stats.csv")
    
    # Prepare both basic and enhanced features
    feat_basic = make_features(df, window=3)
    feat_enhanced = create_enhanced_features(feat_basic, window=3)
    
    X_basic, y_basic = get_X_y(feat_basic)
    X_enhanced, y_enhanced = get_X_y(feat_enhanced)
    
    # Split data
    X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(
        X_basic, y_basic, test_size=0.2, shuffle=False, random_state=42
    )
    
    X_train_enhanced, X_test_enhanced, y_train_enhanced, y_test_enhanced = train_test_split(
        X_enhanced, y_enhanced, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Load existing models
    models = {}
    try:
        models['current_lgbm'] = joblib.load("models/lgbm.joblib")
        models['tuned_lgbm'] = joblib.load("models/lgbm_tuned.joblib")
        models['enhanced_lgbm'] = joblib.load("models/lgbm_enhanced.joblib")
    except FileNotFoundError as e:
        print(f"⚠️ Some model files not found: {e}")
        return None
    
    # Evaluate models
    results = {}
    
    # Current models with basic features
    for name in ['current_lgbm', 'tuned_lgbm']:
        if name in models:
            model = models[name]
            y_pred = model.predict(X_test_basic)
            
            mae = mean_absolute_error(y_test_basic, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test_basic, y_pred))
            r2 = r2_score(y_test_basic, y_pred)
            
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'y_pred': y_pred,
                'y_test': y_test_basic,
                'features': 'basic'
            }
    
    # Enhanced model with enhanced features
    if 'enhanced_lgbm' in models:
        model = models['enhanced_lgbm']
        y_pred = model.predict(X_test_enhanced)
        
        mae = mean_absolute_error(y_test_enhanced, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_enhanced, y_pred))
        r2 = r2_score(y_test_enhanced, y_pred)
        
        results['enhanced_lgbm'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred,
            'y_test': y_test_enhanced,
            'features': 'enhanced'
        }
    
    return results, X_train_enhanced, X_test_enhanced

def plot_enhanced_comparison(results):
    """Create comparison plots for enhanced vs current models."""
    if not results:
        print("No results to plot!")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. MAE Comparison
    models = list(results.keys())
    maes = [results[name]['mae'] for name in models]
    
    bars1 = axes[0,0].bar(models, maes, alpha=0.7)
    axes[0,0].set_ylabel('Mean Absolute Error (Lower is Better)')
    axes[0,0].set_title('Model Comparison: MAE')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mae in zip(bars1, maes):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{mae:.3f}', ha='center', va='bottom')
    
    # 2. R² Comparison
    r2s = [results[name]['r2'] for name in models]
    
    bars2 = axes[0,1].bar(models, r2s, alpha=0.7, color='orange')
    axes[0,1].set_ylabel('R² Score (Higher is Better)')
    axes[0,1].set_title('Model Comparison: R² Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, r2 in zip(bars2, r2s):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{r2:.3f}', ha='center', va='bottom')
    
    # 3. RMSE Comparison
    rmses = [results[name]['rmse'] for name in models]
    
    bars3 = axes[0,2].bar(models, rmses, alpha=0.7, color='green')
    axes[0,2].set_ylabel('RMSE (Lower is Better)')
    axes[0,2].set_title('Model Comparison: RMSE')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rmse in zip(bars3, rmses):
        axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{rmse:.3f}', ha='center', va='bottom')
    
    # 4. Prediction vs Actual for all models
    for i, (name, result) in enumerate(results.items()):
        axes[1,0].scatter(result['y_test'], result['y_pred'], 
                          alpha=0.4, s=10, label=name.replace('_', ' ').title())
    
    # Add perfect prediction line
    all_y_test = np.concatenate([r['y_test'] for r in results.values()])
    axes[1,0].plot([all_y_test.min(), all_y_test.max()], 
                   [all_y_test.min(), all_y_test.max()], 'r--', lw=2)
    axes[1,0].set_xlabel('Actual FPL Points')
    axes[1,0].set_ylabel('Predicted FPL Points')
    axes[1,0].set_title('Predicted vs Actual (All Models)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Error distribution comparison
    for name, result in results.items():
        errors = result['y_test'] - result['y_pred']
        axes[1,1].hist(errors, alpha=0.5, bins=30, label=name.replace('_', ' ').title())
    
    axes[1,1].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('Prediction Error')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Error Distribution Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Improvement analysis
    if len(results) >= 2:
        # Calculate improvements
        baseline_mae = results.get('current_lgbm', {}).get('mae', float('inf'))
        enhanced_mae = results.get('enhanced_lgbm', {}).get('mae', float('inf'))
        
        if baseline_mae != float('inf') and enhanced_mae != float('inf'):
            improvement = ((baseline_mae - enhanced_mae) / baseline_mae) * 100
            
            axes[1,2].bar(['Current', 'Enhanced'], [baseline_mae, enhanced_mae], 
                          alpha=0.7, color=['red', 'green'])
            axes[1,2].set_ylabel('MAE')
            axes[1,2].set_title(f'MAE Improvement: {improvement:.1f}%')
            axes[1,2].grid(True, alpha=0.3)
            
            # Add value labels
            axes[1,2].text(0, baseline_mae + 0.001, f'{baseline_mae:.3f}', 
                           ha='center', va='bottom')
            axes[1,2].text(1, enhanced_mae + 0.001, f'{enhanced_mae:.3f}', 
                           ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/enhanced_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_enhanced_comparison_table(results):
    """Print a formatted comparison table with improvements."""
    if not results:
        print("No results to display!")
        return
    
    print(f"\n{'='*80}")
    print("ENHANCED MODEL PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Features':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<20} {result['mae']:<8.3f} {result['rmse']:<8.3f} "
              f"{result['r2']:<8.3f} {result['features']:<10}")
    
    # Find best model for each metric
    best_mae = min(results.items(), key=lambda x: x[1]['mae'])
    best_r2 = max(results.items(), key=lambda x: x[1]['r2'])
    
    print(f"\n🏆 Best MAE: {best_mae[0].replace('_', ' ').title()} ({best_mae[1]['mae']:.3f})")
    print(f"🏆 Best R²: {best_r2[0].replace('_', ' ').title()} ({best_r2[1]['r2']:.3f})")
    
    # Calculate improvements
    if 'current_lgbm' in results and 'enhanced_lgbm' in results:
        current = results['current_lgbm']
        enhanced = results['enhanced_lgbm']
        
        mae_improvement = ((current['mae'] - enhanced['mae']) / current['mae']) * 100
        r2_improvement = ((enhanced['r2'] - current['r2']) / current['r2']) * 100
        
        print(f"\n📈 IMPROVEMENTS:")
        print(f"MAE Improvement: {mae_improvement:.2f}%")
        print(f"R² Improvement: {r2_improvement:.2f}%")

def analyze_feature_importance_enhanced():
    """Analyze feature importance for the enhanced model."""
    try:
        enhanced_model = joblib.load("models/lgbm_enhanced.joblib")
        
        # Load enhanced features
        df = load_processed("data/processed/player_gw_stats.csv")
        feat = make_features(df, window=3)
        feat_enhanced = create_enhanced_features(feat, window=3)
        X, y = get_X_y(feat_enhanced)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        
        if hasattr(enhanced_model, 'feature_importances_'):
            importances = enhanced_model.feature_importances_
            feature_names = X_train.columns
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print(f"\n🔍 Enhanced Model - Top 20 Most Important Features:")
            print("=" * 80)
            for idx, row in importance_df.head(20).iterrows():
                print(f"{row['feature']:<40} {row['importance']:>8.2f}")
            
            # Save feature importance
            importance_df.to_csv("models/enhanced_feature_importance.csv", index=False)
            print(f"\n💾 Feature importance saved to models/enhanced_feature_importance.csv")
            
            return importance_df
        else:
            print("⚠️ Enhanced model doesn't have feature_importances_ attribute")
            return None
            
    except FileNotFoundError:
        print("⚠️ Enhanced model not found. Please train it first.")
        return None

def main():
    """Run enhanced model comparison."""
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("🚀 Enhanced Model Comparison")
    print("=" * 50)
    
    # Evaluate models
    results, X_train_enhanced, X_test_enhanced = evaluate_enhanced_model()
    
    if results:
        # Create comparison plots
        print("📊 Creating comparison plots...")
        plot_enhanced_comparison(results)
        
        # Print comparison table
        print_enhanced_comparison_table(results)
        
        # Analyze feature importance
        print("\n🔍 Analyzing enhanced model feature importance...")
        analyze_feature_importance_enhanced()
        
        print(f"\n📈 Enhanced comparison plot saved to 'plots/enhanced_model_comparison.png'")
    else:
        print("❌ No models found to compare. Please ensure all models are trained.")

if __name__ == "__main__":
    main() 