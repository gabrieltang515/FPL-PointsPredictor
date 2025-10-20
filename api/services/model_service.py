import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import sys
from pathlib import Path

# Add src to path to import our existing modules
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.features import load_processed, make_features, get_X_y
from src.enhanced_model import create_enhanced_features
from api.services.fpl_data_service import get_fpl_service

class ModelService:
    """Service for loading and running ML model predictions"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_loaded = False
        self.model_path = project_root / "models" / "lgbm_enhanced.joblib"
        self.fpl_service = get_fpl_service()
        
    def load_model(self):
        """Load the trained model and prepare feature engineering pipeline"""
        try:
            # Load the trained model
            if not self.model_path.exists():
                # Try alternative model files
                alternatives = [
                    "lgbm_enhanced_local.joblib",
                    "lgbm_enhanced_lightweight.joblib", 
                    "best_single.joblib",
                    "lgbm.joblib"
                ]
                
                for alt in alternatives:
                    alt_path = project_root / "models" / alt
                    if alt_path.exists():
                        self.model_path = alt_path
                        break
                else:
                    raise FileNotFoundError("No trained model found in models/ directory")
            
            self.model = joblib.load(self.model_path)
            
            # Load sample data to get feature columns
            df = load_processed(str(project_root / "data" / "processed" / "player_gw_stats.csv"))
            feat = make_features(df, window=3)
            feat_enhanced = create_enhanced_features(feat, window=3)
            X, y = get_X_y(feat_enhanced)
            self.feature_columns = X.columns.tolist()
            
            self.is_loaded = True
            print(f"✅ Model loaded from {self.model_path}")
            print(f"📊 Feature columns: {len(self.feature_columns)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def predict_player_points(self, 
                            player_id: int, 
                            gameweek: int,
                            opponent_team: str = None,
                            was_home: bool = None,
                            fixture_difficulty: int = None) -> Dict:
        """
        Predict points for a specific player with live FPL metadata
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Get player metadata from FPL API
            player_meta = self.fpl_service.get_player_metadata(player_id)
            if not player_meta:
                raise ValueError(f"Player {player_id} not found in FPL API")
            
            # Get fixture data for the gameweek if not provided
            fixture = None
            if opponent_team is None or was_home is None or fixture_difficulty is None:
                fixture = self.fpl_service.get_fixture_for_gameweek(player_id, gameweek)
                if fixture:
                    opponent_team = fixture['opponent']
                    was_home = fixture['is_home']
                    fixture_difficulty = fixture['difficulty'] or 3
                else:
                    # Use defaults if no fixture found
                    opponent_team = opponent_team or "Average"
                    was_home = was_home if was_home is not None else True
                    fixture_difficulty = fixture_difficulty or 3
            
            # Load historical data for the player
            df = load_processed(str(project_root / "data" / "processed" / "player_gw_stats.csv"))
            
            # Filter for this player's recent data
            player_data = df[df['player_id'] == player_id].sort_values('gameweek')
            
            if len(player_data) == 0:
                raise ValueError(f"No historical data found for player {player_id}")
            
            # Create a new row for prediction
            latest_data = player_data.iloc[-1].copy()
            latest_data['gameweek'] = gameweek
            latest_data['opponent_team'] = opponent_team
            latest_data['was_home'] = was_home
            latest_data['fixture_difficulty'] = fixture_difficulty
            
            # Add this row to the dataframe for feature engineering
            pred_df = pd.concat([df, latest_data.to_frame().T], ignore_index=True)
            
            # Ensure data types are correct before feature engineering
            numeric_cols = ['total_points', 'minutes', 'goals', 'assists', 'xG', 'xA', 'key_passes', 'npxG']
            for col in numeric_cols:
                if col in pred_df.columns:
                    pred_df[col] = pd.to_numeric(pred_df[col], errors='coerce').fillna(0)
            
            # Create features
            feat = make_features(pred_df, window=3)
            feat_enhanced = create_enhanced_features(feat, window=3)
            
            # Get the last row (our prediction row)
            pred_row = feat_enhanced[feat_enhanced['player_id'] == player_id].iloc[-1]
            
            # Prepare features for prediction
            X_pred = pred_row.drop(['player_id', 'gameweek', 'total_points', 'target']).to_frame().T
            
            # Ensure all feature columns are present
            for col in self.feature_columns:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            
            # Reorder columns to match training data
            X_pred = X_pred[self.feature_columns]
            
            # Ensure all columns are numeric
            X_pred = X_pred.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Make prediction
            prediction = self.model.predict(X_pred)[0]
            
            # Calculate confidence metrics
            recent_form = player_data['total_points'].tail(5).mean()
            consistency = 1 / (player_data['total_points'].tail(5).std() + 1e-8)
            
            return {
                'player_id': player_id,
                'player_name': player_meta['name'],
                'full_name': player_meta['full_name'],
                'team': player_meta['team'],
                'position': player_meta['position'],
                'price': player_meta['price'],
                'predicted_points': round(prediction, 2),
                'recent_form': round(recent_form, 2),
                'consistency_score': round(min(consistency, 10), 2),
                'confidence': self._calculate_confidence(prediction, recent_form, consistency),
                'fixture': {
                    'opponent': opponent_team,
                    'is_home': was_home,
                    'difficulty': fixture_difficulty
                } if fixture else None,
                'availability': player_meta['status'],
                'selected_by': player_meta['selected_by']
            }
            
        except Exception as e:
            print(f"❌ Error predicting for player {player_id}: {e}")
            raise e
    
    def predict_gameweek(self, gameweek: int, top_n: int = 50, 
                        position: Optional[str] = None,
                        max_price: Optional[float] = None) -> List[Dict]:
        """
        Predict points for all available players in a gameweek with live fixtures
        Args:
            gameweek: Gameweek number to predict
            top_n: Number of top players to return
            position: Filter by position (GKP, DEF, MID, FWD)
            max_price: Maximum price filter
        """
        if not self.is_loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Get all available players from FPL API
            all_players = self.fpl_service.get_all_players(position=position, available_only=True)
            
            # Filter by price if specified
            if max_price:
                all_players = [p for p in all_players if p['price'] <= max_price]
            
            # Load historical data
            df = load_processed(str(project_root / "data" / "processed" / "player_gw_stats.csv"))
            historical_player_ids = set(df['player_id'].unique())
            
            predictions = []
            
            for player in all_players[:200]:  # Limit to top 200 by total points for performance
                player_id = player['id']
                
                # Skip if no historical data
                if player_id not in historical_player_ids:
                    continue
                
                try:
                    # Predict with automatic fixture lookup
                    pred_result = self.predict_player_points(
                        player_id=player_id,
                        gameweek=gameweek
                    )
                    predictions.append(pred_result)
                except Exception as e:
                    print(f"Warning: Could not predict for player {player_id}: {e}")
                    continue
            
            # Sort by predicted points
            predictions.sort(key=lambda x: x['predicted_points'], reverse=True)
            
            return predictions[:top_n]
            
        except Exception as e:
            print(f"❌ Error predicting gameweek {gameweek}: {e}")
            raise e
    
    def _calculate_confidence(self, prediction: float, recent_form: float, consistency: float) -> str:
        """Calculate confidence rating for prediction"""
        # Simple confidence calculation based on prediction value and form
        if prediction >= 6 and recent_form >= 4 and consistency >= 2:
            return "High"
        elif prediction >= 4 and recent_form >= 2:
            return "Medium"
        else:
            return "Low"
    
    def get_player_info(self, player_id: int) -> Dict:
        """Get comprehensive player info with live FPL metadata"""
        try:
            # Get live metadata from FPL API
            player_meta = self.fpl_service.get_player_metadata(player_id)
            if not player_meta:
                raise ValueError(f"Player {player_id} not found in FPL API")
            
            # Get upcoming fixtures
            fixtures = self.fpl_service.get_player_fixtures(player_id, num_fixtures=5)
            
            # Get historical data if available
            df = load_processed(str(project_root / "data" / "processed" / "player_gw_stats.csv"))
            player_data = df[df['player_id'] == player_id]
            
            recent_points = []
            if len(player_data) > 0:
                recent_points = player_data['total_points'].tail(5).tolist()
            
            return {
                'player_id': player_id,
                'name': player_meta['name'],
                'full_name': player_meta['full_name'],
                'team': player_meta['team'],
                'position': player_meta['position'],
                'price': player_meta['price'],
                'total_points': player_meta['total_points'],
                'form': player_meta['form'],
                'points_per_game': player_meta['points_per_game'],
                'selected_by': player_meta['selected_by'],
                'status': player_meta['status'],
                'news': player_meta['news'],
                'minutes': player_meta['minutes'],
                'goals_scored': player_meta['goals_scored'],
                'assists': player_meta['assists'],
                'recent_points': recent_points,
                'upcoming_fixtures': fixtures
            }
            
        except Exception as e:
            print(f"❌ Error getting player info for {player_id}: {e}")
            raise e
