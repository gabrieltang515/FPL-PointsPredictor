from fastapi import APIRouter
from api.models.prediction_models import ApiResponse
import random

router = APIRouter(tags=["Test"])

@router.post("/predict/test-player")
async def test_predict_player():
    """Simple test prediction endpoint that doesn't rely on real data"""
    
    # Generate a mock prediction
    predicted_points = round(random.uniform(2.0, 12.0), 2)
    recent_form = round(random.uniform(1.0, 8.0), 2)
    consistency = round(random.uniform(0.5, 5.0), 2)
    
    # Determine confidence based on prediction
    if predicted_points >= 6 and recent_form >= 4:
        confidence = "High"
    elif predicted_points >= 4:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    mock_prediction = {
        'player_id': 123,
        'predicted_points': predicted_points,
        'recent_form': recent_form,
        'consistency_score': consistency,
        'confidence': confidence
    }
    
    return ApiResponse(
        success=True,
        data=mock_prediction,
        message="Test prediction generated successfully"
    )

@router.get("/predict/test-top-picks/{gameweek}")
async def test_top_picks(gameweek: int):
    """Generate mock top picks for testing"""
    
    mock_picks = []
    for i in range(20):
        player_id = 100 + i
        predicted_points = round(random.uniform(1.0, 10.0), 2)
        recent_form = round(random.uniform(1.0, 6.0), 2)
        consistency = round(random.uniform(0.5, 3.0), 2)
        
        mock_picks.append({
            'player_id': player_id,
            'predicted_points': predicted_points,
            'recent_form': recent_form,
            'consistency_score': consistency,
            'confidence': 'High' if predicted_points > 6 else 'Medium' if predicted_points > 4 else 'Low'
        })
    
    # Sort by predicted points (descending)
    mock_picks.sort(key=lambda x: x['predicted_points'], reverse=True)
    
    return ApiResponse(
        success=True,
        data=mock_picks,
        message=f"Generated {len(mock_picks)} test predictions for gameweek {gameweek}"
    )
