from fastapi import APIRouter, HTTPException, Request, Query
from typing import List
from datetime import datetime

from api.models.prediction_models import (
    PlayerPredictionRequest, 
    GameweekPredictionRequest,
    PlayerPrediction,
    GameweekPredictions,
    ApiResponse
)

router = APIRouter(tags=["Predictions"])

@router.post("/predict/player", response_model=ApiResponse)
async def predict_player_points(request: Request, prediction_request: PlayerPredictionRequest):
    """Predict points for a specific player in a specific gameweek"""
    try:
        model_service = request.app.state.model_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        result = model_service.predict_player_points(
            player_id=prediction_request.player_id,
            gameweek=prediction_request.gameweek,
            opponent_team=prediction_request.opponent_team,
            was_home=prediction_request.was_home,
            fixture_difficulty=prediction_request.fixture_difficulty
        )
        
        return ApiResponse(
            success=True,
            data=result,
            message="Prediction generated successfully"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict/gameweek", response_model=ApiResponse)
async def predict_gameweek_points(request: Request, prediction_request: GameweekPredictionRequest):
    """Get predictions for all players in a gameweek with live FPL data"""
    try:
        model_service = request.app.state.model_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        predictions = model_service.predict_gameweek(
            gameweek=prediction_request.gameweek,
            top_n=50
        )
        
        # Add recommendation ratings
        for pred in predictions:
            pred['recommendation'] = (
                "Strong Buy" if pred['predicted_points'] >= 7 else
                "Buy" if pred['predicted_points'] >= 5 else
                "Hold" if pred['predicted_points'] >= 3 else
                "Avoid"
            )
        
        return ApiResponse(
            success=True,
            data={
                'gameweek': prediction_request.gameweek,
                'predictions': predictions,
                'top_picks': predictions[:10],
                'total_players': len(predictions)
            },
            message=f"Generated predictions for {len(predictions)} players"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gameweek prediction failed: {str(e)}")

@router.get("/predict/top-picks/{gameweek}")
async def get_top_picks(request: Request, gameweek: int, limit: int = Query(10, ge=1, le=50)):
    """Get top predicted players for a gameweek"""
    try:
        model_service = request.app.state.model_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        predictions = model_service.predict_gameweek(gameweek=gameweek, top_n=limit)
        
        return ApiResponse(
            success=True,
            data=predictions,
            message=f"Top {len(predictions)} picks for gameweek {gameweek}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get top picks: {str(e)}")
