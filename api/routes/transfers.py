from fastapi import APIRouter, HTTPException, Request, Query
from typing import List, Optional
from pydantic import BaseModel, Field

from api.models.prediction_models import ApiResponse

router = APIRouter(tags=["Transfers"])


class TransferSuggestionRequest(BaseModel):
    """Request for transfer suggestions"""
    current_squad: List[int] = Field(..., min_items=15, max_items=15, description="Current 15 player IDs")
    gameweek: int = Field(..., ge=1, le=38)
    num_transfers: int = Field(1, ge=1, le=15, description="Number of transfers to suggest")
    remaining_budget: float = Field(0.0, description="Extra budget available for transfers (in millions)")
    

class TransferAnalysisRequest(BaseModel):
    """Request for analyzing a specific transfer"""
    player_out_id: int
    player_in_id: int
    gameweek: int
    current_squad: List[int] = Field(..., min_items=15, max_items=15)


@router.post("/transfers/suggest")
async def suggest_transfers(request: Request, transfer_request: TransferSuggestionRequest):
    """
    Suggest optimal transfers based on predictions
    Analyzes current squad and recommends replacements
    """
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Get predictions for current squad
        current_predictions = []
        current_squad_set = set(transfer_request.current_squad)
        
        for player_id in transfer_request.current_squad:
            try:
                pred = model_service.predict_player_points(
                    player_id=player_id,
                    gameweek=transfer_request.gameweek
                )
                current_predictions.append(pred)
            except Exception as e:
                print(f"Warning: Could not predict for current player {player_id}: {e}")
        
        # Get predictions for all available players (not in current squad)
        all_predictions = model_service.predict_gameweek(
            gameweek=transfer_request.gameweek,
            top_n=150
        )
        
        # Filter to only players not in current squad
        available_predictions = [
            p for p in all_predictions 
            if p['player_id'] not in current_squad_set
        ]
        
        # Generate transfer suggestions
        suggestions = []
        
        for current_player in current_predictions:
            current_pos = current_player['position']
            current_price = current_player['price']
            current_pred_points = current_player['predicted_points']
            
            # Find potential replacements in same position
            replacements = [
                p for p in available_predictions 
                if p['position'] == current_pos
                and p['price'] <= current_price + transfer_request.remaining_budget
            ]
            
            # Sort by predicted points
            replacements.sort(key=lambda x: x['predicted_points'], reverse=True)
            
            # Take top 3 replacements
            for replacement in replacements[:3]:
                points_gain = replacement['predicted_points'] - current_pred_points
                
                # Only suggest if significant improvement (at least 1 point)
                if points_gain >= 1.0:
                    suggestions.append({
                        'player_out': current_player,
                        'player_in': replacement,
                        'predicted_points_gain': round(points_gain, 2),
                        'price_difference': round(replacement['price'] - current_price, 1),
                        'value_score': round(points_gain / (replacement['price'] + 0.1), 2),  # Points per million
                        'priority': 'High' if points_gain >= 3 else 'Medium' if points_gain >= 2 else 'Low'
                    })
        
        # Sort by predicted points gain
        suggestions.sort(key=lambda x: x['predicted_points_gain'], reverse=True)
        
        # Limit to requested number
        suggestions = suggestions[:transfer_request.num_transfers * 3]  # Give 3x options
        
        # Calculate impact if top transfers are made
        top_transfers = suggestions[:transfer_request.num_transfers]
        total_points_gain = sum(t['predicted_points_gain'] for t in top_transfers)
        total_cost = sum(t['price_difference'] for t in top_transfers)
        
        return ApiResponse(
            success=True,
            data={
                'suggestions': suggestions,
                'top_transfers': top_transfers,
                'impact_analysis': {
                    'total_predicted_points_gain': round(total_points_gain, 1),
                    'total_cost': round(total_cost, 1),
                    'affordable': total_cost <= transfer_request.remaining_budget,
                    'num_suggestions': len(suggestions)
                },
                'current_squad_summary': {
                    'total_predicted_points': round(sum(p['predicted_points'] for p in current_predictions), 1),
                    'average_predicted_points': round(sum(p['predicted_points'] for p in current_predictions) / len(current_predictions), 2),
                    'weakest_players': sorted(current_predictions, key=lambda x: x['predicted_points'])[:3]
                }
            },
            message=f"Generated {len(suggestions)} transfer suggestions"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer suggestions failed: {str(e)}")


@router.post("/transfers/analyze")
async def analyze_transfer(request: Request, analysis_request: TransferAnalysisRequest):
    """
    Analyze a specific transfer decision
    Compare predicted performance and validate constraints
    """
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Get predictions for both players
        player_out_pred = model_service.predict_player_points(
            player_id=analysis_request.player_out_id,
            gameweek=analysis_request.gameweek
        )
        
        player_in_pred = model_service.predict_player_points(
            player_id=analysis_request.player_in_id,
            gameweek=analysis_request.gameweek
        )
        
        # Calculate new squad after transfer
        new_squad = [
            pid if pid != analysis_request.player_out_id else analysis_request.player_in_id
            for pid in analysis_request.current_squad
        ]
        
        # Validate new squad
        is_valid_budget, total_cost, budget_msg = fpl_service.validate_team_budget(new_squad)
        is_valid_composition, composition_msg = fpl_service.validate_team_composition(new_squad)
        
        # Calculate points difference
        points_diff = player_in_pred['predicted_points'] - player_out_pred['predicted_points']
        price_diff = player_in_pred['price'] - player_out_pred['price']
        
        # Get next 5 gameweeks predictions for both
        future_gameweeks = []
        for gw_offset in range(1, 6):
            future_gw = analysis_request.gameweek + gw_offset
            if future_gw <= 38:
                try:
                    out_future = model_service.predict_player_points(
                        analysis_request.player_out_id, future_gw
                    )
                    in_future = model_service.predict_player_points(
                        analysis_request.player_in_id, future_gw
                    )
                    future_gameweeks.append({
                        'gameweek': future_gw,
                        'player_out_prediction': out_future['predicted_points'],
                        'player_in_prediction': in_future['predicted_points'],
                        'points_difference': round(in_future['predicted_points'] - out_future['predicted_points'], 2)
                    })
                except:
                    continue
        
        # Calculate 5-gameweek impact
        total_5gw_gain = sum(fg['points_difference'] for fg in future_gameweeks)
        
        # Recommendation
        recommendation = "Not Recommended"
        if is_valid_budget and is_valid_composition:
            if points_diff >= 2 and total_5gw_gain >= 5:
                recommendation = "Highly Recommended"
            elif points_diff >= 1 or total_5gw_gain >= 3:
                recommendation = "Consider"
            elif points_diff > 0:
                recommendation = "Marginal Gain"
        
        return ApiResponse(
            success=True,
            data={
                'player_out': player_out_pred,
                'player_in': player_in_pred,
                'immediate_impact': {
                    'points_difference': round(points_diff, 2),
                    'price_difference': round(price_diff, 1),
                    'value_rating': round(points_diff / (player_in_pred['price'] + 0.1), 2)
                },
                'future_impact': {
                    'next_5_gameweeks': future_gameweeks,
                    'total_predicted_gain': round(total_5gw_gain, 1),
                    'average_gain_per_gw': round(total_5gw_gain / len(future_gameweeks), 2) if future_gameweeks else 0
                },
                'validation': {
                    'is_valid': is_valid_budget and is_valid_composition,
                    'budget_check': {
                        'is_valid': is_valid_budget,
                        'message': budget_msg
                    },
                    'composition_check': {
                        'is_valid': is_valid_composition,
                        'message': composition_msg
                    }
                },
                'recommendation': recommendation
            },
            message=f"Transfer analysis complete: {recommendation}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transfer analysis failed: {str(e)}")


@router.get("/transfers/hot-picks/{gameweek}")
async def get_hot_picks(request: Request, gameweek: int, 
                       position: Optional[str] = Query(None),
                       max_price: Optional[float] = Query(None),
                       limit: int = Query(20, ge=1, le=50)):
    """
    Get trending players with high predicted points (good transfer targets)
    """
    try:
        model_service = request.app.state.model_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Get predictions
        predictions = model_service.predict_gameweek(
            gameweek=gameweek,
            position=position,
            max_price=max_price,
            top_n=100
        )
        
        # Calculate value scores and sort
        for pred in predictions:
            pred['value_score'] = round(pred['predicted_points'] / pred['price'], 2)
            pred['form_score'] = round((pred['predicted_points'] + pred['recent_form']) / 2, 2)
        
        # Sort by value score
        predictions.sort(key=lambda x: x['value_score'], reverse=True)
        
        hot_picks = predictions[:limit]
        
        return ApiResponse(
            success=True,
            data={
                'hot_picks': hot_picks,
                'filters': {
                    'position': position,
                    'max_price': max_price,
                    'gameweek': gameweek
                }
            },
            message=f"Found {len(hot_picks)} hot picks for gameweek {gameweek}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get hot picks: {str(e)}")


@router.get("/transfers/differential-picks/{gameweek}")
async def get_differential_picks(request: Request, gameweek: int,
                                 max_ownership: float = Query(5.0, description="Max ownership %"),
                                 limit: int = Query(20, ge=1, le=50)):
    """
    Get low-owned players with high predicted points (differential picks)
    """
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Get predictions
        predictions = model_service.predict_gameweek(
            gameweek=gameweek,
            top_n=200
        )
        
        # Filter by ownership and sort by predicted points
        differentials = [
            p for p in predictions
            if p['selected_by'] <= max_ownership and p['predicted_points'] >= 4.0
        ]
        
        differentials.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        return ApiResponse(
            success=True,
            data={
                'differential_picks': differentials[:limit],
                'criteria': {
                    'max_ownership': max_ownership,
                    'min_predicted_points': 4.0,
                    'gameweek': gameweek
                }
            },
            message=f"Found {len(differentials[:limit])} differential picks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get differential picks: {str(e)}")
