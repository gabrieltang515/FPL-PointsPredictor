from fastapi import APIRouter, HTTPException, Request, Query
from typing import List, Optional
from pydantic import BaseModel, Field

from api.models.prediction_models import ApiResponse

router = APIRouter(tags=["Team Builder"])


class TeamRequest(BaseModel):
    """Request model for team validation/optimization"""
    player_ids: List[int] = Field(..., min_items=15, max_items=15, description="15 player IDs")
    formation: Optional[str] = Field(None, description="Formation (e.g., 3-4-3, 4-4-2)")
    

class TeamOptimizationRequest(BaseModel):
    """Request for AI-optimized team"""
    gameweek: int = Field(..., ge=1, le=38)
    budget: float = Field(100.0, ge=0, le=100, description="Budget in millions")
    formation: Optional[str] = Field("3-4-3", description="Preferred formation")
    include_player_ids: Optional[List[int]] = Field(None, description="Players to include")
    exclude_player_ids: Optional[List[int]] = Field(None, description="Players to exclude")


class StartingElevenRequest(BaseModel):
    """Request for optimal starting XI from squad"""
    squad_player_ids: List[int] = Field(..., min_items=15, max_items=15)
    gameweek: int = Field(..., ge=1, le=38)
    formation: str = Field("3-4-3")
    captain_id: Optional[int] = None


@router.post("/team/validate")
async def validate_team(request: Request, team_request: TeamRequest):
    """Validate if a team meets FPL rules (budget, composition)"""
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        # Validate budget
        is_valid_budget, total_cost, budget_msg = fpl_service.validate_team_budget(
            team_request.player_ids
        )
        
        # Validate composition
        is_valid_composition, composition_msg = fpl_service.validate_team_composition(
            team_request.player_ids
        )
        
        # Get player details
        players = []
        for player_id in team_request.player_ids:
            player = fpl_service.get_player_metadata(player_id)
            if player:
                players.append(player)
        
        return ApiResponse(
            success=is_valid_budget and is_valid_composition,
            data={
                'is_valid': is_valid_budget and is_valid_composition,
                'budget': {
                    'is_valid': is_valid_budget,
                    'total_cost': total_cost,
                    'remaining': 100.0 - total_cost,
                    'message': budget_msg
                },
                'composition': {
                    'is_valid': is_valid_composition,
                    'message': composition_msg
                },
                'players': players
            },
            message="Team validation complete"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.post("/team/optimize")
async def optimize_team(request: Request, opt_request: TeamOptimizationRequest):
    """
    Generate an optimized team based on predictions
    Uses greedy algorithm to maximize predicted points within constraints
    """
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Get predictions for the gameweek
        all_predictions = model_service.predict_gameweek(
            gameweek=opt_request.gameweek,
            top_n=200  # Get more to have options
        )
        
        # Filter out excluded players
        if opt_request.exclude_player_ids:
            all_predictions = [
                p for p in all_predictions 
                if p['player_id'] not in opt_request.exclude_player_ids
            ]
        
        # Build team using greedy algorithm
        selected_players = []
        position_requirements = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {}
        total_cost = 0.0
        
        # First, add any required players
        if opt_request.include_player_ids:
            for player_id in opt_request.include_player_ids:
                player_pred = next((p for p in all_predictions if p['player_id'] == player_id), None)
                if player_pred:
                    selected_players.append(player_pred)
                    position_counts[player_pred['position']] += 1
                    team_counts[player_pred['team']] = team_counts.get(player_pred['team'], 0) + 1
                    total_cost += player_pred['price']
        
        # Sort by predicted points per price (value)
        all_predictions.sort(
            key=lambda x: x['predicted_points'] / x['price'], 
            reverse=True
        )
        
        # Greedy selection
        for pred in all_predictions:
            if len(selected_players) >= 15:
                break
            
            # Skip if already selected
            if pred['player_id'] in [p['player_id'] for p in selected_players]:
                continue
            
            position = pred['position']
            team = pred['team']
            price = pred['price']
            
            # Check constraints
            if position_counts.get(position, 0) >= position_requirements.get(position, 0):
                continue
            if team_counts.get(team, 0) >= 3:
                continue
            if total_cost + price > opt_request.budget:
                continue
            
            # Add player
            selected_players.append(pred)
            position_counts[position] = position_counts.get(position, 0) + 1
            team_counts[team] = team_counts.get(team, 0) + 1
            total_cost += price
        
        # Calculate team stats
        total_predicted_points = sum(p['predicted_points'] for p in selected_players)
        
        return ApiResponse(
            success=True,
            data={
                'team': selected_players,
                'total_players': len(selected_players),
                'total_cost': round(total_cost, 1),
                'remaining_budget': round(opt_request.budget - total_cost, 1),
                'total_predicted_points': round(total_predicted_points, 1),
                'average_predicted_points': round(total_predicted_points / len(selected_players), 2) if selected_players else 0,
                'position_counts': position_counts,
                'is_complete': len(selected_players) == 15
            },
            message=f"Generated {'complete' if len(selected_players) == 15 else 'partial'} team with {len(selected_players)} players"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Team optimization failed: {str(e)}")


@router.post("/team/starting-eleven")
async def get_starting_eleven(request: Request, squad_request: StartingElevenRequest):
    """
    Select optimal starting XI from a 15-player squad based on predictions
    """
    try:
        model_service = request.app.state.model_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        # Get predictions for all squad players
        predictions = []
        for player_id in squad_request.squad_player_ids:
            try:
                pred = model_service.predict_player_points(
                    player_id=player_id,
                    gameweek=squad_request.gameweek
                )
                predictions.append(pred)
            except:
                continue
        
        # Parse formation (e.g., "3-4-3" -> [3, 4, 3])
        try:
            formation_parts = [int(x) for x in squad_request.formation.split('-')]
            def_count, mid_count, fwd_count = formation_parts
        except:
            def_count, mid_count, fwd_count = 3, 4, 3
        
        # Sort predictions by position and predicted points
        by_position = {
            'GKP': sorted([p for p in predictions if p['position'] == 'GKP'], 
                         key=lambda x: x['predicted_points'], reverse=True),
            'DEF': sorted([p for p in predictions if p['position'] == 'DEF'], 
                         key=lambda x: x['predicted_points'], reverse=True),
            'MID': sorted([p for p in predictions if p['position'] == 'MID'], 
                         key=lambda x: x['predicted_points'], reverse=True),
            'FWD': sorted([p for p in predictions if p['position'] == 'FWD'], 
                         key=lambda x: x['predicted_points'], reverse=True),
        }
        
        # Select starting XI
        starting_xi = []
        bench = []
        
        # Add 1 GKP
        if by_position['GKP']:
            starting_xi.append(by_position['GKP'][0])
            bench.extend(by_position['GKP'][1:])
        
        # Add defenders
        starting_xi.extend(by_position['DEF'][:def_count])
        bench.extend(by_position['DEF'][def_count:])
        
        # Add midfielders
        starting_xi.extend(by_position['MID'][:mid_count])
        bench.extend(by_position['MID'][mid_count:])
        
        # Add forwards
        starting_xi.extend(by_position['FWD'][:fwd_count])
        bench.extend(by_position['FWD'][fwd_count:])
        
        # Sort bench by predicted points
        bench.sort(key=lambda x: x['predicted_points'], reverse=True)
        
        # Select captain (highest predicted points or specified)
        captain = None
        if squad_request.captain_id:
            captain = next((p for p in starting_xi if p['player_id'] == squad_request.captain_id), None)
        if not captain and starting_xi:
            captain = max(starting_xi, key=lambda x: x['predicted_points'])
        
        # Calculate total predicted points (captain gets 2x)
        total_predicted = sum(p['predicted_points'] for p in starting_xi)
        if captain:
            total_predicted += captain['predicted_points']  # Captain bonus
        
        return ApiResponse(
            success=True,
            data={
                'starting_xi': starting_xi,
                'bench': bench,
                'captain': captain,
                'formation': squad_request.formation,
                'total_predicted_points': round(total_predicted, 1)
            },
            message=f"Selected starting XI with {squad_request.formation} formation"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Starting XI selection failed: {str(e)}")
