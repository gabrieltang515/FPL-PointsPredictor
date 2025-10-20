from fastapi import APIRouter, HTTPException, Request, Query
from typing import List
from api.models.prediction_models import PlayerInfo, PlayerDetailedInfo, ApiResponse

router = APIRouter(tags=["Players"])

@router.get("/players", response_model=ApiResponse)
async def get_players(request: Request, 
                     limit: int = Query(50, ge=1, le=200),
                     position: str = Query(None, description="Filter by position (FWD, MID, DEF, GKP)"),
                     max_price: float = Query(None, description="Maximum price filter")):
    """Get list of all players with live FPL data"""
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        # Get players from FPL API with metadata
        all_players = fpl_service.get_all_players(position=position, available_only=False)
        
        # Apply price filter
        if max_price:
            all_players = [p for p in all_players if p['price'] <= max_price]
        
        # Sort by total points
        all_players.sort(key=lambda x: x['total_points'], reverse=True)
        
        return ApiResponse(
            success=True,
            data=all_players[:limit],
            message=f"Retrieved {len(all_players[:limit])} players"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get players: {str(e)}")

@router.get("/players/search/{query}")
async def search_players(request: Request, query: str, limit: int = Query(20, ge=1, le=50)):
    """Search for players by name"""
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        results = fpl_service.search_players(query, limit=limit)
        
        return ApiResponse(
            success=True,
            data=results,
            message=f"Found {len(results)} players matching '{query}'"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/players/{player_id}", response_model=ApiResponse)
async def get_player_details(request: Request, player_id: int):
    """Get comprehensive player details with live FPL data and predictions"""
    try:
        model_service = request.app.state.model_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        player_info = model_service.get_player_info(player_id)
        
        return ApiResponse(
            success=True,
            data=player_info,
            message=f"Retrieved details for {player_info['name']}"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get player details: {str(e)}")

@router.get("/players/{player_id}/form")
async def get_player_form(request: Request, player_id: int, games: int = Query(5, ge=1, le=20)):
    """Get recent form data for a player"""
    try:
        model_service = request.app.state.model_service
        
        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")
        
        player_info = model_service.get_player_info(player_id)
        
        form_data = {
            'player_id': player_id,
            'recent_points': player_info['recent_points'][-games:],
            'average_points': player_info['recent_form'],
            'total_points': player_info['total_points_season'],
            'games_analyzed': min(games, len(player_info['recent_points']))
        }
        
        return ApiResponse(
            success=True,
            data=form_data,
            message=f"Retrieved form data for player {player_id}"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get player form: {str(e)}")
