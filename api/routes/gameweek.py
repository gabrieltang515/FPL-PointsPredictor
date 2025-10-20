from fastapi import APIRouter, HTTPException, Request

from api.models.prediction_models import ApiResponse

router = APIRouter(tags=["Gameweek"])


@router.get("/gameweek/current")
async def get_current_gameweek(request: Request):
    """Get the current active gameweek from FPL API"""
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        current_gw = fpl_service.get_current_gameweek()
        
        # Get gameweek details
        bootstrap = fpl_service.get_bootstrap_data()
        events = bootstrap.get('events', [])
        
        current_event = next((e for e in events if e['id'] == current_gw), None)
        
        if current_event:
            return ApiResponse(
                success=True,
                data={
                    'current_gameweek': current_gw,
                    'name': current_event.get('name'),
                    'deadline_time': current_event.get('deadline_time'),
                    'finished': current_event.get('finished', False),
                    'is_current': current_event.get('is_current', False),
                    'is_next': current_event.get('is_next', False)
                },
                message=f"Current gameweek is {current_gw}"
            )
        else:
            return ApiResponse(
                success=True,
                data={'current_gameweek': current_gw},
                message=f"Current gameweek is {current_gw}"
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get current gameweek: {str(e)}")


@router.get("/gameweek/all")
async def get_all_gameweeks(request: Request):
    """Get information about all gameweeks"""
    try:
        model_service = request.app.state.model_service
        fpl_service = model_service.fpl_service
        
        bootstrap = fpl_service.get_bootstrap_data()
        events = bootstrap.get('events', [])
        
        gameweeks = []
        for event in events:
            gameweeks.append({
                'id': event.get('id'),
                'name': event.get('name'),
                'deadline_time': event.get('deadline_time'),
                'finished': event.get('finished', False),
                'is_current': event.get('is_current', False),
                'is_next': event.get('is_next', False),
                'highest_score': event.get('highest_score'),
                'average_score': event.get('average_entry_score')
            })
        
        return ApiResponse(
            success=True,
            data={'gameweeks': gameweeks},
            message=f"Retrieved {len(gameweeks)} gameweeks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get gameweeks: {str(e)}")
