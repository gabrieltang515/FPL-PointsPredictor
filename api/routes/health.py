from fastapi import APIRouter, Request
from datetime import datetime
from api.models.prediction_models import HealthCheck, ApiResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthCheck)
async def health_check(request: Request):
    """Check API health and model status"""
    model_service = getattr(request.app.state, 'model_service', None)
    
    return HealthCheck(
        status="healthy",
        model_loaded=model_service.is_loaded if model_service else False,
        version="1.0.0",
        timestamp=datetime.now()
    )
