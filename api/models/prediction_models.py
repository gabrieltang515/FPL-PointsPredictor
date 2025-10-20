from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PlayerPredictionRequest(BaseModel):
    """Request model for single player prediction"""
    player_id: int = Field(..., description="FPL player ID")
    gameweek: int = Field(..., description="Target gameweek for prediction")
    opponent_team: str = Field(..., description="Opponent team name")
    was_home: bool = Field(..., description="Whether playing at home")
    fixture_difficulty: Optional[int] = Field(3, description="Fixture difficulty (1-5)")

class GameweekPredictionRequest(BaseModel):
    """Request model for gameweek predictions"""
    gameweek: int = Field(..., description="Target gameweek for predictions")
    include_unavailable: bool = Field(False, description="Include injured/suspended players")

class PlayerPrediction(BaseModel):
    """Response model for player prediction"""
    player_id: int
    player_name: str
    position: str
    team: str
    predicted_points: float
    confidence_interval: Optional[Dict[str, float]] = None
    form_rating: Optional[str] = None
    recommendation: Optional[str] = None

class GameweekPredictions(BaseModel):
    """Response model for gameweek predictions"""
    gameweek: int
    predictions: List[PlayerPrediction]
    top_picks: List[PlayerPrediction]
    generated_at: datetime

class PlayerInfo(BaseModel):
    """Model for player information"""
    id: int
    name: str
    position: str
    team: str
    current_price: Optional[float] = None
    total_points: int
    form: Optional[float] = None
    selected_by_percent: Optional[float] = None
    points_per_game: Optional[float] = None

class PlayerDetailedInfo(PlayerInfo):
    """Extended player information with recent form"""
    recent_points: List[int]
    recent_minutes: List[int]
    next_fixtures: List[Dict[str, Any]]
    injury_status: Optional[str] = None
    news: Optional[str] = None

class ApiResponse(BaseModel):
    """Generic API response wrapper"""
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime
