from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes import predictions, players, health, test_predictions, team_builder, transfers, gameweek
from api.services.model_service import ModelService
from api.services.fpl_data_service import get_fpl_service

# Initialize FastAPI app
app = FastAPI(
    title="FPL Points Predictor API",
    description="API for predicting Fantasy Premier League player points using LightGBM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service on startup
@app.on_event("startup")
async def startup_event():
    """Load the trained model and FPL data on startup"""
    try:
        # Initialize FPL data service and fetch initial data
        print("🔄 Initializing FPL data service...")
        fpl_service = get_fpl_service()
        fpl_service.get_bootstrap_data()  # Fetch and cache data
        print("✅ FPL data service initialized")
        
        # Load ML model
        print("🔄 Loading ML model...")
        model_service = ModelService()
        model_service.load_model()
        app.state.model_service = model_service
        print("✅ Model loaded successfully")
        
        # Log current gameweek
        current_gw = fpl_service.get_current_gameweek()
        print(f"📅 Current gameweek: {current_gw}")
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        raise e

# Include routers
app.include_router(health.router, prefix="/api/v1")
app.include_router(predictions.router, prefix="/api/v1")
app.include_router(players.router, prefix="/api/v1")
app.include_router(test_predictions.router, prefix="/api/v1")
app.include_router(gameweek.router, prefix="/api/v1")
app.include_router(team_builder.router, prefix="/api/v1")
app.include_router(transfers.router, prefix="/api/v1")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "FPL Points Predictor API",
        "version": "1.0.0",
        "docs": "/docs"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
