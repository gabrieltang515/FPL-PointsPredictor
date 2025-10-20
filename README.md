# FPL Points Predictor

A full-stack web application for predicting Fantasy Premier League (FPL) player points using machine learning with **live FPL API integration**, team building, and transfer suggestions.

## 🎯 Key Features

✅ **Live FPL Data** - Real-time player stats, prices, and fixtures from the official FPL API
✅ **AI Predictions** - LightGBM model with 0.89 R² score predicting player points
✅ **Team Builder** - Generate optimal 15-player squads within £100M budget
✅ **Transfer Suggestions** - Smart recommendations based on predicted points and value
✅ **No API Key Required** - All FPL data is free and publicly accessible!

## Project Structure

```
fpl-points-predictor/
├── api/                    # FastAPI backend
│   ├── main.py            # API entry point
│   ├── models/            # Pydantic models
│   ├── routes/            # API endpoints
│   └── services/          # Business logic
├── frontend/              # React TypeScript app
│   ├── src/
│   │   ├── components/    # Reusable components
│   │   ├── pages/         # Page components
│   │   └── services/      # API client
│   └── package.json
├── src/                   # ML pipeline (existing)
├── models/                # Trained models
├── data/                  # Dataset
└── requirements.txt       # ML dependencies
```

## Features

### Backend (FastAPI)
- **Live FPL Data Service**: Fetches and caches current season data from official FPL API
- **Smart Predictions**: ML-powered predictions with live player metadata and fixtures
- **Team Builder API**: Optimize squads with budget and composition constraints
- **Transfer Engine**: Suggest best transfers based on predicted value
- **Player Search**: Find players by name with autocomplete
- **Current Gameweek Detection**: Automatically fetches active gameweek
- **Health Monitoring**: API status and model availability
- **Automatic Documentation**: Swagger UI at `/docs`

### Frontend (React + TypeScript + Material-UI)
- **Dashboard**: Live stats with current gameweek and top predictions
- **Player Predictions**: Individual player predictions with real names and fixtures
- **Gameweek Predictions**: Sortable table with prices, positions, and teams
- **Team Builder**: AI-generated optimal squads within budget constraints
- **Transfer Suggestions**: Hot picks and differential players for each gameweek
- **Live Player Data**: Real player names, teams, positions, prices, and availability
- **Responsive Design**: Works on desktop and mobile
- **FPL-themed UI**: Authentic green and purple color scheme

## Quick Start

### 1. Start the API Server

```bash
# Install API dependencies
cd api
pip install -r requirements.txt

# Run the FastAPI server
cd ..
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

### 2. Start the React Frontend

```bash
# Install frontend dependencies
cd frontend
npm install

# Start the development server
npm start
```

The frontend will be available at http://localhost:3000

## API Endpoints

### Health
- `GET /api/v1/health` - Check API and model status

### Gameweek
- `GET /api/v1/gameweek/current` - Get current active gameweek
- `GET /api/v1/gameweek/all` - Get all gameweeks with deadlines

### Predictions
- `POST /api/v1/predict/player` - Predict points for a specific player (auto-fetches fixtures)
- `POST /api/v1/predict/gameweek` - Get predictions for all available players
- `GET /api/v1/predict/top-picks/{gameweek}` - Get top predicted players

### Players
- `GET /api/v1/players` - List all players with live FPL data (filter by position, price)
- `GET /api/v1/players/search/{query}` - Search players by name
- `GET /api/v1/players/{player_id}` - Get comprehensive player details
- `GET /api/v1/players/{player_id}/form` - Get recent form data

### Team Builder
- `POST /api/v1/team/validate` - Validate team against FPL rules
- `POST /api/v1/team/optimize` - Generate AI-optimized squad (15 players, £100M budget)
- `POST /api/v1/team/starting-eleven` - Select best starting XI from squad

### Transfers
- `POST /api/v1/transfers/suggest` - Get smart transfer recommendations
- `POST /api/v1/transfers/analyze` - Analyze specific transfer decision
- `GET /api/v1/transfers/hot-picks/{gameweek}` - Best value players (pts per £M)
- `GET /api/v1/transfers/differential-picks/{gameweek}` - Low-owned gems (<5% ownership)

## Model Performance

Enhanced LightGBM model with feature engineering:
- **MAE**: ~0.23 points (mean absolute error)
- **R²**: ~0.89 (explains 89% of variance)
- **RMSE**: ~0.92 points (root mean squared error)

The model uses:
- Rolling averages (3-game windows)
- Expected goals (xG) and assists (xA)
- Fixture difficulty ratings
- Home/away performance
- Recent form and consistency metrics

## Development

### Adding New Features

1. **Backend**: Add new routes in `api/routes/`
2. **Frontend**: Add new pages in `frontend/src/pages/`
3. **API Client**: Update `frontend/src/services/api.ts`

### Environment Variables

Create `.env` files for configuration:

**Frontend** (`frontend/.env`):
```
REACT_APP_API_URL=http://localhost:8000/api/v1
```

### Model Updates

The API automatically loads the best available model from the `models/` directory:
1. `lgbm_enhanced.joblib` (preferred)
2. `lgbm_enhanced_local.joblib`
3. `lgbm_enhanced_lightweight.joblib`
4. `best_single.joblib`
5. `lgbm.joblib`

## Deployment

### Production Setup

1. **Backend**: Deploy FastAPI with Gunicorn + Nginx
2. **Frontend**: Build and serve with nginx or CDN
3. **Database**: Add PostgreSQL for player data
4. **Cache**: Add Redis for prediction caching

### Docker (Future Enhancement)

```dockerfile
# Example Dockerfile structure
FROM python:3.9-slim
WORKDIR /app
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
RUN pip install -r api/requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## What's New in v2.0 🚀

### ✅ Completed
- **Live FPL API Integration**: Real-time data from official API (no auth required!)
- **Player Metadata**: Actual names, teams, positions, prices, and availability
- **Current Gameweek Detection**: Automatically tracks the active gameweek
- **Fixture Integration**: Predictions use real opponent data and difficulty ratings
- **Team Builder**: AI-powered squad optimization with budget constraints
- **Transfer Suggestions**: Smart recommendations for hot picks and differentials
- **Enhanced Frontend**: Professional UI with live player data throughout

### 🔜 Future Enhancements
1. **User Authentication**: Save personal teams and track performance
2. **Price Change Predictions**: Forecast player price rises/falls
3. **Captain Recommendations**: AI-powered captaincy suggestions
4. **Mini-League Integration**: Compare with friends
5. **Historical Performance Tracking**: Track prediction accuracy over time
6. **Mobile App**: Native or PWA version
7. **Database**: PostgreSQL for persistence and faster queries

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details
