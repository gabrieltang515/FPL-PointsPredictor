import React, { useEffect, useState } from 'react';
import {
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Alert,
  CircularProgress,
  Chip,
  Button,
} from '@mui/material';
import {
  TrendingUp,
  Person,
  Scoreboard,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { healthCheck, getTopPicks, getCurrentGameweek, HealthCheck } from '../services/api';

const Dashboard: React.FC = () => {
  const [health, setHealth] = useState<HealthCheck | null>(null);
  const [topPicks, setTopPicks] = useState<any[]>([]);
  const [currentGameweek, setCurrentGameweek] = useState<number>(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setLoading(true);
        
        // Check API health
        const healthData = await healthCheck();
        setHealth(healthData);
        
        // Get current gameweek
        const gwData = await getCurrentGameweek();
        const gw = gwData.current_gameweek || 1;
        setCurrentGameweek(gw);
        
        // Load top picks for current gameweek
        if (healthData.model_loaded) {
          const picks = await getTopPicks(gw, 5);
          setTopPicks(picks);
        }
        
        setError(null);
      } catch (error) {
        console.error('Dashboard loading error:', error);
        const errorMessage = error instanceof Error ? (error as Error).message : 'Failed to load dashboard data';
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    loadDashboardData();
  }, []);

  const StatCard: React.FC<{ title: string; value: string; icon: React.ReactNode; color: string }> = ({
    title,
    value,
    icon,
    color,
  }) => (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box>
            <Typography color="textSecondary" gutterBottom variant="body2">
              {title}
            </Typography>
            <Typography variant="h4" component="div" sx={{ color }}>
              {value}
            </Typography>
          </Box>
          <Box sx={{ color, fontSize: 48 }}>{icon}</Box>
        </Box>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress size={60} />
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        FPL Predictor Dashboard
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Health Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {health?.model_loaded ? (
              <CheckCircle sx={{ color: 'success.main', fontSize: 32 }} />
            ) : (
              <Error sx={{ color: 'error.main', fontSize: 32 }} />
            )}
            <Box>
              <Typography variant="h6">
                Model Status: {health?.model_loaded ? 'Ready' : 'Loading'}
              </Typography>
              <Typography color="textSecondary">
                {health?.model_loaded
                  ? 'All prediction services are available'
                  : 'Prediction services are currently unavailable'}
              </Typography>
            </Box>
            <Box sx={{ ml: 'auto' }}>
              <Chip
                label={health?.status || 'Unknown'}
                color={health?.model_loaded ? 'success' : 'error'}
                variant="outlined"
              />
            </Box>
          </Box>
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="API Version"
            value={health?.version || '1.0.0'}
            icon={<TrendingUp />}
            color="primary.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Top Picks Available"
            value={topPicks.length.toString()}
            icon={<Person />}
            color="secondary.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={4}>
          <StatCard
            title="Current Gameweek"
            value={currentGameweek.toString()}
            icon={<Scoreboard />}
            color="primary.main"
          />
        </Grid>
      </Grid>

      {/* Top Picks Preview */}
      {health?.model_loaded && topPicks.length > 0 && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h5" component="h2" gutterBottom>
              Top Picks for Gameweek {currentGameweek}
            </Typography>
            <Grid container spacing={2}>
              {topPicks.slice(0, 5).map((pick, index) => (
                <Grid item xs={12} sm={6} md={4} key={pick.player_id}>
                  <Card variant="outlined">
                    <CardContent sx={{ py: 2 }}>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Box>
                          <Typography variant="h6">
                            {pick.player_name || pick.name || `Player ${pick.player_id}`}
                          </Typography>
                          <Typography color="textSecondary" variant="body2" sx={{ mb: 0.5 }}>
                            {pick.team} • {pick.position} • £{pick.price}M
                          </Typography>
                          <Typography color="primary" variant="body2" fontWeight="bold">
                            Predicted: {pick.predicted_points?.toFixed(1) || 'N/A'} pts
                          </Typography>
                        </Box>
                        <Chip
                          label={`#${index + 1}`}
                          color="primary"
                          size="small"
                        />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
            <Box sx={{ mt: 2, textAlign: 'center' }}>
              <Button
                variant="contained"
                onClick={() => navigate('/gameweek-predictions')}
              >
                View All Predictions
              </Button>
            </Box>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <Card>
        <CardContent>
          <Typography variant="h5" component="h2" gutterBottom>
            Quick Actions
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<Person />}
                onClick={() => navigate('/player-predictions')}
                disabled={!health?.model_loaded}
              >
                Player Predictions
              </Button>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<Scoreboard />}
                onClick={() => navigate('/gameweek-predictions')}
                disabled={!health?.model_loaded}
              >
                Gameweek Predictions
              </Button>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Button
                variant="outlined"
                fullWidth
                startIcon={<TrendingUp />}
                onClick={() => window.open('http://localhost:8000/docs', '_blank')}
              >
                API Documentation
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Dashboard;
