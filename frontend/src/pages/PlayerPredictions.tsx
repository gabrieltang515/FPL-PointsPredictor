import React, { useState } from 'react';
import {
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Grid,
  Box,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Chip,
} from '@mui/material';
import { Search, TrendingUp } from '@mui/icons-material';
import { predictPlayer, PlayerPredictionRequest } from '../services/api';

const PlayerPredictions: React.FC = () => {
  const [formData, setFormData] = useState<PlayerPredictionRequest>({
    player_id: 0,
    gameweek: 20,
    opponent_team: '',
    was_home: true,
    fixture_difficulty: 3,
  });
  const [prediction, setPrediction] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (field: keyof PlayerPredictionRequest, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handlePredict = async () => {
    if (!formData.player_id || !formData.opponent_team) {
      setError('Please fill in all required fields');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const result = await predictPlayer(formData);
      setPrediction(result.data);
    } catch (err: any) {
      console.error('Prediction error:', err);
      setError(err instanceof Error ? err.message : 'Failed to get prediction');
      setPrediction(null);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceColor = (confidence: string) => {
    switch (confidence?.toLowerCase()) {
      case 'high': return 'success';
      case 'medium': return 'warning';
      case 'low': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Player Predictions
      </Typography>
      <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
        Get point predictions for individual players based on their form and upcoming fixtures.
      </Typography>

      <Grid container spacing={3}>
        {/* Prediction Form */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h5" component="h2" gutterBottom>
                Prediction Parameters
              </Typography>

              {error && (
                <Alert severity="error" sx={{ mb: 3 }}>
                  {error}
                </Alert>
              )}

              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <TextField
                    label="Player ID"
                    type="number"
                    fullWidth
                    value={formData.player_id || ''}
                    onChange={(e) => handleInputChange('player_id', parseInt(e.target.value) || 0)}
                    helperText="Enter the FPL player ID (e.g., 123)"
                    required
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <TextField
                    label="Gameweek"
                    type="number"
                    fullWidth
                    value={formData.gameweek}
                    onChange={(e) => handleInputChange('gameweek', parseInt(e.target.value) || 1)}
                    inputProps={{ min: 1, max: 38 }}
                  />
                </Grid>

                <Grid item xs={12} sm={6}>
                  <FormControl fullWidth>
                    <InputLabel>Fixture Difficulty</InputLabel>
                    <Select
                      value={formData.fixture_difficulty}
                      onChange={(e) => handleInputChange('fixture_difficulty', e.target.value)}
                      label="Fixture Difficulty"
                    >
                      <MenuItem value={1}>1 - Very Easy</MenuItem>
                      <MenuItem value={2}>2 - Easy</MenuItem>
                      <MenuItem value={3}>3 - Average</MenuItem>
                      <MenuItem value={4}>4 - Hard</MenuItem>
                      <MenuItem value={5}>5 - Very Hard</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>

                <Grid item xs={12}>
                  <TextField
                    label="Opponent Team"
                    fullWidth
                    value={formData.opponent_team}
                    onChange={(e) => handleInputChange('opponent_team', e.target.value)}
                    placeholder="e.g., Manchester City"
                    required
                  />
                </Grid>

                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={formData.was_home}
                        onChange={(e) => handleInputChange('was_home', e.target.checked)}
                      />
                    }
                    label="Playing at Home"
                  />
                </Grid>

                <Grid item xs={12}>
                  <Button
                    variant="contained"
                    size="large"
                    fullWidth
                    startIcon={loading ? <CircularProgress size={20} /> : <Search />}
                    onClick={handlePredict}
                    disabled={loading || !formData.player_id || !formData.opponent_team}
                  >
                    {loading ? 'Predicting...' : 'Get Prediction'}
                  </Button>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Prediction Results */}
        <Grid item xs={12} lg={6}>
          {prediction ? (
            <Card>
              <CardContent>
                <Typography variant="h5" component="h2" gutterBottom>
                  Prediction Results
                </Typography>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                  <TrendingUp sx={{ fontSize: 48, color: 'primary.main' }} />
                  <Box>
                    <Typography variant="h3" component="div" color="primary.main">
                      {prediction.predicted_points}
                    </Typography>
                    <Typography variant="body1" color="textSecondary">
                      Predicted Points
                    </Typography>
                  </Box>
                </Box>

                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 2 }}>
                        <Typography variant="h6" color="primary">
                          {prediction.recent_form}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Recent Form
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 2 }}>
                        <Typography variant="h6" color="secondary">
                          {prediction.consistency_score}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Consistency
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                <Box sx={{ mt: 3, textAlign: 'center' }}>
                  <Chip
                    label={`${prediction.confidence || 'Medium'} Confidence`}
                    color={getConfidenceColor(prediction.confidence) as any}
                    size="medium"
                    variant="outlined"
                  />
                </Box>

                <Box sx={{ mt: 3, p: 2, backgroundColor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="body2" color="textSecondary">
                    <strong>Prediction Details:</strong><br />
                    Player ID: {prediction.player_id}<br />
                    Based on recent form and fixture difficulty analysis
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          ) : (
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 8 }}>
                <Typography variant="h6" color="textSecondary">
                  Enter player details to see prediction
                </Typography>
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  Fill out the form on the left and click "Get Prediction"
                </Typography>
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default PlayerPredictions;
