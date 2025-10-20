import React, { useState, useEffect } from 'react';
import {
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Box,
  Alert,
  CircularProgress,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  TablePagination,
  InputAdornment,
} from '@mui/material';
import { Search, TrendingUp, Star } from '@mui/icons-material';
import { getTopPicks, getCurrentGameweek } from '../services/api';

const GameweekPredictions: React.FC = () => {
  const [gameweek, setGameweek] = useState(1);
  const [predictions, setPredictions] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const loadPredictions = async (gw: number) => {
    try {
      setLoading(true);
      setError(null);
      const result = await getTopPicks(gw, 50);
      setPredictions(result);
      setPage(0); // Reset to first page
    } catch (err: any) {
      console.error('Predictions error:', err);
      setError(err instanceof Error ? err.message : 'Failed to load predictions');
      setPredictions([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Load current gameweek on mount
    const initializeGameweek = async () => {
      try {
        const gwData = await getCurrentGameweek();
        const gw = gwData.current_gameweek || 1;
        setGameweek(gw);
        loadPredictions(gw);
      } catch (err) {
        console.error('Failed to get current gameweek:', err);
        loadPredictions(gameweek);
      }
    };
    initializeGameweek();
  }, []);

  const handleGameweekChange = () => {
    loadPredictions(gameweek);
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const getRecommendationColor = (points: number) => {
    if (points >= 6) return 'success';
    if (points >= 4) return 'warning';
    return 'default';
  };

  const getRecommendationText = (points: number) => {
    if (points >= 6) return 'Strong Pick';
    if (points >= 4) return 'Consider';
    return 'Avoid';
  };

  // Calculate some stats
  const avgPrediction = predictions.length > 0 
    ? (predictions.reduce((sum, p) => sum + (p.predicted_points || 0), 0) / predictions.length).toFixed(1)
    : '0';
  
  const topPrediction = predictions.length > 0 
    ? Math.max(...predictions.map(p => p.predicted_points || 0)).toFixed(1)
    : '0';

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Gameweek Predictions
      </Typography>
      <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
        View predicted points for all players in a specific gameweek.
      </Typography>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} sm={6} md={4}>
              <TextField
                label="Gameweek"
                type="number"
                value={gameweek}
                onChange={(e) => setGameweek(parseInt(e.target.value) || 1)}
                inputProps={{ min: 1, max: 38 }}
                fullWidth
                InputProps={{
                  startAdornment: <InputAdornment position="start">#</InputAdornment>,
                }}
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={loading ? <CircularProgress size={20} /> : <Search />}
                onClick={handleGameweekChange}
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Get Predictions'}
              </Button>
            </Grid>
            <Grid item xs={12} md={4}>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center', py: 1 }}>
                      <Typography variant="h6" color="primary">
                        {topPrediction}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        Top Prediction
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
                <Grid item xs={6}>
                  <Card variant="outlined">
                    <CardContent sx={{ textAlign: 'center', py: 1 }}>
                      <Typography variant="h6" color="secondary">
                        {avgPrediction}
                      </Typography>
                      <Typography variant="caption" color="textSecondary">
                        Average
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Predictions Table */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <TrendingUp sx={{ color: 'primary.main' }} />
            <Typography variant="h5" component="h2">
              Gameweek {gameweek} Predictions
            </Typography>
            {predictions.length > 0 && (
              <Chip
                label={`${predictions.length} players`}
                color="primary"
                variant="outlined"
                size="small"
              />
            )}
          </Box>

          {loading ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <CircularProgress size={60} />
              <Typography variant="body1" sx={{ mt: 2 }}>
                Loading predictions...
              </Typography>
            </Box>
          ) : predictions.length > 0 ? (
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Rank</TableCell>
                    <TableCell>Player</TableCell>
                    <TableCell>Team</TableCell>
                    <TableCell>Position</TableCell>
                    <TableCell align="right">Price</TableCell>
                    <TableCell align="right">Predicted Points</TableCell>
                    <TableCell align="right">Form</TableCell>
                    <TableCell align="center">Recommendation</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {predictions
                    .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                    .map((prediction, index) => (
                      <TableRow
                        key={prediction.player_id}
                        sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                      >
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {page * rowsPerPage + index + 1}
                            {page * rowsPerPage + index < 3 && (
                              <Star sx={{ color: 'gold', fontSize: 20 }} />
                            )}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" fontWeight="medium">
                            {prediction.player_name || prediction.name || `Player ${prediction.player_id}`}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            {prediction.full_name || ''}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {prediction.team || 'Unknown'}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={prediction.position || 'UNK'} 
                            size="small" 
                            color={
                              prediction.position === 'FWD' ? 'error' :
                              prediction.position === 'MID' ? 'success' :
                              prediction.position === 'DEF' ? 'info' :
                              'default'
                            }
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2">
                            £{prediction.price?.toFixed(1) || 'N/A'}M
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="h6" color="primary">
                            {prediction.predicted_points?.toFixed(1) || 'N/A'}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">
                          <Typography variant="body2">
                            {prediction.recent_form?.toFixed(1) || 'N/A'}
                          </Typography>
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={getRecommendationText(prediction.predicted_points || 0)}
                            color={getRecommendationColor(prediction.predicted_points || 0) as any}
                            size="small"
                            variant="outlined"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                </TableBody>
              </Table>
              <TablePagination
                rowsPerPageOptions={[5, 10, 25, 50]}
                component="div"
                count={predictions.length}
                rowsPerPage={rowsPerPage}
                page={page}
                onPageChange={handleChangePage}
                onRowsPerPageChange={handleChangeRowsPerPage}
              />
            </TableContainer>
          ) : (
            <Box sx={{ textAlign: 'center', py: 8 }}>
              <Typography variant="h6" color="textSecondary">
                No predictions available
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                {loading ? 'Loading...' : 'Click "Get Predictions" to load data for this gameweek'}
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>
    </Box>
  );
};

export default GameweekPredictions;
