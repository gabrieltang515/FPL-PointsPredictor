import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Typography,
  Card,
  CardContent,
  Box,
  Alert,
  CircularProgress,
  Grid,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { Person, TrendingUp, Timeline } from '@mui/icons-material';
import { getPlayerDetails, getPlayerForm } from '../services/api';

const PlayerDetails: React.FC = () => {
  const { playerId } = useParams<{ playerId: string }>();
  const [playerInfo, setPlayerInfo] = useState<any>(null);
  const [playerForm, setPlayerForm] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadPlayerData = async () => {
      if (!playerId) return;

      try {
        setLoading(true);
        setError(null);

        const [detailsResult, formResult] = await Promise.all([
          getPlayerDetails(parseInt(playerId)),
          getPlayerForm(parseInt(playerId), 10)
        ]);

        setPlayerInfo(detailsResult);
        setPlayerForm(formResult);
      } catch (err: any) {
        console.error('Player data loading error:', err);
        setError(err instanceof Error ? err.message : 'Failed to load player data');
      } finally {
        setLoading(false);
      }
    };

    loadPlayerData();
  }, [playerId]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress size={60} />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ flexGrow: 1 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Player Details
      </Typography>

      {/* Player Header */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 3 }}>
            <Person sx={{ fontSize: 64, color: 'primary.main' }} />
            <Box>
              <Typography variant="h4" component="h2">
                Player {playerId}
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
                <Chip label="Position: Unknown" variant="outlined" />
                <Chip label="Team: Unknown" variant="outlined" />
              </Box>
            </Box>
          </Box>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        {/* Season Stats */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <TrendingUp sx={{ color: 'primary.main' }} />
                <Typography variant="h5" component="h2">
                  Season Statistics
                </Typography>
              </Box>

              {playerInfo && (
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 2 }}>
                        <Typography variant="h4" color="primary">
                          {playerInfo.total_points || 0}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Total Points
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 2 }}>
                        <Typography variant="h4" color="secondary">
                          {playerInfo.points_per_game?.toFixed(1) || '0.0'}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Points per Game
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 2 }}>
                        <Typography variant="h4" color="primary">
                          {playerInfo.games_played || 0}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Games Played
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={6}>
                    <Card variant="outlined">
                      <CardContent sx={{ textAlign: 'center', py: 2 }}>
                        <Typography variant="h4" color="secondary">
                          {playerInfo.form?.toFixed(1) || '0.0'}
                        </Typography>
                        <Typography variant="body2" color="textSecondary">
                          Recent Form
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Form */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
                <Timeline sx={{ color: 'secondary.main' }} />
                <Typography variant="h5" component="h2">
                  Recent Form
                </Typography>
              </Box>

              {playerForm && playerForm.recent_points && (
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Game</TableCell>
                        <TableCell align="right">Points</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {playerForm.recent_points.map((points: number, index: number) => (
                        <TableRow key={index}>
                          <TableCell>
                            Game -{playerForm.recent_points.length - index}
                          </TableCell>
                          <TableCell align="right">
                            <Chip
                              label={points}
                              color={points >= 6 ? 'success' : points >= 3 ? 'warning' : 'default'}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              )}

              {playerForm && (
                <Box sx={{ mt: 2, p: 2, backgroundColor: 'grey.50', borderRadius: 1 }}>
                  <Typography variant="body2" color="textSecondary">
                    <strong>Form Analysis:</strong><br />
                    Average Points: {playerForm.average_points?.toFixed(1) || 'N/A'}<br />
                    Games Analyzed: {playerForm.games_analyzed || 0}
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Additional Information */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h5" component="h2" gutterBottom>
                Additional Information
              </Typography>
              <Alert severity="info">
                This is a demo version. In a production app, this section would include:
                <ul>
                  <li>Upcoming fixtures and difficulty ratings</li>
                  <li>Injury status and news updates</li>
                  <li>Price changes and ownership statistics</li>
                  <li>Performance charts and trends</li>
                  <li>Head-to-head comparisons with similar players</li>
                </ul>
              </Alert>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PlayerDetails;
