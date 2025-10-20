import React, { useState, useEffect } from 'react';
import {
  Typography,
  Card,
  CardContent,
  Button,
  Box,
  Alert,
  CircularProgress,
  Grid,
  TextField,
  MenuItem,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { AutoAwesome, Add, Delete } from '@mui/icons-material';
import { optimizeTeam, getCurrentGameweek, searchPlayers } from '../services/api';

const TeamBuilder: React.FC = () => {
  const [gameweek, setGameweek] = useState(1);
  const [budget, setBudget] = useState(100);
  const [formation, setFormation] = useState('3-4-3');
  const [optimizedTeam, setOptimizedTeam] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initGameweek = async () => {
      try {
        const gwData = await getCurrentGameweek();
        setGameweek(gwData.current_gameweek || 1);
      } catch (err) {
        console.error('Failed to get current gameweek:', err);
      }
    };
    initGameweek();
  }, []);

  const handleOptimize = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await optimizeTeam({
        gameweek,
        budget,
        formation
      });
      
      setOptimizedTeam(result);
    } catch (err: unknown) {
      console.error('Team optimization error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to optimize team';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const formations = ['3-4-3', '3-5-2', '4-3-3', '4-4-2', '4-5-1', '5-3-2', '5-4-1'];

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Team Builder
      </Typography>
      <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
        Build an AI-optimized squad within budget constraints using predicted points.
      </Typography>

      {/* Configuration Card */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Gameweek"
                type="number"
                value={gameweek}
                onChange={(e) => setGameweek(parseInt(e.target.value) || 1)}
                inputProps={{ min: 1, max: 38 }}
                fullWidth
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Budget (£M)"
                type="number"
                value={budget}
                onChange={(e) => setBudget(parseFloat(e.target.value) || 100)}
                inputProps={{ min: 0, max: 100, step: 0.5 }}
                fullWidth
              />
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <TextField
                label="Formation"
                select
                value={formation}
                onChange={(e) => setFormation(e.target.value)}
                fullWidth
              >
                {formations.map((f) => (
                  <MenuItem key={f} value={f}>
                    {f}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={loading ? <CircularProgress size={20} /> : <AutoAwesome />}
                onClick={handleOptimize}
                disabled={loading}
                sx={{ height: '56px' }}
              >
                {loading ? 'Optimizing...' : 'Generate Team'}
              </Button>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Results */}
      {optimizedTeam && (
        <>
          {/* Summary Stats */}
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h5" color="primary">
                    {optimizedTeam.total_players}/15
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Players Selected
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h5" color="success.main">
                    £{optimizedTeam.total_cost?.toFixed(1)}M
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Total Cost
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h5" color="info.main">
                    £{optimizedTeam.remaining_budget?.toFixed(1)}M
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Remaining
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined">
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h5" color="secondary.main">
                    {optimizedTeam.total_predicted_points?.toFixed(1)}
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Predicted Points
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>

          {/* Status Messages */}
          {!optimizedTeam.is_complete && (
            <Alert severity="warning" sx={{ mb: 3 }}>
              Could not complete a full 15-player squad with the given constraints. 
              Try increasing the budget or adjusting formation.
            </Alert>
          )}

          {optimizedTeam.is_complete && (
            <Alert severity="success" sx={{ mb: 3 }}>
              ✅ Complete squad generated! Your team is ready for gameweek {gameweek}.
            </Alert>
          )}

          {/* Team Table */}
          <Card>
            <CardContent>
              <Typography variant="h5" component="h2" gutterBottom>
                Your Optimized Squad
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Player</TableCell>
                      <TableCell>Team</TableCell>
                      <TableCell>Position</TableCell>
                      <TableCell align="right">Price</TableCell>
                      <TableCell align="right">Predicted Points</TableCell>
                      <TableCell align="right">Form</TableCell>
                      <TableCell align="center">Fixture</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {optimizedTeam.team?.map((player: any) => (
                      <TableRow key={player.player_id}>
                        <TableCell>
                          <Typography variant="body2" fontWeight="medium">
                            {player.player_name || player.name}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            {player.full_name}
                          </Typography>
                        </TableCell>
                        <TableCell>{player.team}</TableCell>
                        <TableCell>
                          <Chip
                            label={player.position}
                            size="small"
                            color={
                              player.position === 'FWD' ? 'error' :
                              player.position === 'MID' ? 'success' :
                              player.position === 'DEF' ? 'info' :
                              'default'
                            }
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell align="right">£{player.price?.toFixed(1)}M</TableCell>
                        <TableCell align="right">
                          <Typography variant="body2" fontWeight="bold" color="primary">
                            {player.predicted_points?.toFixed(1)}
                          </Typography>
                        </TableCell>
                        <TableCell align="right">{player.recent_form?.toFixed(1)}</TableCell>
                        <TableCell align="center">
                          {player.fixture && (
                            <Chip
                              label={`${player.fixture.is_home ? 'vs' : '@'} ${player.fixture.opponent}`}
                              size="small"
                              color={
                                player.fixture.difficulty <= 2 ? 'success' :
                                player.fixture.difficulty >= 4 ? 'error' :
                                'default'
                              }
                            />
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>

              {/* Position Breakdown */}
              <Box sx={{ mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Squad Composition
                </Typography>
                <Grid container spacing={2}>
                  {Object.entries(optimizedTeam.position_counts || {}).map(([position, count]) => (
                    <Grid item key={position}>
                      <Chip
                        label={`${count} ${position}`}
                        color={
                          position === 'FWD' ? 'error' :
                          position === 'MID' ? 'success' :
                          position === 'DEF' ? 'info' :
                          'default'
                        }
                      />
                    </Grid>
                  ))}
                </Grid>
              </Box>
            </CardContent>
          </Card>
        </>
      )}

      {/* Help Text */}
      {!optimizedTeam && !loading && (
        <Card variant="outlined">
          <CardContent>
            <Typography variant="h6" gutterBottom>
              How It Works
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Our AI-powered team builder uses machine learning predictions to create an optimal squad:
            </Typography>
            <ul>
              <li>
                <Typography variant="body2" color="textSecondary">
                  Maximizes predicted points within your budget
                </Typography>
              </li>
              <li>
                <Typography variant="body2" color="textSecondary">
                  Ensures valid FPL squad composition (2 GKP, 5 DEF, 5 MID, 3 FWD)
                </Typography>
              </li>
              <li>
                <Typography variant="body2" color="textSecondary">
                  Respects the 3-player limit per team
                </Typography>
              </li>
              <li>
                <Typography variant="body2" color="textSecondary">
                  Considers fixture difficulty and player form
                </Typography>
              </li>
            </ul>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default TeamBuilder;
