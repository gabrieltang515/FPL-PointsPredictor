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
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  Tabs,
  Tab,
  Divider,
} from '@mui/material';
import { TrendingUp, SwapHoriz, Whatshot, Stars } from '@mui/icons-material';
import { getHotPicks, getDifferentialPicks, getCurrentGameweek } from '../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

const Transfers: React.FC = () => {
  const [gameweek, setGameweek] = useState(1);
  const [tabValue, setTabValue] = useState(0);
  const [hotPicks, setHotPicks] = useState<any[]>([]);
  const [differentials, setDifferentials] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const initData = async () => {
      try {
        const gwData = await getCurrentGameweek();
        const gw = gwData.current_gameweek || 1;
        setGameweek(gw);
        loadHotPicks(gw);
        loadDifferentials(gw);
      } catch (err) {
        console.error('Failed to initialize:', err);
      }
    };
    initData();
  }, []);

  const loadHotPicks = async (gw: number) => {
    try {
      setLoading(true);
      setError(null);
      const picks = await getHotPicks(gw, undefined, undefined, 30);
      setHotPicks(picks);
    } catch (err: unknown) {
      console.error('Hot picks error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load hot picks';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const loadDifferentials = async (gw: number) => {
    try {
      setLoading(true);
      setError(null);
      const picks = await getDifferentialPicks(gw, 5, 30);
      setDifferentials(picks);
    } catch (err: unknown) {
      console.error('Differentials error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to load differentials';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleGameweekChange = () => {
    loadHotPicks(gameweek);
    loadDifferentials(gameweek);
  };

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getValueRating = (valueScore: number) => {
    if (valueScore >= 1.5) return { label: 'Excellent', color: 'success' as const };
    if (valueScore >= 1.0) return { label: 'Good', color: 'info' as const };
    if (valueScore >= 0.7) return { label: 'Fair', color: 'warning' as const };
    return { label: 'Poor', color: 'error' as const };
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Transfer Suggestions
      </Typography>
      <Typography variant="body1" color="textSecondary" sx={{ mb: 4 }}>
        Discover the best transfer targets based on AI predictions and value analysis.
      </Typography>

      {/* Gameweek Selector */}
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
              />
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Button
                variant="contained"
                size="large"
                fullWidth
                startIcon={loading ? <CircularProgress size={20} /> : <TrendingUp />}
                onClick={handleGameweekChange}
                disabled={loading}
              >
                {loading ? 'Loading...' : 'Refresh Data'}
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

      {/* Tabs */}
      <Card>
        <CardContent>
          <Tabs value={tabValue} onChange={handleTabChange} aria-label="transfer tabs">
            <Tab icon={<Whatshot />} label="Hot Picks" iconPosition="start" />
            <Tab icon={<Stars />} label="Differentials" iconPosition="start" />
          </Tabs>

          {/* Hot Picks Tab */}
          <TabPanel value={tabValue} index={0}>
            <Typography variant="h6" gutterBottom>
              🔥 Best Value Players for Gameweek {gameweek}
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              Players with the highest predicted points per million spent (value score).
            </Typography>

            {loading ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : hotPicks.length > 0 ? (
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Rank</TableCell>
                      <TableCell>Player</TableCell>
                      <TableCell>Team</TableCell>
                      <TableCell>Position</TableCell>
                      <TableCell align="right">Price</TableCell>
                      <TableCell align="right">Predicted Pts</TableCell>
                      <TableCell align="right">Value Score</TableCell>
                      <TableCell align="center">Rating</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {hotPicks.map((player, index) => {
                      const valueRating = getValueRating(player.value_score || 0);
                      return (
                        <TableRow key={player.player_id}>
                          <TableCell>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              {index + 1}
                              {index < 3 && <Whatshot sx={{ color: 'orange', fontSize: 20 }} />}
                            </Box>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" fontWeight="medium">
                              {player.player_name || player.name}
                            </Typography>
                            <Typography variant="caption" color="textSecondary">
                              {player.selected_by?.toFixed(1)}% selected
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
                          <TableCell align="right">
                            <Typography variant="body2" fontWeight="bold">
                              {player.value_score?.toFixed(2)}
                            </Typography>
                          </TableCell>
                          <TableCell align="center">
                            <Chip
                              label={valueRating.label}
                              color={valueRating.color}
                              size="small"
                            />
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Alert severity="info">No hot picks available. Try refreshing the data.</Alert>
            )}
          </TabPanel>

          {/* Differentials Tab */}
          <TabPanel value={tabValue} index={1}>
            <Typography variant="h6" gutterBottom>
              ⭐ Low-Owned Gems for Gameweek {gameweek}
            </Typography>
            <Typography variant="body2" color="textSecondary" paragraph>
              High-scoring players with low ownership (&lt;5%) - perfect for gaining an edge!
            </Typography>

            {loading ? (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <CircularProgress />
              </Box>
            ) : differentials.length > 0 ? (
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Rank</TableCell>
                      <TableCell>Player</TableCell>
                      <TableCell>Team</TableCell>
                      <TableCell>Position</TableCell>
                      <TableCell align="right">Price</TableCell>
                      <TableCell align="right">Predicted Pts</TableCell>
                      <TableCell align="right">Ownership</TableCell>
                      <TableCell align="center">Potential</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {differentials.map((player, index) => (
                      <TableRow key={player.player_id}>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {index + 1}
                            {index < 3 && <Stars sx={{ color: 'gold', fontSize: 20 }} />}
                          </Box>
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2" fontWeight="medium">
                            {player.player_name || player.name}
                          </Typography>
                          <Typography variant="caption" color="textSecondary">
                            Form: {player.recent_form?.toFixed(1)}
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
                        <TableCell align="right">
                          <Chip
                            label={`${player.selected_by?.toFixed(1)}%`}
                            size="small"
                            color="success"
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell align="center">
                          <Chip
                            label={player.predicted_points >= 6 ? 'High' : player.predicted_points >= 4 ? 'Medium' : 'Low'}
                            color={
                              player.predicted_points >= 6 ? 'success' :
                              player.predicted_points >= 4 ? 'info' :
                              'default'
                            }
                            size="small"
                          />
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Alert severity="info">No differential picks available. Try refreshing the data.</Alert>
            )}
          </TabPanel>
        </CardContent>
      </Card>

      {/* Info Box */}
      <Card variant="outlined" sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Transfer Strategy Tips
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                🔥 Hot Picks
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Players offering the best value (predicted points per £1M). Great for maximizing your budget efficiency.
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="subtitle2" gutterBottom>
                ⭐ Differentials
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Low-owned players with high predicted points. Perfect for gaining an edge over your rivals in mini-leagues.
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Transfers;
