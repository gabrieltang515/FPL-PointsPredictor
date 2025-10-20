import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

import Layout from './components/Layout/Layout';
import Dashboard from './pages/Dashboard';
import PlayerPredictions from './pages/PlayerPredictions';
import GameweekPredictions from './pages/GameweekPredictions';
import PlayerDetails from './pages/PlayerDetails';
import TeamBuilder from './pages/TeamBuilder';
import Transfers from './pages/Transfers';

// FPL-themed color scheme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#00ff87', // FPL green
      dark: '#00cc6a',
      light: '#4dff9e',
      contrastText: '#000000',
    },
    secondary: {
      main: '#37003c', // FPL purple
      dark: '#1a0018',
      light: '#6d2c5c',
      contrastText: '#ffffff',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
    text: {
      primary: '#37003c',
      secondary: '#666666',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontWeight: 700,
      color: '#37003c',
    },
    h2: {
      fontWeight: 600,
      color: '#37003c',
    },
    h3: {
      fontWeight: 600,
      color: '#37003c',
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <Layout>
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/player-predictions" element={<PlayerPredictions />} />
              <Route path="/gameweek-predictions" element={<GameweekPredictions />} />
              <Route path="/team-builder" element={<TeamBuilder />} />
              <Route path="/transfers" element={<Transfers />} />
              <Route path="/player/:playerId" element={<PlayerDetails />} />
            </Routes>
          </Layout>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;