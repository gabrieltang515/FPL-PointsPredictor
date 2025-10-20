import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 503) {
      // Model not loaded
      throw new Error('Prediction service is currently unavailable. Please try again later.');
    }
    if (error.response?.status === 404) {
      throw new Error('Player not found.');
    }
    if (error.response?.status >= 500) {
      throw new Error('Server error. Please try again later.');
    }
    throw error;
  }
);

// Types
export interface PlayerPredictionRequest {
  player_id: number;
  gameweek: number;
  opponent_team: string;
  was_home: boolean;
  fixture_difficulty?: number;
}

export interface GameweekPredictionRequest {
  gameweek: number;
  include_unavailable?: boolean;
}

export interface PlayerPrediction {
  player_id: number;
  player_name: string;
  position: string;
  team: string;
  predicted_points: number;
  confidence_interval?: { [key: string]: number };
  form_rating?: string;
  recommendation?: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

export interface HealthCheck {
  status: string;
  model_loaded: boolean;
  version: string;
  timestamp: string;
}

export interface PlayerInfo {
  id: number;
  name: string;
  full_name?: string;
  position: string;
  team: string;
  team_id?: number;
  price: number;
  total_points: number;
  form: number;
  selected_by: number;
  points_per_game: number;
  status?: string;
  news?: string;
  minutes?: number;
  goals_scored?: number;
  assists?: number;
}

// API functions
export const healthCheck = async (): Promise<HealthCheck> => {
  const response = await api.get<HealthCheck>('/health');
  return response.data;
};

export const predictPlayer = async (request: PlayerPredictionRequest): Promise<any> => {
  const response = await api.post<ApiResponse>('/predict/player', request);
  return response.data;
};

export const predictGameweek = async (request: GameweekPredictionRequest): Promise<any> => {
  const response = await api.post<ApiResponse>('/predict/gameweek', request);
  return response.data;
};

export const getTopPicks = async (gameweek: number, limit: number = 10): Promise<any[]> => {
  const response = await api.get<ApiResponse>(`/predict/top-picks/${gameweek}?limit=${limit}`);
  return response.data.data || [];
};

export const getPlayers = async (limit: number = 50, position?: string): Promise<PlayerInfo[]> => {
  const params = new URLSearchParams({ limit: limit.toString() });
  if (position) params.append('position', position);
  
  const response = await api.get<ApiResponse>(`/players?${params}`);
  return response.data.data || [];
};

export const getPlayerDetails = async (playerId: number): Promise<any> => {
  const response = await api.get<ApiResponse>(`/players/${playerId}`);
  return response.data.data;
};

export const getPlayerForm = async (playerId: number, games: number = 5): Promise<any> => {
  const response = await api.get<ApiResponse>(`/players/${playerId}/form?games=${games}`);
  return response.data.data;
};

export const searchPlayers = async (query: string, limit: number = 20): Promise<PlayerInfo[]> => {
  const response = await api.get<ApiResponse>(`/players/search/${query}?limit=${limit}`);
  return response.data.data || [];
};

// Gameweek endpoints
export const getCurrentGameweek = async (): Promise<any> => {
  const response = await api.get<ApiResponse>('/gameweek/current');
  return response.data.data;
};

// Team builder endpoints
export const validateTeam = async (playerIds: number[]): Promise<any> => {
  const response = await api.post<ApiResponse>('/team/validate', { player_ids: playerIds });
  return response.data.data;
};

export const optimizeTeam = async (params: {
  gameweek: number;
  budget?: number;
  formation?: string;
  include_player_ids?: number[];
  exclude_player_ids?: number[];
}): Promise<any> => {
  const response = await api.post<ApiResponse>('/team/optimize', params);
  return response.data.data;
};

export const getStartingEleven = async (params: {
  squad_player_ids: number[];
  gameweek: number;
  formation: string;
  captain_id?: number;
}): Promise<any> => {
  const response = await api.post<ApiResponse>('/team/starting-eleven', params);
  return response.data.data;
};

// Transfer endpoints
export const suggestTransfers = async (params: {
  current_squad: number[];
  gameweek: number;
  num_transfers?: number;
  remaining_budget?: number;
}): Promise<any> => {
  const response = await api.post<ApiResponse>('/transfers/suggest', params);
  return response.data.data;
};

export const analyzeTransfer = async (params: {
  player_out_id: number;
  player_in_id: number;
  gameweek: number;
  current_squad: number[];
}): Promise<any> => {
  const response = await api.post<ApiResponse>('/transfers/analyze', params);
  return response.data.data;
};

export const getHotPicks = async (gameweek: number, position?: string, maxPrice?: number, limit: number = 20): Promise<any[]> => {
  let url = `/transfers/hot-picks/${gameweek}?limit=${limit}`;
  if (position) url += `&position=${position}`;
  if (maxPrice) url += `&max_price=${maxPrice}`;
  const response = await api.get<ApiResponse>(url);
  return response.data.data?.hot_picks || [];
};

export const getDifferentialPicks = async (gameweek: number, maxOwnership: number = 5, limit: number = 20): Promise<any[]> => {
  const response = await api.get<ApiResponse>(`/transfers/differential-picks/${gameweek}?max_ownership=${maxOwnership}&limit=${limit}`);
  return response.data.data?.differential_picks || [];
};

export default api;
