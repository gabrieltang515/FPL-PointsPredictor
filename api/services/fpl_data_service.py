"""
FPL Data Service - Fetches and caches live data from the official FPL API
"""

import requests
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class FPLDataService:
    """Service for fetching and caching live FPL data"""
    
    # FPL API endpoints (free, no auth required!)
    BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
    ELEMENT_SUMMARY_URL = "https://fantasy.premierleague.com/api/element-summary/{}/"
    
    def __init__(self, cache_duration_minutes: int = 60):
        self.cache_duration = timedelta(minutes=cache_duration_minutes)
        self._bootstrap_cache = None
        self._bootstrap_cache_time = None
        self._fixtures_cache = None
        self._fixtures_cache_time = None
        
        # Indexed data structures for fast lookups
        self.players_by_id: Dict[int, Dict] = {}
        self.teams_by_id: Dict[int, Dict] = {}
        self.positions_by_id: Dict[int, Dict] = {}
        
    def _is_cache_valid(self, cache_time: Optional[datetime]) -> bool:
        """Check if cache is still valid"""
        if cache_time is None:
            return False
        return datetime.now() - cache_time < self.cache_duration
    
    def _fetch_json(self, url: str, timeout: int = 10) -> Dict:
        """Fetch JSON data from URL with error handling"""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise
    
    def get_bootstrap_data(self, force_refresh: bool = False) -> Dict:
        """
        Get bootstrap-static data (players, teams, gameweeks, etc.)
        This is cached for performance.
        """
        if force_refresh or not self._is_cache_valid(self._bootstrap_cache_time):
            logger.info("Fetching fresh bootstrap data from FPL API...")
            self._bootstrap_cache = self._fetch_json(self.BOOTSTRAP_URL)
            self._bootstrap_cache_time = datetime.now()
            
            # Index the data for fast lookups
            self._index_bootstrap_data()
            logger.info("✅ Bootstrap data loaded and indexed")
        
        return self._bootstrap_cache
    
    def _index_bootstrap_data(self):
        """Index bootstrap data for fast lookups"""
        if not self._bootstrap_cache:
            return
        
        # Index players
        self.players_by_id = {
            player['id']: player 
            for player in self._bootstrap_cache.get('elements', [])
        }
        
        # Index teams
        self.teams_by_id = {
            team['id']: team 
            for team in self._bootstrap_cache.get('teams', [])
        }
        
        # Index positions
        self.positions_by_id = {
            pos['id']: pos 
            for pos in self._bootstrap_cache.get('element_types', [])
        }
        
        logger.info(f"Indexed {len(self.players_by_id)} players, {len(self.teams_by_id)} teams")
    
    def get_fixtures(self, force_refresh: bool = False) -> List[Dict]:
        """Get all fixtures with caching"""
        if force_refresh or not self._is_cache_valid(self._fixtures_cache_time):
            logger.info("Fetching fixtures from FPL API...")
            self._fixtures_cache = self._fetch_json(self.FIXTURES_URL)
            self._fixtures_cache_time = datetime.now()
            logger.info(f"✅ Loaded {len(self._fixtures_cache)} fixtures")
        
        return self._fixtures_cache
    
    def get_current_gameweek(self) -> int:
        """Get the current active gameweek"""
        data = self.get_bootstrap_data()
        events = data.get('events', [])
        
        # Find the current or next gameweek
        for event in events:
            if event.get('is_current', False):
                return event['id']
            if event.get('is_next', False):
                return event['id']
        
        # Fallback: find the first future gameweek
        for event in events:
            if not event.get('finished', False):
                return event['id']
        
        # Default to gameweek 1 if nothing found
        return 1
    
    def get_player_metadata(self, player_id: int) -> Optional[Dict]:
        """
        Get comprehensive player metadata by ID
        Returns: {
            'id', 'name', 'web_name', 'team', 'position', 'price', 
            'total_points', 'form', 'selected_by', 'availability', ...
        }
        """
        # Ensure data is loaded
        if not self.players_by_id:
            self.get_bootstrap_data()
        
        player = self.players_by_id.get(player_id)
        if not player:
            return None
        
        team = self.teams_by_id.get(player.get('team'))
        position = self.positions_by_id.get(player.get('element_type'))
        
        return {
            'id': player['id'],
            'name': player.get('web_name', f"Player {player_id}"),
            'full_name': f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            'team': team.get('short_name', 'Unknown') if team else 'Unknown',
            'team_id': player.get('team'),
            'position': position.get('singular_name_short', 'UNK') if position else 'UNK',
            'position_id': player.get('element_type'),
            'price': player.get('now_cost', 0) / 10,  # Price is in tenths
            'total_points': player.get('total_points', 0),
            'form': float(player.get('form', 0)),
            'points_per_game': float(player.get('points_per_game', 0)),
            'selected_by': float(player.get('selected_by_percent', 0)),
            'status': player.get('status', 'a'),  # a=available, d=doubtful, i=injured, etc.
            'news': player.get('news', ''),
            'chance_of_playing': player.get('chance_of_playing_next_round'),
            'minutes': player.get('minutes', 0),
            'goals_scored': player.get('goals_scored', 0),
            'assists': player.get('assists', 0),
            'clean_sheets': player.get('clean_sheets', 0),
            'bonus': player.get('bonus', 0),
        }
    
    def get_all_players(self, position: Optional[str] = None, 
                       available_only: bool = True) -> List[Dict]:
        """
        Get all players with metadata
        Args:
            position: Filter by position (GKP, DEF, MID, FWD)
            available_only: Only return available players (not injured/suspended)
        """
        if not self.players_by_id:
            self.get_bootstrap_data()
        
        players = []
        for player_id in self.players_by_id.keys():
            metadata = self.get_player_metadata(player_id)
            if not metadata:
                continue
            
            # Apply filters
            if position and metadata['position'] != position:
                continue
            if available_only and metadata['status'] != 'a':
                continue
            
            players.append(metadata)
        
        return players
    
    def get_player_fixtures(self, player_id: int, 
                           num_fixtures: int = 5) -> List[Dict]:
        """
        Get upcoming fixtures for a player
        Returns list of fixtures with opponent, difficulty, and home/away
        """
        player = self.get_player_metadata(player_id)
        if not player:
            return []
        
        team_id = player['team_id']
        fixtures = self.get_fixtures()
        
        # Filter fixtures for this team that haven't been played
        upcoming = []
        for fixture in fixtures:
            if fixture.get('finished', False):
                continue
            
            is_home = fixture.get('team_h') == team_id
            is_away = fixture.get('team_a') == team_id
            
            if not (is_home or is_away):
                continue
            
            opponent_id = fixture.get('team_a') if is_home else fixture.get('team_h')
            opponent = self.teams_by_id.get(opponent_id, {})
            
            difficulty = fixture.get('team_h_difficulty') if is_home else fixture.get('team_a_difficulty')
            
            upcoming.append({
                'gameweek': fixture.get('event'),
                'opponent': opponent.get('short_name', 'Unknown'),
                'opponent_id': opponent_id,
                'is_home': is_home,
                'difficulty': difficulty,
                'kickoff_time': fixture.get('kickoff_time'),
            })
        
        # Sort by gameweek and limit
        upcoming.sort(key=lambda x: x['gameweek'] or 999)
        return upcoming[:num_fixtures]
    
    def get_fixture_for_gameweek(self, player_id: int, 
                                  gameweek: int) -> Optional[Dict]:
        """Get the fixture for a specific player in a specific gameweek"""
        fixtures = self.get_player_fixtures(player_id, num_fixtures=10)
        for fixture in fixtures:
            if fixture['gameweek'] == gameweek:
                return fixture
        return None
    
    def search_players(self, query: str, limit: int = 20) -> List[Dict]:
        """
        Search for players by name
        Returns list of matching players with metadata
        """
        if not self.players_by_id:
            self.get_bootstrap_data()
        
        query_lower = query.lower()
        matches = []
        
        for player_id in self.players_by_id.keys():
            metadata = self.get_player_metadata(player_id)
            if not metadata:
                continue
            
            # Search in both web_name and full_name
            if (query_lower in metadata['name'].lower() or 
                query_lower in metadata['full_name'].lower()):
                matches.append(metadata)
        
        # Sort by total points (best players first)
        matches.sort(key=lambda x: x['total_points'], reverse=True)
        return matches[:limit]
    
    def get_top_players(self, limit: int = 100, 
                       position: Optional[str] = None) -> List[Dict]:
        """Get top players by total points"""
        players = self.get_all_players(position=position)
        players.sort(key=lambda x: x['total_points'], reverse=True)
        return players[:limit]
    
    def validate_team_budget(self, player_ids: List[int]) -> Tuple[bool, float, str]:
        """
        Validate if a team is within budget (£100M)
        Returns: (is_valid, total_cost, message)
        """
        BUDGET_LIMIT = 100.0
        total_cost = 0.0
        
        for player_id in player_ids:
            player = self.get_player_metadata(player_id)
            if not player:
                return False, 0.0, f"Player {player_id} not found"
            total_cost += player['price']
        
        is_valid = total_cost <= BUDGET_LIMIT
        message = "Within budget" if is_valid else f"Over budget by £{total_cost - BUDGET_LIMIT:.1f}M"
        
        return is_valid, total_cost, message
    
    def validate_team_composition(self, player_ids: List[int]) -> Tuple[bool, str]:
        """
        Validate team composition rules:
        - Exactly 15 players
        - 2 GKP, 5 DEF, 5 MID, 3 FWD
        - Max 3 players per team
        """
        if len(player_ids) != 15:
            return False, f"Team must have 15 players (has {len(player_ids)})"
        
        # Count positions
        position_counts = {'GKP': 0, 'DEF': 0, 'MID': 0, 'FWD': 0}
        team_counts = {}
        
        for player_id in player_ids:
            player = self.get_player_metadata(player_id)
            if not player:
                return False, f"Player {player_id} not found"
            
            position_counts[player['position']] = position_counts.get(player['position'], 0) + 1
            team_counts[player['team_id']] = team_counts.get(player['team_id'], 0) + 1
        
        # Check position requirements
        required = {'GKP': 2, 'DEF': 5, 'MID': 5, 'FWD': 3}
        for pos, required_count in required.items():
            if position_counts.get(pos, 0) != required_count:
                return False, f"Need {required_count} {pos}, have {position_counts.get(pos, 0)}"
        
        # Check max 3 per team
        for team_id, count in team_counts.items():
            if count > 3:
                team = self.teams_by_id.get(team_id, {})
                team_name = team.get('short_name', f'Team {team_id}')
                return False, f"Max 3 players per team (have {count} from {team_name})"
        
        return True, "Team composition valid"


# Singleton instance
_fpl_service_instance = None

def get_fpl_service() -> FPLDataService:
    """Get singleton FPL data service instance"""
    global _fpl_service_instance
    if _fpl_service_instance is None:
        _fpl_service_instance = FPLDataService()
    return _fpl_service_instance
