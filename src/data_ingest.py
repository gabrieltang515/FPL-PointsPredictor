import os
import requests
import pandas as pd

# URLs for FPL API endpoints
BOOTSTRAP_STATIC_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def fetch_json(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def save_df_to_csv(df, filename):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved {filename} to {filepath}")

def main():
    # Fetch bootstrap-static
    print("Fetching bootstrap-static...")
    bootstrap_data = fetch_json(BOOTSTRAP_STATIC_URL)

    # Flatten and save each top-level key that is a list
    for key, value in bootstrap_data.items():
        if isinstance(value, list):
            df = pd.json_normalize(value)
            save_df_to_csv(df, f"bootstrap_{key}.csv")

    # Fetch fixtures
    print("Fetching fixtures...")
    fixtures_data = fetch_json(FIXTURES_URL)
    fixtures_df = pd.json_normalize(fixtures_data)
    save_df_to_csv(fixtures_df, "fixtures.csv")

# AI: “Merge player element-summary JSON into a single DataFrame of (player_id, gameweek, total_points, minutes, goals, assists),
# then join fixture info (opponent, home/away) and save to data/processed/player_gw_stats.csv”

import time
import json

def fetch_player_history(player_id):
    url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def build_player_gw_stats():

    # Load player list from previously saved bootstrap_elements.csv
    elements_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'bootstrap_elements.csv')
    players_df = pd.read_csv(elements_path)
    player_ids = players_df['id'].tolist()
    print(f"▶ Will fetch histories for {len(player_ids)} players")
    print(f"▶ Sample player IDs: {player_ids[:5]}")

    HISTORY_DIR = os.path.join(RAW_DATA_DIR, 'history')
    os.makedirs(HISTORY_DIR, exist_ok=True)

    all_histories = []
    for pid in player_ids:
        try:
            data = fetch_player_history(pid)

            # Combine this season’s history (often empty) with past seasons’
            raw_hist   = data.get('history', [])
            past_hist  = data.get('history_past', [])
            full_hist  = raw_hist + past_hist

            # if pid <= 3:
            #     print(f"DEBUG: keys in JSON for player {pid}:", data.keys())

            # ——— NEW: dump the raw JSON for caching ———
            with open(os.path.join(HISTORY_DIR, f"summary_{pid}.json"), 'w') as fp:
                json.dump(data, fp)
            print(f"Saved raw JSON for player {pid}")
            history = data.get('history', [])
            print(f"  • player {pid} → {len(history)} history rows")

            for entry in history:
                all_histories.append({
                    'player_id': pid,
                    'gameweek': entry.get('round'),
                    'total_points': entry.get('total_points'),
                    'minutes': entry.get('minutes'),
                    'goals': entry.get('goals_scored'),
                    'assists': entry.get('assists'),
                    'opponent_team': entry.get('opponent_team'),
                    'was_home': entry.get('was_home')
                })
            time.sleep(0.5)  # Be polite to the API
        except Exception as e:
            print(f"Error fetching player {pid}: {e}")
            continue
    
    print(f"▶ Total history entries collected: {len(all_histories)}")
    stats_df = pd.DataFrame(all_histories)

    # Map opponent_team id to team name using bootstrap_teams.csv
    teams_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'bootstrap_teams.csv')
    teams_df = pd.read_csv(teams_path)[['id', 'name']]
    teams_df = teams_df.rename(columns={'id': 'opponent_team', 'name': 'opponent_name'})

    print("▶ stats_df columns:", stats_df.columns.tolist())
    print("▶ teams_df columns:", teams_df.columns.tolist())


    stats_df = stats_df.merge(teams_df, on='opponent_team', how='left')

    # Convert was_home to string
    stats_df['home_away'] = stats_df['was_home'].map({True: 'H', False: 'A', 1: 'H', 0: 'A'})
    stats_df = stats_df.drop(columns=['was_home'])

    # Save to processed
    processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'player_gw_stats.csv')
    stats_df.to_csv(out_path, index=False)
    print(f"Saved merged player gameweek stats to {out_path}")





if __name__ == "__main__":
    main()
    build_player_gw_stats()

