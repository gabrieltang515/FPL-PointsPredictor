# src/historical_ingest.py

import os
import pandas as pd

# 1. Where your raw historical CSV lives (from your project root)
RAW_CSV = os.path.join("data", "raw", "historical", "fpl_training.csv")

# 2. Where we want to write the processed per‐gameweek file
OUT_CSV = os.path.join("data", "processed", "player_gw_stats.csv")

def backfill_historical():
    # Load the raw CSV
    df = pd.read_csv(RAW_CSV)

    # Rename the columns so they match our pipeline's schema
    df = df.rename(columns={
        "element":       "player_id",     # player ID
        "round":         "gameweek",      # gameweek number
        "total_points":  "total_points",  # FPL points that week
        "minutes":       "minutes",       # minutes played
        "goals_scored":  "goals",         # goals
        "assists":       "assists",       # assists
        "was_home":      "was_home",      # home/away indicator
        "opponent_team": "opponent_team", # opponent team
        "xG":            "xG",            # expected goals
        "xA":            "xA",            # expected assists
        "key_passes":    "key_passes",    # key passes
        "npxG":          "npxG",          # non-penalty expected goals
        "element_type":  "element_type"   # player position type
    })

    # Select the columns we need, including the new ones
    df = df[[
        "player_id",
        "gameweek", 
        "total_points",
        "minutes",
        "goals",
        "assists",
        "was_home",
        "opponent_team",
        "xG",
        "xA", 
        "key_passes",
        "npxG",
        "element_type"
    ]]

    # Handle missing values in advanced metrics
    for col in ["xG", "xA", "key_passes", "npxG"]:
        df[col] = df[col].fillna(0)

    # Ensure the output folder exists
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    # Write out the backfilled per‐GW stats
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df):,} rows to {OUT_CSV}")
    print(f"Columns: {df.columns.tolist()}")

if __name__ == "__main__":
    backfill_historical()
