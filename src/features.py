# src/features.py
import pandas as pd
import numpy as np
from typing import Tuple

def load_processed(path: str = "data/processed/player_gw_stats.csv") -> pd.DataFrame:
    """Load your merged GW-level stats into a DataFrame."""
    return pd.read_csv(path)


def make_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Given a df with columns ['player_id','gameweek','total_points',
    'minutes','goals','assists','opponent_team','was_home'], this will:

      1) Merge in fixture_difficulty
      2) Merge in & one-hot encode position
      3) Sort and compute rolling features
      4) Shift target to next GW
    """
    # Ensure was_home is boolean
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(bool)
        # Create home_away column for compatibility
        df["home_away"] = df["was_home"].apply(lambda x: "H" if x else "A")
    else:
        # If was_home is missing, create a default (you might want to handle this differently)
        print("Warning: was_home column not found, creating default values")
        df["was_home"] = True  # Default to home
        df["home_away"] = "H"

    # ─── 1) Fixture difficulty ─────────────────────────────────────────────
    import os
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    fix_path = os.path.join(pkg_root, "data", "raw", "fixtures.csv")
    print("→ loading fixtures from:", fix_path)
    
    try:
        raw_fix = pd.read_csv(fix_path)
        raw_fix.columns = raw_fix.columns.str.strip()
        # index by event and keep only the two difficulty columns
        fix = raw_fix.set_index("event")[["team_h_difficulty","team_a_difficulty"]]

        # join those difficulties onto each player-GW by the 'round' key
        df = df.join(fix, on="gameweek")

        # compute fixture_difficulty directly from was_home (bool)
        df["fixture_difficulty"] = np.where(
            df["was_home"],
            df["team_h_difficulty"],
            df["team_a_difficulty"],
        )

        # clean up helper cols
        df = df.drop(columns=["team_h_difficulty","team_a_difficulty"])
        print("→ fixture difficulty added successfully")
    except Exception as e:
        print(f"→ Warning: Could not load fixture difficulty: {e}")
        # Create a default fixture difficulty if we can't load it
        df["fixture_difficulty"] = 3  # Default medium difficulty

    # ─── 2) Player position one-hot ────────────────────────────────────────
    try:
        elems = pd.read_csv("data/raw/bootstrap_elements.csv")[["id","element_type"]]
        types = pd.read_csv("data/raw/bootstrap_element_types.csv")[["id","singular_name_short"]]
        types = types.rename(columns={"id":"element_type","singular_name_short":"position"})
        elems = elems.merge(types, on="element_type")
        elems = elems.rename(columns={"id":"player_id"})
        df = df.merge(elems[["player_id","position"]], on="player_id", how="left")

        # one-hot encode the four positions
        df = pd.concat([df, pd.get_dummies(df["position"], prefix="pos")], axis=1)
        df = df.drop(columns=["element_type","position"])
        print("→ position encoding added successfully")
    except Exception as e:
        print(f"→ Warning: Could not load position data: {e}")
        # Create default position columns if we can't load them
        for pos in ["pos_FWD", "pos_MID", "pos_DEF", "pos_GKP"]:
            df[pos] = 0

    # ─── 3) Team encoding ──────────────────────────────────────────────────
    if "opponent_team" in df.columns:
        # Convert team names to numeric codes
        team_encoder = {team: idx for idx, team in enumerate(df["opponent_team"].unique())}
        df["opponent_team_encoded"] = df["opponent_team"].map(team_encoder)
        df = df.drop(columns=["opponent_team"])
        print(f"→ encoded {len(team_encoder)} teams")

    # ─── 4) Home/Away encoding ─────────────────────────────────────────────
    if "home_away" in df.columns:
        # Convert H/A to numeric (H=1, A=0)
        df["home_away_encoded"] = (df["home_away"] == "H").astype(int)
        df = df.drop(columns=["home_away"])
        print("→ encoded home/away")

    # ─── 5) Rolling/window features ────────────────────────────────────────
    df = df.sort_values(["player_id","gameweek"])

    # last-window mean of points
    df[f"points_roll{window}"] = (
        df.groupby("player_id")["total_points"]
          .shift(1)
          .rolling(window, min_periods=1)
          .mean()
    )
    # last-window sum of minutes
    df[f"minutes_roll{window}"] = (
        df.groupby("player_id")["minutes"]
          .shift(1)
          .rolling(window, min_periods=1)
          .sum()
    )

    # ─── new advanced‐metrics rolls ─────────────────────────────────
    for col in ["xG","xA","key_passes","npxG"]:
        if col in df.columns:
            df[f"{col}_roll{window}"] = (
                df.groupby("player_id")[col]
                  .shift(1)
                  .rolling(window, min_periods=1)
                  .mean()
            )

    # ─── 6) Next-GW target ─────────────────────────────────────────────────
    df["target"] = df.groupby("player_id")["total_points"].shift(-1)
    df = df.dropna(subset=["target"])  # remove end-of-season rows

    # ─── 7) Handle missing values ───────────────────────────────────────────
    # Fill NaN values with appropriate defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "target":  # Don't fill the target
            if col in ["fixture_difficulty", "opponent_team_encoded"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
    
    print("→ handled missing values")

    return df


def get_X_y(feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split into feature matrix X and target vector y.
    Drops identifiers and the raw total_points column.
    """
    X = feature_df.drop(
        columns=[
            "player_id",
            "gameweek",
            "total_points",
            "target"
        ]
    )
    y = feature_df["target"]
    return X, y
