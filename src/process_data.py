# src/process_data.py
import os, glob, json
import pandas as pd

# where your raw data is
RAW = "data/raw"
OUT  = "data/processed/player_gw_stats.csv"

# 1) load bootstrap element info (static per player)
elems = pd.read_csv(os.path.join(RAW, "bootstrap_elements.csv"))

# 2) load each element‐summary JSON (per‐player history)
histories = []
for jfile in glob.glob(os.path.join(RAW, "summary_*.json")):
    rec = json.load(open(jfile))
    dfh = pd.DataFrame(rec["history"])      # history is a list of GW‐rows
    dfh["player_id"] = rec["id"]
    histories.append(dfh)
hist_df = pd.concat(histories, ignore_index=True)

# 3) merge them
#    here we keep only the columns your model’s feature‐code expects
merged = (
    hist_df
    .rename(columns={"round":"gameweek"})
    [["player_id","gameweek","total_points","minutes","goals_scored","assists"]]
)

# 4) save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
merged.to_csv(OUT, index=False)
print(f"Wrote {OUT} ({len(merged)} rows)")
