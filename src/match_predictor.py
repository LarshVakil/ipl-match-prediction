import warnings
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELIVERIES_FILE = os.path.join(BASE_DIR, 'data', 'deliveries_updated_ipl_upto_2025.csv')
MATCHES_FILE = os.path.join(BASE_DIR, 'data', 'ipl.csv')
SQUADS_FILE = os.path.join(BASE_DIR, 'data', 'ipl_2026_full_squads.csv')

RECENT_MATCH_WINDOW = 8

# --- TEAM NAME STANDARDIZATION ---
# This fixes the "Bangalore vs Bengaluru" and "Delhi Daredevils vs Capitals" mismatches
NAME_MAP = {
    'Royal Challengers Bangalore': 'RCB',
    'Royal Challengers Bengaluru': 'RCB',
    'Kings XI Punjab': 'PBKS',
    'Punjab Kings': 'PBKS',
    'Delhi Daredevils': 'DC',
    'Delhi Capitals': 'DC',
    'Sunrisers Hyderabad': 'SRH',
    'Mumbai Indians': 'MI',
    'Kolkata Knight Riders': 'KKR',
    'Rajasthan Royals': 'RR',
    'Chennai Super Kings': 'CSK',
    'Gujarat Titans': 'GT',
    'Lucknow Super Giants': 'LSG',
    'Rising Pune Supergiant': 'RPS',
    'Rising Pune Supergiants': 'RPS',
    'Pune Warriors': 'PWI',
    'Deccan Chargers': 'DEC',
    'Kochi Tuskers Kerala': 'KTK',
    'Gujarat Lions': 'GL'
}

def clean_team_name(name):
    """Standardizes team names to simple acronyms to prevent mismatch errors."""
    return NAME_MAP.get(str(name).strip(), str(name).strip())

def load_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    print(f"{name} loaded -> {df.shape}")
    return df

def calculate_match_winners(deliveries):
    deliveries.columns = deliveries.columns.str.strip()
    
    # Clean Team Names Immediately
    deliveries["batting_team"] = deliveries["batting_team"].apply(clean_team_name)
    
    deliveries["total_runs"] = deliveries["batsman_runs"] + deliveries["extras"]
    
    # Group to get match totals
    scores = deliveries.groupby(["matchId", "inning", "batting_team"])["total_runs"].sum().reset_index()
    
    i1 = scores[scores["inning"] == 1][["matchId", "batting_team", "total_runs"]]
    i2 = scores[scores["inning"] == 2][["matchId", "batting_team", "total_runs"]]
    
    i1.columns = ["matchId", "team1", "runs1"]
    i2.columns = ["matchId", "team2", "runs2"]
    
    results = i1.merge(i2, on="matchId")
    results["winner"] = results.apply(lambda r: r["team1"] if r["runs1"] > r["runs2"] else r["team2"], axis=1)
    
    # Create STRING key (safest for Pandas)
    results["match_lookup"] = results.apply(lambda r: f"{min(r['team1'], r['team2'])}_vs_{max(r['team1'], r['team2'])}", axis=1)
    
    print(f"Processed {len(results)} matches from deliveries.")
    return results

def calculate_head_to_head(match_results):
    h2h = {}
    for _, r in match_results.iterrows():
        t1, t2, w = r["team1"], r["team2"], r["winner"]
        key = tuple(sorted([t1, t2]))
        if key not in h2h: h2h[key] = {t1: 0, t2: 0}
        h2h[key][w] = h2h[key].get(w, 0) + 1
    return h2h

def get_h2h_score(t1, t2, h2h_dict):
    key = tuple(sorted([t1, t2]))
    if key not in h2h_dict: return 0.0
    a, b = h2h_dict[key].get(t1, 0), h2h_dict[key].get(t2, 0)
    return (a - b) / (a + b) if (a + b) > 0 else 0.0

def calculate_recent_form(match_results):
    history = {}
    for _, r in match_results.iterrows():
        for t in [r["team1"], r["team2"]]: history.setdefault(t, [])
        history[r["winner"]].append(1)
        loser = r["team1"] if r["winner"] == r["team2"] else r["team2"]
        history[loser].append(0)
    return {t: (sum(res[-RECENT_MATCH_WINDOW:]) / len(res[-RECENT_MATCH_WINDOW:]) if res else 0.5) for t, res in history.items()}

def calculate_venue_toss_bias(matches):
    venue_bias = {}
    for _, r in matches.iterrows():
        v = r.get("venue", "Unknown")
        w = clean_team_name(r.get("winner"))
        tw = clean_team_name(r.get("toss_winner"))
        
        venue_bias.setdefault(v, {"favored": 0, "total": 0})
        venue_bias[v]["total"] += 1
        if tw == w: venue_bias[v]["favored"] += 1
    return {v: (d["favored"]/d["total"] - 0.5 if d["total"] > 0 else 0) for v, d in venue_bias.items()}

def build_features(match_results, team_power, h2h, form, matches, venue_bias):
    X, y = [], []
    
    # Prepare matches metadata with STRING keys and Clean Names
    matches['team1_clean'] = matches['team1'].apply(clean_team_name)
    matches['team2_clean'] = matches['team2'].apply(clean_team_name)
    
    matches['match_lookup'] = matches.apply(
        lambda r: f"{min(r['team1_clean'], r['team2_clean'])}_vs_{max(r['team1_clean'], r['team2_clean'])}", 
        axis=1
    )
    
    # Drop duplicates to keep the latest info for that matchup
    meta_lookup = matches.drop_duplicates(subset=['match_lookup'], keep='last').set_index('match_lookup')

    print(f"Metadata ready with {len(meta_lookup)} unique matchups.")

    for _, r in match_results.iterrows():
        lookup_key = r["match_lookup"]
        
        # Safe lookup: Check if key exists
        if lookup_key not in meta_lookup.index:
            continue

        m = meta_lookup.loc[lookup_key]
        if isinstance(m, pd.DataFrame): m = m.iloc[0]
        
        t1, t2 = r["team1"], r["team2"]
        v_name = m.get("venue", "Unknown")

        # Robust feature extraction
        features = [
            team_power.get(t1, 0.5) - team_power.get(t2, 0.5), 
            get_h2h_score(t1, t2, h2h),                       
            form.get(t1, 0.5) - form.get(t2, 0.5),             
            venue_bias.get(v_name, 0),                        
            1 if clean_team_name(m.get("toss_winner")) == t1 else -1 
        ]
        
        X.append(features)
        y.append(1 if r["winner"] == t1 else 0)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    print("IPL 2026 Match Prediction Model - Final Fix\n")

    # 1. Load Data
    deliveries = load_csv(DELIVERIES_FILE, "Deliveries")
    matches_df = load_csv(MATCHES_FILE, "Matches")
    squads = load_csv(SQUADS_FILE, "Squads")

    # 2. Process
    results = calculate_match_winners(deliveries)
    h2h_data = calculate_head_to_head(results)
    recent_form = calculate_recent_form(results)
    v_bias = calculate_venue_toss_bias(matches_df)
    
    # Power Scores (Clean names first)
    pwr = {clean_team_name(team): np.random.uniform(0.45, 0.55) for team in squads["team"].unique()}

    # 3. Build & Train
    X, y = build_features(results, pwr, h2h_data, recent_form, matches_df, v_bias)
    
    if len(X) < 10:
        print(f"Error: Only {len(X)} matches found. Check 'NAME_MAP' in script for missing team names.")
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = XGBClassifier(n_estimators=200, learning_rate=0.05, eval_metric="logloss")
        model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, model.predict(X_test))
        print("\n" + "="*40)
        print(f"SUCCESS: Model trained on {len(X)} matches.")
        print(f"ACCURACY: {acc:.4f}")
        print("="*40 + "\n")
        
        # Example
        t1, t2 = "MI", "CSK" # Using new short codes
        test_feat = np.array([[
            pwr.get(t1, 0.5) - pwr.get(t2, 0.5),
            get_h2h_score(t1, t2, h2h_data),
            recent_form.get(t1, 0.5) - recent_form.get(t2, 0.5),
            0, 
            1 
        ]])
        prob = model.predict_proba(scaler.transform(test_feat))[0]
        print(f"Prediction for {t1} vs {t2}: {t1} Win Prob: {prob[1]:.2%}")