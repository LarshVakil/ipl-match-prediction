import pandas as pd
import numpy as np
import warnings

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

DELIVERIES_FILE = 'data/deliveries_updated_ipl_upto_2025.csv'
IPL_FILE = 'data/ipl.csv'
SQUADS_FILE = 'data/ipl_2026_full_squads.csv'

RECENT_MATCH_WINDOW = 8

def load_csv(path, name):
    df = pd.read_csv(path)
    print(f"{name} loaded -> {df.shape}")
    return df

def calculate_match_winners(deliveries):
    deliveries["total_runs"] = deliveries["batsman_runs"] + deliveries["extras"]

    scores = (
        deliveries
        .groupby(["matchId", "inning", "batting_team"])["total_runs"]
        .sum()
        .reset_index()
    )

    i1 = scores[scores["inning"] == 1][["matchId", "batting_team", "total_runs"]]
    i2 = scores[scores["inning"] == 2][["matchId", "batting_team", "total_runs"]]

    i1.columns = ["matchId", "team1", "runs1"]
    i2.columns = ["matchId", "team2", "runs2"]

    results = i1.merge(i2, on="matchId")

    def winner(row):
        if row["runs1"] > row["runs2"]:
            return row["team1"]
        elif row["runs2"] > row["runs1"]:
            return row["team2"]
        return "Tie"

    results["winner"] = results.apply(winner, axis=1)
    results = results[results["winner"] != "Tie"]

    print(f"Usable matches -> {len(results)}")
    return results

def calculate_head_to_head(match_results):
    h2h = {}

    for _, r in match_results.iterrows():
        t1, t2, w = r["team1"], r["team2"], r["winner"]
        key = tuple(sorted([t1, t2]))

        if key not in h2h:
            h2h[key] = {t1: 0, t2: 0}

        h2h[key][w] += 1

    return h2h


def get_h2h_score(team_a, team_b, h2h):
    key = tuple(sorted([team_a, team_b]))
    if key not in h2h:
        return 0.0

    a = h2h[key].get(team_a, 0)
    b = h2h[key].get(team_b, 0)
    total = a + b

    return (a - b) / total if total > 0 else 0.0

def calculate_recent_form(match_results):
    history = {}

    for _, r in match_results.iterrows():
        for t in [r["team1"], r["team2"]]:
            history.setdefault(t, [])

        history[r["winner"]].append(1)
        loser = r["team1"] if r["winner"] == r["team2"] else r["team2"]
        history[loser].append(0)

    recent_form = {}
    for team, results in history.items():
        recent = results[-RECENT_MATCH_WINDOW:]
        recent_form[team] = sum(recent) / len(recent) if recent else 0.5

    return recent_form

def calculate_venue_toss_bias(matches):
    venue_bias = {}

    for _, r in matches.iterrows():
        venue = r["venue"]
        winner = r["winner"]
        toss_winner = r["toss_winner"]

        venue_bias.setdefault(venue, {"favored": 0, "total": 0})
        venue_bias[venue]["total"] += 1

        if toss_winner == winner:
            venue_bias[venue]["favored"] += 1

    for v in venue_bias:
        t = venue_bias[v]["total"]
        venue_bias[v]["bias"] = venue_bias[v]["favored"] / t - 0.5 if t > 0 else 0

    return venue_bias

def calculate_team_power_scores(squads):
    team_power = {}
    for team in squads["team"].unique():
        team_power[team] = np.random.uniform(0.45, 0.55)
    return team_power

def build_features(match_results, team_power, h2h, form, matches, venue_bias):
    X, y = [], []
    meta = matches.set_index("id")

    for _, r in match_results.iterrows():
        mid = r["matchId"]
        if mid not in meta.index:
            continue

        m = meta.loc[mid]
        t1, t2 = r["team1"], r["team2"]

        features = [
            team_power.get(t1, 0.5) - team_power.get(t2, 0.5),
            get_h2h_score(t1, t2, h2h),
            form.get(t1, 0.5) - form.get(t2, 0.5),
            venue_bias.get(m["venue"], {}).get("bias", 0),
            1 if m["toss_winner"] == t1 else -1
        ]

        X.append(features)
        y.append(1 if r["winner"] == t1 else 0)

    return np.array(X), np.array(y)

def train_model(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.2, random_state=42
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(Xtr, ytr)

    preds = model.predict(Xte)
    test_accuracy = accuracy_score(yte, preds)

    print("\n==============================")
    print(f"TEST SET ACCURACY: {test_accuracy:.4f}")
    print("==============================\n")

    return model, scaler, test_accuracy

def predict_match(team_a, team_b, venue, toss_winner,
                  model, scaler, team_power, h2h, form, venue_bias):

    features = np.array([[
        team_power.get(team_a, 0.5) - team_power.get(team_b, 0.5),
        get_h2h_score(team_a, team_b, h2h),
        form.get(team_a, 0.5) - form.get(team_b, 0.5),
        venue_bias.get(venue, {}).get("bias", 0),
        1 if toss_winner == team_a else -1
    ]])

    probs = model.predict_proba(scaler.transform(features))[0]

    return {
        "team_a": team_a,
        "team_b": team_b,
        "venue": venue,
        "toss_winner": toss_winner,
        "predicted_winner": team_a if probs[1] > probs[0] else team_b,
        "team_a_win_prob": round(float(probs[1]), 3),
        "team_b_win_prob": round(float(probs[0]), 3)
    }


if __name__ == "__main__":
    print("IPL 2026 Match Prediction Model\n")

    deliveries = load_csv(DELIVERIES_FILE, "Deliveries")
    matches = load_csv(MATCHES_FILE, "Matches")
    squads = load_csv(SQUADS_FILE, "Squads")

    match_results = calculate_match_winners(deliveries)

    h2h = calculate_head_to_head(match_results)
    form = calculate_recent_form(match_results)
    venue_bias = calculate_venue_toss_bias(matches)
    team_power = calculate_team_power_scores(squads)

    X, y = build_features(match_results, team_power, h2h, form, matches, venue_bias)
    model, scaler, test_acc = train_model(X, y)

    print(f"Final stored test accuracy -> {test_acc:.4f}\n")

    teams = list(team_power.keys())
    example = predict_match(
        team_a=teams[0],
        team_b=teams[1],
        venue=matches.iloc[0]["venue"],
        toss_winner=teams[0],
        model=model,
        scaler=scaler,
        team_power=team_power,
        h2h=h2h,
        form=form,
        venue_bias=venue_bias
    )

    print("Example prediction:")
    print(example)
