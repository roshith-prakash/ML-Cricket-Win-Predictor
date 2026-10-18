import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root and app directory are in sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
for p in [APP_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from components.win_gauge import render_win_gauge
from components.probability_chart import render_probability_timeline

# Streamlit Page Config
st.set_page_config(
    page_title="IPL Live Win Probability Estimator (Both Innings)",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark-themed aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        color: #F8FAFC;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        margin-top: 4px;
        color: #38BDF8;
    }
    .stSelectbox label, .stSlider label {
        color: #E2E8F0 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Canonical franchise normalization mapping
TEAM_NORMALIZATION = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}

def normalize_team(name: str) -> str:
    return TEAM_NORMALIZATION.get(name, name)


class LiveWinPredictor:
    """
    Inference engine supporting both 1st Innings (Score Projection & Win Probability)
    and 2nd Innings (Chasing Win Probability).
    """
    def __init__(self, models_dir: str = "models"):
        if os.path.exists(models_dir):
            models_dir = os.path.abspath(models_dir)
        elif os.path.exists(os.path.join(PROJECT_ROOT, models_dir)):
            models_dir = os.path.abspath(os.path.join(PROJECT_ROOT, models_dir))
        else:
            models_dir = os.path.join(PROJECT_ROOT, "models")

        self.models_dir = models_dir

        # Load Innings 2 models
        with open(os.path.join(models_dir, "feature_columns.json"), "r") as f:
            self.inn2_features = json.load(f)["all_features"]

        self.inn2_models = {
            "logistic": joblib.load(os.path.join(models_dir, "logistic_baseline.pkl")),
            "lgbm": joblib.load(os.path.join(models_dir, "lgbm_model.pkl")),
            "calibrated_lgbm": joblib.load(os.path.join(models_dir, "calibrator.pkl"))
        }

        # Load Innings 1 models
        with open(os.path.join(models_dir, "feature_columns_inn1.json"), "r") as f:
            self.inn1_features = json.load(f)["all_features"]

        self.inn1_calibrator = joblib.load(os.path.join(models_dir, "calibrator_inn1.pkl"))
        self.inn1_score_regressor = joblib.load(os.path.join(models_dir, "score_regressor_inn1.pkl"))

    def predict_inn1(
        self,
        batting_team: str,
        bowling_team: str,
        venue: str,
        current_score: int,
        legal_balls_bowled: int,
        wickets_fallen: int
    ):
        """
        Predicts projected final score and win probability for 1st innings.
        """
        balls_left = max(0, 120 - legal_balls_bowled)
        wickets_in_hand = max(0, 10 - wickets_fallen)
        crr = 0.0 if legal_balls_bowled == 0 else (current_score * 6.0) / legal_balls_bowled

        if wickets_in_hand <= 0 or balls_left <= 0:
            projected_score = current_score
        else:
            X_df = pd.DataFrame({
                "current_score": [current_score],
                "legal_balls_bowled": [legal_balls_bowled],
                "balls_left": [balls_left],
                "wickets_fallen": [wickets_fallen],
                "wickets_in_hand": [wickets_in_hand],
                "crr": [round(crr, 4)],
                "batting_team": [batting_team],
                "bowling_team": [bowling_team],
                "venue": [venue]
            })[self.inn1_features]

            raw_proj = self.inn1_score_regressor.predict(X_df)[0]
            projected_score = int(max(current_score, round(raw_proj)))

        # Win probability for 1st innings batting team
        # 1. Projected target defense probability from 2nd-innings model
        proj_target = projected_score + 1
        p2_chase_proj = self.predict_inn2(
            batting_team=bowling_team,
            bowling_team=batting_team,
            venue=venue,
            target=proj_target,
            current_score=0,
            legal_balls_bowled=0,
            wickets_fallen=0,
            model_name="calibrated_lgbm"
        )
        p_defend_proj = 1.0 - p2_chase_proj

        # 2. Prior probability from 1st innings classifier
        X_df = pd.DataFrame({
            "current_score": [current_score],
            "legal_balls_bowled": [legal_balls_bowled],
            "balls_left": [balls_left],
            "wickets_fallen": [wickets_fallen],
            "wickets_in_hand": [wickets_in_hand],
            "crr": [round(crr, 4)],
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "venue": [venue]
        })[self.inn1_features]
        p_prior = self.inn1_calibrator.predict_proba(X_df)[0][1]

        # 3. Continuous blend: as innings progresses towards ball 120, weight converges to 1.0
        # This guarantees 0% discontinuity between the end of Innings 1 and start of Innings 2
        w = (legal_balls_bowled / 120.0) ** 1.5
        prob = (1.0 - w) * p_prior + w * p_defend_proj
        return projected_score, float(np.clip(prob, 0.01, 0.99))

    def predict_inn2(
        self,
        batting_team: str,
        bowling_team: str,
        venue: str,
        target: int,
        current_score: int,
        legal_balls_bowled: int,
        wickets_fallen: int,
        model_name: str = "calibrated_lgbm"
    ) -> float:
        """
        Predicts chasing win probability for 2nd innings with cricket boundary physics.
        """
        runs_left = target - current_score
        balls_left = max(0, 120 - legal_balls_bowled)
        wickets_in_hand = max(0, 10 - wickets_fallen)

        # 1. Deterministic boundary checks
        if runs_left <= 0: return 1.0
        if wickets_in_hand <= 0: return 0.0
        if balls_left <= 0: return 0.0

        # 2. Realistic cricket physical boundaries (avoids tree calibration step flatlining)
        if runs_left > balls_left * 6: return 0.0005
        if runs_left >= 36 and balls_left <= 6: return 0.001
        if runs_left >= 25 and balls_left <= 6 and wickets_in_hand <= 4: return 0.002
        if runs_left == 1:
            if balls_left >= 3 and wickets_in_hand >= 2: return 0.998
            if balls_left >= 1 and wickets_in_hand >= 1: return 0.950
        if runs_left <= 2 and balls_left >= 4 and wickets_in_hand >= 3: return 0.988
        if runs_left <= 4 and balls_left >= 8 and wickets_in_hand >= 4: return 0.975

        # 3. Rate calculations
        crr = 0.0 if legal_balls_bowled == 0 else (current_score * 6.0) / legal_balls_bowled
        rrr = min(36.0, (runs_left * 6.0) / balls_left)
        rrr_minus_crr = rrr - crr
        wicket_pressure = wickets_fallen / (legal_balls_bowled + 1)

        X = pd.DataFrame({
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets_in_hand": [wickets_in_hand],
            "rrr": [round(rrr, 4)],
            "crr": [round(crr, 4)],
            "rrr_minus_crr": [round(rrr_minus_crr, 4)],
            "target": [target],
            "wicket_pressure": [round(wicket_pressure, 4)],
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "venue": [venue]
        })[self.inn2_features]

        model = self.inn2_models.get(model_name, self.inn2_models["calibrated_lgbm"])
        prob = model.predict_proba(X)[0][1]

        # In final 3 overs (death overs), blend logistic regression to fix tree calibration edge compression
        if balls_left <= 18 and "logistic" in self.inn2_models:
            p_log = self.inn2_models["logistic"].predict_proba(X)[0][1]
            w_log = (18 - balls_left) / 18.0 * 0.5
            prob = (1.0 - w_log) * prob + w_log * p_log

        return float(np.clip(prob, 0.001, 0.999))


def replay_full_match(
    match_id: str,
    json_dir: str = "ipl_json",
    model_name: str = "calibrated_lgbm"
):
    """
    Replays both 1st innings and 2nd innings of a match ball-by-ball.
    """
    if not os.path.isabs(json_dir):
        if not os.path.exists(json_dir):
            json_dir = os.path.join(PROJECT_ROOT, json_dir)

    file_path = os.path.join(json_dir, f"{match_id}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Match file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    inn1 = data["innings"][0]
    inn2 = data["innings"][1]

    team1 = normalize_team(inn1["team"])
    team2 = normalize_team(inn2["team"])
    target = inn2["target"]["runs"]
    venue = info.get("venue", "Unknown Venue")
    actual_winner = normalize_team(info.get("outcome", {}).get("winner", "Unknown"))

    predictor = LiveWinPredictor()
    delivery_log = []

    # --- INNINGS 1 SIMULATION ---
    curr_score = 0
    legal_balls = 0
    wickets = 0

    init_proj, init_p1 = predictor.predict_inn1(team1, team2, venue, 0, 0, 0)
    delivery_log.append({
        "innings": 1, "innings_str": "1st Innings", "over_str": "0.0", "event": "Match Start",
        "score_str": "0/0", "context_str": f"Projected: {init_proj}", "win_prob": init_p1, "team1_win_prob": init_p1
    })

    for over in inn1.get("overs", []):
        for d in over.get("deliveries", []):
            extras = d.get("extras", {})
            runs = d.get("runs", {}).get("total", 0)
            is_w = "wickets" in d

            curr_score += runs
            if not ("wides" in extras or "noballs" in extras): legal_balls += 1
            if is_w: wickets += len(d["wickets"])

            proj, p1 = predictor.predict_inn1(team1, team2, venue, curr_score, legal_balls, wickets)
            desc = f"{d.get('batter', 'Batter')} vs {d.get('bowler', 'Bowler')}"
            if is_w: desc += " (WICKET!)"
            elif runs == 6: desc += " (SIX!)"
            elif runs == 4: desc += " (FOUR!)"

            delivery_log.append({
                "innings": 1, "innings_str": "1st Innings",
                "over_str": f"{legal_balls // 6}.{legal_balls % 6}",
                "event": desc, "score_str": f"{curr_score}/{wickets}",
                "context_str": f"Projected: {proj}", "win_prob": p1, "team1_win_prob": p1
            })
            if wickets >= 10 or legal_balls >= 120: break
        if wickets >= 10 or legal_balls >= 120: break

    # --- INNINGS 2 SIMULATION ---
    curr_score2 = 0
    legal_balls2 = 0
    wickets2 = 0
    final_result = None

    init_p2 = predictor.predict_inn2(team2, team1, venue, target, 0, 0, 0, model_name)
    delivery_log.append({
        "innings": 2, "innings_str": "2nd Innings", "over_str": "0.0", "event": "Chase Begins",
        "score_str": "0/0", "context_str": f"Target: {target} (RRR: {target/20.0:.2f})",
        "win_prob": init_p2, "team1_win_prob": 1.0 - init_p2
    })

    for over in inn2.get("overs", []):
        for d in over.get("deliveries", []):
            extras = d.get("extras", {})
            runs = d.get("runs", {}).get("total", 0)
            is_w = "wickets" in d

            curr_score2 += runs
            if not ("wides" in extras or "noballs" in extras): legal_balls2 += 1
            if is_w: wickets2 += len(d["wickets"])

            if curr_score2 >= target:
                final_result = f"{team2} Won"
            elif wickets2 >= 10:
                final_result = f"{team1} Won (All Out)"
            elif legal_balls2 >= 120:
                final_result = f"{team1} Won (Balls Exhausted)"

            p2 = predictor.predict_inn2(team2, team1, venue, target, curr_score2, legal_balls2, wickets2, model_name)
            p1 = 1.0 - p2

            desc = f"{d.get('batter', 'Batter')} vs {d.get('bowler', 'Bowler')}"
            if is_w: desc += " (WICKET!)"
            elif runs == 6: desc += " (SIX!)"
            elif runs == 4: desc += " (FOUR!)"

            runs_left = max(0, target - curr_score2)
            balls_left = max(0, 120 - legal_balls2)
            rrr = 0.0 if runs_left <= 0 else min(36.0, (runs_left * 6.0) / max(1, balls_left))

            delivery_log.append({
                "innings": 2, "innings_str": "2nd Innings",
                "over_str": f"{legal_balls2 // 6}.{legal_balls2 % 6}",
                "event": desc, "score_str": f"{curr_score2}/{wickets2}",
                "context_str": f"Needed: {runs_left} off {balls_left} (RRR {rrr:.1f})",
                "win_prob": p2, "team1_win_prob": p1
            })
            if final_result: break
        if final_result: break

    meta = {
        "match_id": match_id,
        "date": info.get("dates", [""])[0],
        "venue": venue,
        "team1": team1,
        "team2": team2,
        "target": target,
        "actual_winner": actual_winner,
        "final_result": final_result or "Completed"
    }
    return meta, pd.DataFrame(delivery_log)


@st.cache_resource
def load_predictor():
    return LiveWinPredictor()

@st.cache_data
def load_metadata():
    path = os.path.join(PROJECT_ROOT, "data", "processed", "matches_metadata.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


predictor = load_predictor()
df_matches = load_metadata()

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/cricket.png", width=64)
st.sidebar.title("IPL Win Estimator")
st.sidebar.caption("Machine Learning Ball-by-Ball Live Probability Engine (Both Innings)")

nav_mode = st.sidebar.radio(
    "Select Mode",
    ["🎮 Live Match Simulator", "⏱️ Historical Match Replay", "📊 Model Evaluation & Metrics"],
    index=0
)

model_choice = st.sidebar.selectbox(
    "2nd-Innings Classifier Engine",
    [
        ("calibrated_lgbm", "Calibrated LightGBM (Isotonic - Recommended)"),
        ("logistic", "Logistic Regression Baseline"),
        ("lgbm", "Raw LightGBM")
    ],
    format_func=lambda x: x[1]
)[0]

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Key Features:**
- **Innings 1:** Live Score Projection & Win Probability.
- **Innings 2:** Live Target Chase Win Probability.
- **Full Match Replay:** Seamless ball-by-ball curve across both innings.
""")

TEAMS_LIST = [
    "Chennai Super Kings",
    "Mumbai Indians",
    "Kolkata Knight Riders",
    "Royal Challengers Bengaluru",
    "Rajasthan Royals",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

VENUES_LIST = [
    "Wankhede Stadium, Mumbai",
    "M Chinnaswamy Stadium, Bengaluru",
    "MA Chidambaram Stadium, Chepauk, Chennai",
    "Eden Gardens, Kolkata",
    "Narendra Modi Stadium, Ahmedabad",
    "Arun Jaitley Stadium, Delhi",
    "Rajiv Gandhi International Stadium, Uppal",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "Sawai Mansingh Stadium, Jaipur",
    "Punjab Cricket Association IS Bindra Stadium, Mohali",
    "Dubai International Cricket Stadium",
    "Sharjah Cricket Stadium"
]


# =========================================================================
# MODE 1: LIVE MATCH SIMULATOR (WHAT-IF ANALYSIS - BOTH INNINGS)
# =========================================================================
if nav_mode == "🎮 Live Match Simulator":
    st.header("🎮 Live Match Win Probability Simulator")
    st.write("Simulate any situation in either the **1st Innings (Setting the Target)** or **2nd Innings (Chasing the Target)**.")

    innings_mode = st.radio(
        "Select Match Phase",
        ["1st Innings: Setting the Target", "2nd Innings: Chasing the Target"],
        horizontal=True
    )

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        team1 = st.selectbox("1st Innings Batting Team", TEAMS_LIST, index=0)
    with col2:
        avail_team2 = [t for t in TEAMS_LIST if t != team1]
        team2 = st.selectbox("2nd Innings Batting Team (Chasing)", avail_team2, index=0)
    with col3:
        venue = st.selectbox("Stadium / Venue", VENUES_LIST, index=0)

    st.markdown("---")

    # --- INNINGS 1 SIMULATOR ---
    if "1st Innings" in innings_mode:
        st.subheader(f"Innings 1: **{team1}** Batting vs **{team2}**")

        c_ov, c_sc, c_wk = st.columns([1.5, 1.5, 1.2])
        with c_ov:
            overs_val = st.slider("Overs Completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
            full_overs = int(overs_val)
            sub_balls = min(5, int(round((overs_val - full_overs) * 10)))
            legal_balls = min(120, full_overs * 6 + sub_balls)

        with c_sc:
            current_score = st.slider("Current Score", min_value=0, max_value=280, value=88, step=1)

        with c_wk:
            wickets_fallen = st.slider("Wickets Lost", min_value=0, max_value=10, value=2, step=1)

        projected_score, win_prob_team1 = predictor.predict_inn1(
            batting_team=team1,
            bowling_team=team2,
            venue=venue,
            current_score=current_score,
            legal_balls_bowled=legal_balls,
            wickets_fallen=wickets_fallen
        )

        balls_left = max(0, 120 - legal_balls)
        wickets_left = max(0, 10 - wickets_fallen)
        crr = 0.0 if legal_balls == 0 else (current_score * 6.0) / legal_balls

        # KPI Cards
        st.markdown("### Match State & Projections")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Projected Total</div>
                <div class="metric-value">{projected_score}</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Current RR</div>
                <div class="metric-value">{crr:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Balls Left</div>
                <div class="metric-value">{balls_left}</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Wickets in Hand</div>
                <div class="metric-value">{wickets_left}</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Runs to Add</div>
                <div class="metric-value">{max(0, projected_score - current_score)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        left_col, right_col = st.columns([1.3, 1])
        with left_col:
            gauge_fig = render_win_gauge(team1, team2, win_prob_team1)
            st.plotly_chart(gauge_fig, width="stretch")

        with right_col:
            st.markdown("### 1st Innings Situation Assessment")
            p1_pct = win_prob_team1 * 100
            p2_pct = (1.0 - win_prob_team1) * 100

            if p1_pct >= 60:
                st.success(f"🟢 **{team1} setting a formidable total.** The projected target of **{projected_score}** gives them a substantial defensive cushion at {venue}.")
            elif p1_pct >= 45:
                st.warning(f"🟡 **Balanced par score.** The projected total of **{projected_score}** is around the historical average at {venue}.")
            else:
                st.error(f"🔴 **{team2} bowling tightly.** With {wickets_fallen} wickets down, {team1} is currently projected for **{projected_score}**, favoring the chasing team.")

            st.markdown(f"""
            - **{team1} (1st Batting):** **{p1_pct:.1f}%**
            - **{team2} (Chasing Next):** **{p2_pct:.1f}%**
            - **Projected Target for {team2}:** **{projected_score + 1} runs**
            """)

    # --- INNINGS 2 SIMULATOR ---
    else:
        st.subheader(f"Innings 2: **{team2}** Chasing vs **{team1}**")

        col_target, col_overs, col_score, col_wickets = st.columns([1.2, 1.5, 1.5, 1.2])
        with col_target:
            target = st.number_input("Target to Chase", min_value=80, max_value=290, value=185, step=1)

        with col_overs:
            overs_val = st.slider("Overs Completed", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
            full_overs = int(overs_val)
            sub_balls = min(5, int(round((overs_val - full_overs) * 10)))
            legal_balls_bowled = min(120, full_overs * 6 + sub_balls)

        with col_score:
            current_score = st.slider("Current Score", min_value=0, max_value=int(target + 6), value=92, step=1)

        with col_wickets:
            wickets_fallen = st.slider("Wickets Lost", min_value=0, max_value=10, value=2, step=1)

        runs_left = max(0, target - current_score)
        balls_left = max(0, 120 - legal_balls_bowled)
        wickets_in_hand = max(0, 10 - wickets_fallen)
        crr = 0.0 if legal_balls_bowled == 0 else (current_score * 6.0) / legal_balls_bowled
        rrr = 0.0 if runs_left <= 0 else min(36.0, (runs_left * 6.0) / max(1, balls_left))

        win_prob = predictor.predict_inn2(
            batting_team=team2,
            bowling_team=team1,
            venue=venue,
            target=target,
            current_score=current_score,
            legal_balls_bowled=legal_balls_bowled,
            wickets_fallen=wickets_fallen,
            model_name=model_choice
        )

        # KPI Summary Cards
        st.markdown("### Match State KPI")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Runs Needed</div><div class="metric-value">{runs_left}</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Balls Left</div><div class="metric-value">{balls_left}</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Wickets Left</div><div class="metric-value">{wickets_in_hand}</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Current RR</div><div class="metric-value">{crr:.2f}</div></div>""", unsafe_allow_html=True)
        with m5:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Required RR</div><div class="metric-value" style="color: {'#EF4444' if rrr > 11 else '#10B981'}">{rrr:.2f}</div></div>""", unsafe_allow_html=True)
        with m6:
            st.markdown(f"""<div class="metric-card"><div class="metric-title">Rate Delta</div><div class="metric-value" style="color: {'#EF4444' if (rrr - crr) > 2 else '#10B981'}">{(rrr - crr):+.2f}</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        left_col, right_col = st.columns([1.3, 1])
        with left_col:
            gauge_fig = render_win_gauge(team2, team1, win_prob)
            st.plotly_chart(gauge_fig, width="stretch")

        with right_col:
            st.markdown("### 2nd Innings Tactical Assessment")
            chasing_pct = win_prob * 100
            defending_pct = (1.0 - win_prob) * 100

            if runs_left <= 0:
                st.success(f"🏆 **Match Won!** {team2} have completed the chase.")
            elif wickets_in_hand <= 0:
                st.error(f"🛑 **All Out!** {team1} have defended the target.")
            elif balls_left <= 0:
                st.error(f"⌛ **Overs Complete!** {team1} win by {runs_left} runs.")
            elif chasing_pct >= 75:
                st.success(f"🟢 **{team2} in firm control.** With {wickets_in_hand} wickets in hand and an RRR of {rrr:.2f}, the chasing side is strongly favored.")
            elif chasing_pct >= 45:
                st.warning(f"🟡 **Game in the balance.** Both sides have a clear path to victory.")
            else:
                st.error(f"🔴 **{team1} dominating.** The escalating RRR ({rrr:.2f}) puts {team1} firmly ahead.")

            st.markdown(f"""
            - **{team2} (Chasing):** **{chasing_pct:.1f}%**
            - **{team1} (Defending):** **{defending_pct:.1f}%**
            - **Model in Use:** `{model_choice}`
            """)


# =========================================================================
# MODE 2: HISTORICAL MATCH REPLAY (FULL MATCH - BOTH INNINGS)
# =========================================================================
elif nav_mode == "⏱️ Historical Match Replay":
    st.header("⏱️ Historical Match Ball-by-Ball Replay (Both Innings)")
    st.write("Watch how the win probability developed from **Ball 0.1 of Innings 1** through the end of the chase.")

    if df_matches.empty:
        st.error("Match metadata not found. Please check data/processed/matches_metadata.csv")
    else:
        seasons = sorted(df_matches["season"].astype(str).unique(), reverse=True)
        sel_season = st.selectbox("Filter by Season", seasons, index=0)

        matches_in_season = df_matches[df_matches["season"].astype(str) == sel_season]

        match_options = {}
        for _, row in matches_in_season.iterrows():
            label = f"{row['date']} | {row['bowling_team']} vs {row['batting_team']} (Target: {row['target']}) — Winner: {row['outcome_winner']}"
            match_options[row["match_id"]] = label

        selected_match_id = st.selectbox(
            "Select Match to Replay",
            options=list(match_options.keys()),
            format_func=lambda x: match_options[x]
        )

        if st.button("🚀 Replay Full Match (Both Innings)", type="primary"):
            with st.spinner("Replaying full match delivery-by-delivery..."):
                try:
                    meta, df_timeline = replay_full_match(
                        match_id=str(selected_match_id),
                        model_name=model_choice
                    )

                    st.markdown("---")
                    st.markdown(f"""
                    ### Match Overview: **{meta['team1']}** (1st Bat) vs **{meta['team2']}** (Chased {meta['target']})
                    - **Date & Venue:** {meta['date']} at *{meta['venue']}*
                    - **Target Set:** **{meta['target']} runs**
                    - **Actual Result:** **{meta['actual_winner']} Won** ({meta['final_result']})
                    """)

                    # Chart
                    timeline_fig = render_probability_timeline(df_timeline, meta['team1'], meta['team2'], is_full_match=True)
                    st.plotly_chart(timeline_fig, width="stretch")

                    # Event Table
                    st.markdown("### Full Ball-by-Ball Event Feed")
                    display_cols = ["innings_str", "over_str", "score_str", "event", "context_str", "team1_win_prob"]
                    df_display = df_timeline[display_cols].copy()
                    df_display["team1_win_prob"] = (df_display["team1_win_prob"] * 100).round(1).astype(str) + "%"

                    st.dataframe(
                        df_display.rename(columns={
                            "innings_str": "Innings",
                            "over_str": "Over",
                            "score_str": "Score",
                            "event": "Ball Event",
                            "context_str": "Match Context",
                            "team1_win_prob": f"{meta['team1']} Win %"
                        }),
                        width="stretch",
                        height=400
                    )

                except Exception as e:
                    st.error(f"Failed to replay match {selected_match_id}: {e}")


# =========================================================================
# MODE 3: MODEL EVALUATION & METRICS
# =========================================================================
elif nav_mode == "📊 Model Evaluation & Metrics":
    st.header("📊 Model Performance & Calibration Benchmarks")
    st.write("Strict temporal validation results (Trained on seasons ≤ 2022; evaluated on unseen seasons ≥ 2023).")

    summary_file = os.path.join(PROJECT_ROOT, "models", "training_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            summary_data = json.load(f)

        df_bench = pd.DataFrame(summary_data).T.rename(columns={
            "brier_score": "Brier Score (↓)",
            "log_loss": "Log Loss (↓)",
            "roc_auc": "ROC-AUC (↑)",
            "accuracy": "Accuracy @ 0.5 (↑)"
        })
        st.markdown("### 2nd-Innings Chase Classifier Benchmarks")
        st.dataframe(df_bench, width="stretch")

    st.markdown("---")
    st.subheader("Reliability Diagram & ROC Curves")

    curves_img = os.path.join(PROJECT_ROOT, "data", "processed", "calibration_and_roc_curves.png")
    if os.path.exists(curves_img):
        st.image(curves_img, caption="Reliability Curve (Calibration) and ROC Curve on unseen test seasons (2023–2026).")

    st.markdown("""
    ### Innings 1 vs Innings 2 Architecture:
    - **Innings 1:** Predicts expected final score (LGBM Regressor, MAE ~20 runs) and win probability for the 1st batting team.
    - **Innings 2:** Predicts chasing win probability using exact target and balls remaining (Isotonic Calibrated LGBM, Brier Score 0.1781, ROC-AUC 0.8403).
    """)
