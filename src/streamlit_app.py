import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path
import requests
import os

st.set_page_config(
    page_title="LoL Draft Winrate Predictor",
    layout="wide",  # Use wide layout for more space
    initial_sidebar_state="collapsed" # Optionally collapse sidebar
)

# --- Configuration and Paths (must be accessible from app.py) ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path("/tmp/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
MODEL_COLUMNS_PATH = MODELS_DIR / "model_columns.joblib"

MODEL_URL = "https://drive.google.com/uc?export=download&id=1_YVZWwh3bcOFoK0JgABANPFEbIrP26QH"
MODEL_COLUMNS_URL = "https://drive.google.com/uc?export=download&id=1fzx6J6FeK5KOpk_0OxT9FduF6A3FdsoR"

def download_model():
    if not MODEL_PATH.exists():
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
    if not MODEL_COLUMNS_PATH.exists():
        r = requests.get(MODEL_COLUMNS_URL)
        with open(MODEL_COLUMNS_PATH, "wb") as f:
            f.write(r.content)

download_model()

class WinRatePredictor:
    def __init__(self, model_path, columns_path):
        self.model = None
        self.model_columns = None
        try:
            self.model = joblib.load(model_path)
            self.model_columns = joblib.load(columns_path)
        except FileNotFoundError:
            st.error(f"Error: Model or columns file not found. Ensure that {model_path} and {columns_path} exist.")
        except Exception as e:
            st.error(f"Error loading the model: {e}")

    def predict(self, team1_picks, team2_picks):
        if self.model is None or self.model_columns is None:
            return {"error": "Model not loaded."}

        if len(team1_picks) != 5 or len(team2_picks) != 5:
            return {"error": "Each team must have exactly 5 champions."}

        input_df = pd.DataFrame(0, index=[0], columns=self.model_columns)

        for champion in team1_picks:
            col_name = f"teamA_{champion}"
            if col_name in input_df.columns:
                input_df.loc[0, col_name] = 1
            else:
                st.warning(f"Champion '{champion}' for Blue Side unknown to the model, ignored.")

        for champion in team2_picks:
            col_name = f"teamB_{champion}"
            if col_name in input_df.columns:
                input_df.loc[0, col_name] = 1
            else:
                st.warning(f"Champion '{champion}' for Red Side unknown to the model, ignored.")

        probabilities = self.model.predict_proba(input_df)[0]

        return {
            'team1_win_probability': probabilities[1],
            'team2_win_probability': probabilities[0]
        }

# --- Initialize the predictor ---
predictor = WinRatePredictor(MODEL_PATH, MODEL_COLUMNS_PATH)

# --- Get the list of all champions known by the model ---
all_champions = []
if predictor.model_columns is not None:
    champions_t1 = [col.replace("teamA_", "") for col in predictor.model_columns if col.startswith("teamA_")]
    champions_t2 = [col.replace("teamB_", "") for col in predictor.model_columns if col.startswith("teamB_")]
    all_champions = sorted(list(set(champions_t1 + champions_t2)))
    if not all_champions:
        st.error("No champions could be extracted from the model columns. Please check the model_columns.joblib file.")

# --- Streamlit Interface ---

# Conditional CSS injection
def inject_progress_bar_css(color_hex):
    st.markdown(
        f"""
        <style>
        .stProgress > div > div > div > div {{
            background-color: {color_hex};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <style>
    .main-header {
        font-size: 3.5em;
        font-weight: bold;
        text-align: center;
        color: #F7E6B8; /* A light gold/yellow for a game-like feel */
        text-shadow: 2px 2px 4px #000000;
        margin-bottom: 0.5em;
    }
    .subheader {
        font-size: 1.8em;
        font-weight: bold;
        color: #ADD8E6; /* Light blue for team names */
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    .stSelectbox>div>div>div>span {
        color: #D3D3D3; /* Lighter text for selectbox options */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green for the predict button */
        color: white;
        font-size: 1.2em;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
    }
    .stMetric > div > div > div {
        font-size: 2em;
        font-weight: bold;
        color: #F7E6B8; /* Light gold for metric values */
    }
    .stMetric > div > label {
        font-size: 1.2em;
        color: #D3D3D3; /* Lighter text for metric labels */
    }
    /* Removed .stProgress styling from here, will inject dynamically */
    .stMarkdown p {
        font-size: 1.1em;
        color: #A9A9A9; /* Darker grey for general text */
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-radius: 5px;
        padding: 10px;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 5px;
        padding: 10px;
    }
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='main-header'>LoL Draft Winrate Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>Predict the win chances of the professional League of Legends team compositions!</h3>", unsafe_allow_html=True)

st.markdown("---")

if not predictor.model or not all_champions:
    st.info("Please train the model (`src/train.py`) to generate the necessary files.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2 class='subheader'>🔵 Blue Side Composition</h2>", unsafe_allow_html=True)
        team1_picks = []
        for i in range(5):
            pick = st.selectbox(f"Pick {i+1}", all_champions, key=f"t1_pick_{i}")
            team1_picks.append(pick)

    with col2:
        st.markdown("<h2 class='subheader'>🔴 Red Side Composition</h2>", unsafe_allow_html=True)
        team2_picks = []
        for i in range(5):
            pick = st.selectbox(f"Pick {i+1}", all_champions, key=f"t2_pick_{i}")
            team2_picks.append(pick)

    st.markdown("---")

    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("Predict Winrate"):
        st.markdown("</div>", unsafe_allow_html=True) # Close the div after the button
        # Check for duplicates within each team
        if len(set(team1_picks)) != 5:
            st.error("Blue Side contains duplicate champions. Each champion must be unique.")
        elif len(set(team2_picks)) != 5:
            st.error("Red Side contains duplicate champions. Each champion must be unique.")
        elif len(set(team1_picks).intersection(set(team2_picks))) > 0:
            st.error("The same champion cannot be in both teams (mirror pick).")
        else:
            with st.spinner("Calculating prediction..."):
                prediction = predictor.predict(team1_picks, team2_picks)

                if "error" in prediction:
                    st.error(prediction["error"])
                else:
                    team1_winrate = prediction['team1_win_probability'] * 100
                    team2_winrate = prediction['team2_win_probability'] * 100

                    st.markdown("---")
                    st.markdown("<h2 class='subheader'>🎯 Prediction Results</h2>", unsafe_allow_html=True)
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric(label="Blue Side Winrate", value=f"{team1_winrate:.2f}%")
                    with res_col2:
                        st.metric(label="Red Side Winrate", value=f"{team2_winrate:.2f}%")
                    
                    # Dynamically set progress bar color
                    if team1_winrate > team2_winrate:
                        inject_progress_bar_css("#4682B4") # SteelBlue for Blue Side
                        st.progress(team1_winrate / 100) 
                        st.balloons() # Add a little celebration!
                    elif team2_winrate > team1_winrate:
                        inject_progress_bar_css("#B22222") # FireBrick for Red Side
                        st.progress(team2_winrate / 100) # Show progress for Red Side's winrate
                        st.balloons() # Add a little celebration!
                    else:
                        inject_progress_bar_css("#808080") # Grey for balanced
                        st.progress(0.5) # Show a neutral 50% for balanced
                        st.info("The match is very balanced. May the best team win!")

st.markdown("---")
st.markdown("Developed with love for the LoL community by Clément D'ELBREIL.")