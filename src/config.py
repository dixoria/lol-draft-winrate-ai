# src/config.py

from pathlib import Path

# Répertoire de base du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Répertoires de données et de modèles
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Créer les répertoires s'ils n'existent pas
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Chemins vers les fichiers
RAW_DATA_PATH = BASE_DIR / "data" / "2025_LoL_esports_match_data_from_OraclesElixir.csv"
CLEAN_DATA_PATH = DATA_DIR / "clean_dataset.csv"
MODEL_PATH = MODELS_DIR / "random_forest_model.joblib"
# --- NOUVEAU ---
# Chemin vers la liste des colonnes du modèle
MODEL_COLUMNS_PATH = MODELS_DIR / "model_columns.joblib"

# Colonne de la cible
TARGET_COLUMN = 'team1_wins'