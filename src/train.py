# src/train.py (avec RandomizedSearchCV pour le réglage des hyperparamètres)

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import config
from scipy.stats import randint, uniform # Pour les distributions aléatoires des hyperparamètres

def transform_to_composition_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme le DataFrame en format symétrique pour éviter le biais de position.
    """
    print("🔄 Transformation des données au format symétrique...")
    
    pick_cols = [f'team{t}_pick{i}' for t in [1, 2] for i in range(1, 6)]
    all_champions = pd.unique(df[pick_cols].values.ravel('K'))
    all_champions = [c for c in all_champions if pd.notna(c)]
    print(f"🌍 Nombre total de champions uniques trouvés : {len(all_champions)}")

    # Créer les colonnes pour les deux équipes (A et B au lieu de team1/team2)
    feature_column_names = [f"teamA_{c}" for c in all_champions] + \
                           [f"teamB_{c}" for c in all_champions]
    
    # Créer le DataFrame final qui contiendra les données originales + les données inversées
    all_rows = []
    
    for index, row in df.iterrows():
        team1_picks = [row[f'team1_pick{i}'] for i in range(1, 6)]
        team2_picks = [row[f'team2_pick{i}'] for i in range(1, 6)]
        
        # Version originale : team1 -> teamA, team2 -> teamB
        row_original = pd.Series(0, index=feature_column_names)
        for champion in team1_picks:
            if pd.notna(champion):
                row_original[f"teamA_{champion}"] = 1
        for champion in team2_picks:
            if pd.notna(champion):
                row_original[f"teamB_{champion}"] = 1
        row_original[config.TARGET_COLUMN] = row[config.TARGET_COLUMN]
        all_rows.append(row_original)
        
        # Version inversée : team2 -> teamA, team1 -> teamB
        row_inverted = pd.Series(0, index=feature_column_names)
        for champion in team2_picks:
            if pd.notna(champion):
                row_inverted[f"teamA_{champion}"] = 1
        for champion in team1_picks:
            if pd.notna(champion):
                row_inverted[f"teamB_{champion}"] = 1
        # Inverser le label : si team1 gagnait (1), maintenant team2 est en teamA donc 0
        row_inverted[config.TARGET_COLUMN] = 1 - row[config.TARGET_COLUMN]
        all_rows.append(row_inverted)
    
    df_encoded = pd.DataFrame(all_rows)
    print(f"📈 Dataset augmenté : {len(df)} -> {len(df_encoded)} exemples")
    
    return df_encoded

def train_model():
    """
    Fonction principale pour entraîner et sauvegarder le modèle.
    """
    print("🚀 Démarrage de l'entraînement...")

    try:
        df_raw = pd.read_csv(config.CLEAN_DATA_PATH)
        print(f"✅ Données brutes chargées avec succès : {len(df_raw)} lignes.")
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier {config.CLEAN_DATA_PATH} n'a pas été trouvé.")
        print("Veuillez d'abord exécuter le script de prétraitement des données (`data_preprocessing.py`).")
        return

    df_transformed = transform_to_composition_format(df_raw)
    
    X = df_transformed.drop(columns=[config.TARGET_COLUMN])
    y = df_transformed[config.TARGET_COLUMN]

    # Sauvegarder les noms des colonnes pour la prédiction future
    # Cela doit être fait AVANT de diviser les données pour la recherche d'hyperparamètres
    # car les colonnes doivent correspondre exactement pour la prédiction.
    joblib.dump(X.columns, config.MODELS_DIR / 'model_columns.joblib')
    print(f"🔢 Le nombre de features est maintenant de : {X.shape[1]}")
    print(f"👀 Aperçu des colonnes d'entraînement (premières 5) : {list(X.columns[:5])}")


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Données divisées : {len(X_train)} pour l'entraînement, {len(X_test)} pour le test.")

    # --- Configuration de la recherche aléatoire des hyperparamètres ---
    print("\n⚙️ Démarrage de la recherche aléatoire des hyperparamètres pour RandomForestClassifier...")

    # Définir le modèle de base pour la recherche
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Définir les distributions des hyperparamètres à échantillonner
    # Tu peux ajuster ces plages en fonction de tes observations initiales ou de la complexité attendue
    param_distributions = {
        'n_estimators': randint(100, 500),         # Nombre d'arbres entre 100 et 499
        'max_depth': [None, 10, 20, 30, 40, 50], # Profondeur maximale (None pour illimitée)
        'min_samples_split': randint(2, 20),       # Minimum d'échantillons pour une division (entre 2 et 19)
        'min_samples_leaf': randint(1, 10),        # Minimum d'échantillons par feuille (entre 1 et 9)
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9, 1.0] # Stratégies pour le nombre de features
    }

    # Initialiser RandomizedSearchCV
    # n_iter: nombre de combinaisons d'hyperparamètres à échantillonner
    # cv: nombre de plis pour la validation croisée
    # scoring: métrique d'évaluation (accuracy est un bon début pour la classification équilibrée)
    # verbose: niveau de détail des logs
    random_search = RandomizedSearchCV(estimator=rf_base,
                                       param_distributions=param_distributions,
                                       n_iter=5, # Choisir un nombre d'itérations raisonnable (par exemple, 50 à 100)
                                       cv=5,       # 5 plis de validation croisée
                                       n_jobs=-1,  # Utiliser tous les cœurs disponibles
                                       verbose=2,  # Afficher les progrès
                                       scoring='accuracy', # ou 'f1', 'roc_auc' si déséquilibré
                                       random_state=42)

    # Lancer la recherche
    random_search.fit(X_train, y_train)

    print("\n✅ Recherche aléatoire des hyperparamètres terminée.")
    print(f"Meilleurs hyperparamètres trouvés : {random_search.best_params_}")
    print(f"Meilleure précision en validation croisée : {random_search.best_score_:.4f}")

    # Utiliser le meilleur modèle trouvé par la recherche
    model = random_search.best_estimator_
    # --- Fin de la configuration de la recherche aléatoire ---

    # 5. Évaluer le modèle (le meilleur modèle trouvé par la recherche)
    print("\n📈 Évaluation du modèle final (avec les meilleurs hyperparamètres)...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nPrécision (Accuracy) sur l'ensemble de test : {accuracy:.4f}")
    print("\nRapport de classification :\n", classification_report(y_test, y_pred))
    print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred))

    # 6. Sauvegarder le modèle (le meilleur modèle)
    print(f"\n💾 Sauvegarde du modèle entraîné dans : {config.MODEL_PATH}")
    joblib.dump(model, config.MODEL_PATH)

    print("\n🎉 Processus d'entraînement terminé avec succès !")

if __name__ == "__main__":
    train_model()