# src/predict.py (version corrigée)

import pandas as pd
import joblib
import config

class WinRatePredictor:
    def __init__(self, model_path, columns_path):
        """
        Initialise le prédicteur en chargeant le modèle et la liste des colonnes.
        """
        try:
            self.model = joblib.load(model_path)
            self.model_columns = joblib.load(columns_path)
            # print("✅ Modèle et liste des colonnes chargés avec succès.")
            # print(f"Le modèle a été entraîné sur {len(self.model_columns)} features.")
        except FileNotFoundError:
            print("❌ Erreur : Fichier de modèle ou de colonnes non trouvé.")
            print("Veuillez d'abord lancer le script d'entraînement `src/train.py` (version améliorée).")
            self.model = None
            self.model_columns = None # Important de le mettre à None aussi

    def predict(self, team1_picks, team2_picks):
        """
        Prédit le taux de victoire pour une composition donnée.

        Args:
            team1_picks (list): Liste de 5 champions pour l'équipe 1.
            team2_picks (list): Liste de 5 champions pour l'équipe 2.

        Returns:
            dict: Un dictionnaire contenant les probabilités de victoire.
        """
        # La vérification doit être faite sur self.model pour savoir s'il a été chargé.
        # Si self.model est None, alors self.model_columns le sera aussi grâce à l'init.
        if self.model is None: # C'est la manière correcte de vérifier si un objet est None
            return {"error": "Modèle non chargé. Veuillez vérifier les fichiers modèle et colonnes."}

        if len(team1_picks) != 5 or len(team2_picks) != 5:
            return {"error": "Chaque équipe doit avoir exactement 5 champions."}

        # 1. Créer un DataFrame avec la structure exacte du modèle (toutes les colonnes à 0)
        input_df = pd.DataFrame(0, index=[0], columns=self.model_columns)

        # 2. Remplir le DataFrame pour l'équipe 1
        for champion in team1_picks:
            col_name = f"team1_{champion}"
            if col_name in input_df.columns:
                input_df.loc[0, col_name] = 1
            else:
                # Ce champion n'était pas dans les données d'entraînement, on l'ignore.
                # Il est crucial de le signaler pour le débogage si un champion est mal orthographié ou nouveau.
                print(f"⚠️ Champion '{champion}' de l'équipe 1 inconnu du modèle (colonne '{col_name}' non trouvée), il sera ignoré.")

        # 3. Remplir le DataFrame pour l'équipe 2
        for champion in team2_picks:
            col_name = f"team2_{champion}"
            if col_name in input_df.columns:
                input_df.loc[0, col_name] = 1
            else:
                print(f"⚠️ Champion '{champion}' de l'équipe 2 inconnu du modèle (colonne '{col_name}' non trouvée), il sera ignoré.")

        # 4. Prédire les probabilités
       #  print("\nInput DataFrame pour la prédiction:")
        # print(input_df.head()) # Afficher les premières colonnes pour vérification
        # Assurez-vous que input_df ne contient que des 0 et des 1
        
        # Pour une validation plus rigoureuse (optionnel mais recommandé):
        # Vérifier que toutes les colonnes de input_df sont bien numériques et que leur ordre est correct.
        # if not pd.api.types.is_numeric_dtype(input_df.values):
        #     print("Erreur: Le DataFrame d'entrée contient des valeurs non numériques.")
        #     return {"error": "Erreur interne: DataFrame d'entrée non numérique."}

        probabilities = self.model.predict_proba(input_df)[0]

        result = {
            'team1_win_probability': probabilities[1], # Classe 1 = team1 gagne
            'team2_win_probability': probabilities[0]  # Classe 0 = team2 gagne
        }
        return result

if __name__ == '__main__':
    predictor = WinRatePredictor(config.MODEL_PATH, config.MODEL_COLUMNS_PATH)

    if predictor.model: # Cette vérification ici est la bonne!
        team1_composition = ['Xayah', 'Fiora', 'Quinn', 'Anivia', 'Sona']
        team2_composition = ['Orianna', 'Aphelios', 'Malzahar', 'Karthus', "Sylas"]

        print(f"\n🔮 Prédiction pour le match :")
        print(f"    🔵 Équipe 1: {', '.join(team1_composition)}")
        print(f"    🔴 Équipe 2: {', '.join(team2_composition)}")

        prediction = predictor.predict(team1_composition, team2_composition)

        if "error" not in prediction:
            team1_winrate = prediction['team1_win_probability'] * 100
            team2_winrate = prediction['team2_win_probability'] * 100
            
            print("\nRésultat de la prédiction :")
            print(f"    🔥 Taux de victoire pour l'Équipe 1 : {team1_winrate:.2f}%")
            print(f"    ❄️ Taux de victoire pour l'Équipe 2 : {team2_winrate:.2f}%")