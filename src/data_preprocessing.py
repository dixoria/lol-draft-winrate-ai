# data_preprocessing.py (script de formatage et de nettoyage)

import pandas as pd
from pathlib import Path # Pour une meilleure gestion des chemins
import sys

import config

# Ajout du chemin du projet au PYTHONPATH si nécessaire pour importer config
# Cela permet d'utiliser BASE_DIR de config.py pour les chemins des fichiers
# Pour l'exécution standalone de ce script, c'est mieux de définir les chemins directement
# Ou de s'assurer que src/config.py est importable si ce script fait partie du package.
# Pour la simplicité ici, je vais définir les chemins directement.

# Assumons que ce script est dans le répertoire 'data/' ou à la racine du projet
# Si ce script est dans 'data/', changez '..' en '.' ou ajustez selon l'emplacement réel.
# Si ce script est à la racine du projet, alors 'data/' et 'models/' sont des sous-dossiers.
# Pour le moment, je vais le mettre à la racine comme votre arborescence suggère qu'il pourrait être lancé de là,
# et config.py gère les chemins relatifs à BASE_DIR.

# Répertoire de base du projet (ajustez si votre script n'est pas à la racine)


def preprocess_lol_data():
    """
    Charge les données brutes, les nettoie et les restructure pour l'entraînement du modèle.
    Sauvegarde le dataset nettoyé dans le dossier data/.
    """
    print("🚀 Démarrage du prétraitement des données League of Legends...")

    # 1. Charger les données brutes
    try:
        # Colonnes intéressantes du dataset brut
        columns_to_keep = ["gameid", "participantid", "pick1", "pick2", "pick3", "pick4", "pick5", "result"]
        df = pd.read_csv(config.RAW_DATA_PATH, usecols=columns_to_keep)
        print(f"✅ Données brutes chargées depuis '{config.RAW_DATA_PATH}' : {len(df)} lignes.")
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier de données brutes '{config.RAW_DATA_PATH}' n'a pas été trouvé.")
        print("Assurez-vous que le chemin est correct et que le fichier existe.")
        sys.exit(1) # Quitter le script en cas d'erreur fatale

    # 2. Filtrer les participants (participantid 100 pour Équipe 1, 200 pour Équipe 2)
    df_filtered = df[df["participantid"].isin([100, 200])].copy() # Utiliser .copy() pour éviter SettingWithCopyWarning
    df_filtered = df_filtered.drop(columns=["participantid"])
    df_filtered = df_filtered.reset_index(drop=True)
    print(f"📊 Données filtrées pour les équipes 100 et 200 : {len(df_filtered)} lignes.")

    # 3. Restructurer le dataset pour avoir une ligne par match
    print("🔄 Restructuration du dataset (une ligne par match)...")
    df_restructured = restructure_dataset(df_filtered)
    print(f"✅ Dataset restructuré : {len(df_restructured)} matchs.")

    # 4. Nettoyage : Suppression des lignes avec des valeurs manquantes (champions non renseignés)
    df_clean = df_restructured.dropna()
    print(f"🧹 Suppression des lignes avec valeurs manquantes : {len(df_restructured) - len(df_clean)} lignes supprimées.")
    print(f"Total de lignes après nettoyage : {len(df_clean)}.")
    if len(df_restructured) > 0:
        print(f"Pourcentage de données conservées : {len(df_clean)/len(df_restructured)*100:.1f}%")

    # 5. Vérifications et affichage des statistiques
    print("\n--- Vérifications du dataset nettoyé ---")
    print("Valeurs manquantes après nettoyage :")
    print(df_clean.isnull().sum())

    all_champions = set()
    for col in ['team1_pick1', 'team1_pick2', 'team1_pick3', 'team1_pick4', 'team1_pick5',
               'team2_pick1', 'team2_pick2', 'team2_pick3', 'team2_pick4', 'team2_pick5']:
        all_champions.update(df_clean[col].unique())
    print(f"Nombre de champions uniques : {len(all_champions)}")
    print(f"Exemples de champions : {list(all_champions)[:10]}")

    print("Distribution des résultats ('team1_wins') :")
    print(df_clean['team1_wins'].value_counts())

    duplicates_in_picks = df_clean.apply(check_duplicate_champions, axis=1)
    if duplicates_in_picks.sum() > 0:
        print(f"⚠️ Attention : {duplicates_in_picks.sum()} lignes contiennent des champions dupliqués dans la même équipe.")
        print("Cela pourrait indiquer une anomalie dans les données brutes ou une logique à affiner si cela n'est pas attendu.")
    else:
        print("✅ Aucune composition d'équipe avec des champions dupliqués trouvée.")

    # 6. Sauvegarder le dataset nettoyé
    df_clean.to_csv(config.CLEAN_DATA_PATH, index=False)
    print(f"\n💾 Dataset nettoyé sauvegardé dans : '{config.CLEAN_DATA_PATH}'")
    print("\n🎉 Prétraitement des données terminé avec succès !")


def restructure_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Transforme le dataset pour avoir une ligne par match, en combinant les picks des deux équipes."""
    matches = []
    
    # Grouper par gameid
    for game_id in df['gameid'].unique():
        game_data = df[df['gameid'] == game_id]
        
        if len(game_data) == 2:  # Vérifier qu'on a bien 2 équipes (participantid 100 et 200)
            team1 = game_data.iloc[0] # Première ligne correspond à team 100 (ou 200, l'ordre n'est pas garanti par gameid)
            team2 = game_data.iloc[1] # Deuxième ligne

            # Si l'ordre n'est pas garanti par le tri initial, il faut s'assurer que team1_wins correspond bien à team1_picks
            # Assumons que la première ligne est Team 1 et la deuxième Team 2 pour la cohérence
            # avec la création de team1_wins ci-dessous.
            # Si 'participantid' était toujours 100 pour la 1ère ligne et 200 pour la 2ème après le filtre,
            # alors l'ordre est bon. Sinon, il faudrait trier ou vérifier 'participantid' ici.
            
            # Créer une ligne avec les deux compositions et le résultat
            match_row = {
                'team1_pick1': team1['pick1'],
                'team1_pick2': team1['pick2'],
                'team1_pick3': team1['pick3'],
                'team1_pick4': team1['pick4'],
                'team1_pick5': team1['pick5'],
                'team2_pick1': team2['pick1'],
                'team2_pick2': team2['pick2'],
                'team2_pick3': team2['pick3'],
                'team2_pick4': team2['pick4'],
                'team2_pick5': team2['pick5'],
                'team1_wins': team1['result']  # 1 si team1 gagne, 0 sinon (basé sur le 'result' de la première ligne)
            }
            matches.append(match_row)
    
    return pd.DataFrame(matches)

def check_duplicate_champions(row):
    """Vérifie si des champions sont dupliqués dans une même composition d'équipe."""
    team1_picks = [row[f'team1_pick{i}'] for i in range(1, 6)]
    team2_picks = [row[f'team2_pick{i}'] for i in range(1, 6)]
    # Retourne True si des doublons sont trouvés dans l'une ou l'autre équipe
    return len(set(team1_picks)) != 5 or len(set(team2_picks)) != 5

if __name__ == "__main__":
    preprocess_lol_data()