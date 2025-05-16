import pandas as pd
import numpy as np
from tqdm import tqdm

# Initialisation de tqdm pour les apply()
tqdm.pandas(desc="Échantillonnage alternatives")

print("=== Script lancé ===")

# 1. Charger les fichiers CSV
global_file = pd.read_csv("C:/Users/lenny/OneDrive/Documents/Stage M1/global.csv")
alt_file = pd.read_csv("C:/Users/lenny/OneDrive/Documents/Stage M1/alternatives.csv")

# 2. Nettoyer les colonnes
global_file = global_file.drop(columns=['ID'], errors='ignore')
global_file = global_file.rename(columns={'unit': 'id'})

# 3. Ajouter un identifiant d'observation unique
global_file = global_file.reset_index().rename(columns={'index': 'obs'})

# 4. Fonction pour échantillonner 8 alternatives pondérées par individu
def sample_alternatives(row, n=8):
    sampled = alt_file.sample(
        n=n,
        weights=alt_file['proba_sampled'],
        replace=False,
        random_state=row['obs']  # reproductibilité
    ).copy()
    sampled['obs'] = row['obs']
    return sampled

# 5. Appliquer l’échantillonnage avec une barre de progression
sampled_list = global_file.progress_apply(sample_alternatives, axis=1)
sampled_df = pd.concat(sampled_list.tolist(), ignore_index=True)

# 6. Marquer les alternatives choisies (choix = 1)
choix_df = global_file.merge(alt_file, on='commune_destination', how='left')
choix_df['choix'] = 1
choix_df = choix_df.rename(columns={'unit': 'unit_1'})

# 7. Préparation des colonnes
sampled_df = sampled_df.rename(columns={'unit': 'unit_sampled'})
sampled_df = sampled_df[['obs', 'unit_sampled'] + [col for col in sampled_df.columns if col not in ['obs', 'unit_sampled']]]

# 8. Joindre avec le fichier des vrais choix
merged_df = sampled_df.merge(choix_df[['obs', 'id', 'unit_1']], on='obs', how='left')

# 9. Supprimer les alternatives réellement choisies
altnon_choix = merged_df[merged_df['unit_sampled'] != merged_df['unit_1']].copy()
altnon_choix['choix'] = 0

# 10. Réduction à 7 alternatives non choisies avec barre de progression
tqdm.pandas(desc="Sélection de 7 alternatives")
altnon_choix = (
    altnon_choix
    .groupby('obs')
    .progress_apply(lambda x: x.sample(n=7, random_state=x['obs'].iloc[0]))
    .reset_index(drop=True)
)

# 11. Ajouter les vrais choix
choix_final = choix_df.copy()
choix_final['unit_sampled'] = choix_final['unit_1']
choix_final = choix_final[['obs', 'id', 'unit_sampled', 'choix']]

# 12. Concaténer tous les choix
final_df = pd.concat([altnon_choix[['obs', 'id', 'unit_sampled', 'choix']], choix_final], ignore_index=True)

# 13. Ajouter identifiant d’alternative
final_df = final_df.sort_values(by=['obs', 'choix'], ascending=[True, False])
final_df['altern'] = final_df.groupby('obs').cumcount() + 1

# 14. Transformer en format wide
choix_df = final_df[final_df['choix'] == 1].copy()
non_choix_df = final_df[final_df['choix'] == 0].copy()
choix_df = choix_df[['obs', 'id', 'unit_sampled']].rename(columns={'unit_sampled': 'choix'})
non_choix_df['alt_num'] = non_choix_df.groupby('obs').cumcount() + 1
non_choix_wide = non_choix_df.pivot(index='obs', columns='alt_num', values='unit_sampled')
non_choix_wide.columns = [f'alt{i}' for i in non_choix_wide.columns]
final_wide = choix_df.merge(non_choix_wide, on='obs')
cols = ['obs', 'id', 'choix'] + [f'alt{i}' for i in range(1, 8)]
final_wide = final_wide[cols]

# 15. Affichage et sauvegarde
print("=== Résultat (5 premières lignes) ===")
print(final_wide.head())

final_wide.to_csv("C:/Users/lenny/OneDrive/Documents/Stage M1/final_format_wide.csv", index=False)

print("=== Sauvegarde terminée. Fichier 'final_format_wide.csv' créé ===")
