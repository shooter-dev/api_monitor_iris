"""
Script pour générer les données de référence Evidently
À exécuter UNE SEULE FOIS pour créer data/reference_data.csv
"""
import pandas as pd

# Mapping des espèces vers des valeurs numériques
SPECIES_MAP = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

# Lire le CSV d'origine
df = pd.read_csv('data/Iris.csv')

# Renommer les colonnes pour correspondre à l'API
df = df.rename(columns={
    'SepalLengthCm': 'sepal_length',
    'SepalWidthCm': 'sepal_width',
    'PetalLengthCm': 'petal_length',
    'PetalWidthCm': 'petal_width',
    'Species': 'prediction_name'
})

# Ajouter la colonne prediction (numérique)
df['prediction'] = df['prediction_name'].map(SPECIES_MAP)

# Sélectionner les colonnes dans le bon ordre
df = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 
         'prediction', 'prediction_name']]

# Sauvegarder
df.to_csv('data/reference_data.csv', index=False)

print(f"✅ Fichier de référence créé : data/reference_data.csv")
print(f"📊 Nombre de lignes : {len(df)}")
print(f"\n📋 Aperçu :")
print(df.head())