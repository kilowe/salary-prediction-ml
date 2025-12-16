# -----------------------------
# JOUR 1 - Chargement des données
# -----------------------------

# 1. Import des bibliothèques
import pandas as pd
import matplotlib.pyplot as plt

# 2. Lecture du fichier CSV
df = pd.read_csv(
    "C:/Users/ocura/Documents/machine_learning_pratique/SALARY_PREDICTION-ML/data/Salary_Data.csv"
)

# 3. Affichage des 5 premières lignes
print("Aperçu du dataset :")
print(df.head())

# 4. Informations générales sur le dataset
print("\nInformations générales :")
print(df.info())

# 5. Statistiques descriptives de base
print("\nStatistiques descriptives :")
print(df.describe())

# 6. Visualisation simple
plt.scatter(df["YearsExperience"], df["Salary"], color="blue")
plt.title("Relation entre l'expérience et le salaire")
plt.xlabel("Années d'expérience")
plt.ylabel("Salaire (USD)")
plt.show()
