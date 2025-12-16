# -----------------------------
# JOUR 3 - Préparation des données
# -----------------------------

# 1. Import des bibliothèques
import pandas as pd
from sklearn.model_selection import train_test_split

# 2. Chargement du dataset (même fichier que Jour 1)
df = pd.read_csv(
    "C:/Users/ocura/Documents/machine_learning_pratique/SALARY_PREDICTION-ML/data/Salary_Data.csv"
)

# 3. Séparation des variables (X et y)
# X = colonnes d'entrée
# y = colonne cible (ce qu'on veut prédire)
X = df[["YearsExperience"]]  # double crochets pour garder un tableau 2D
y = df["Salary"]

print("Aperçu de X :")
print(X.head())

print("\nAperçu de y :")
print(y.head())

# 4. Division du dataset en train et test
# train_test_split sépare de façon aléatoire les données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Vérification des tailles
print("\nTailles des ensembles :")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)
