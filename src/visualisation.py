# ------------------------------------
# JOUR 6 - Visualisation des résultats
# ------------------------------------

# 1. Import des bibliothèques
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# 2. Chargement du dataset
df = pd.read_csv(
    "C:/Users/ocura/Documents/machine_learning_pratique/SALARY_PREDICTION-ML/data/Salary_Data.csv"
)

# 3. Séparation des variables
X = df[["YearsExperience"]]
y = df["Salary"]

# 4. Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Entraînement du modèle
modele = LinearRegression()
modele.fit(X_train, y_train)

# 6. Prédictions
y_pred = modele.predict(X_test)

# -----------------------------
# 1ère visualisation :
# Droite de régression + points d'entraînement
# -----------------------------
plt.figure(figsize=(8, 6))

# nuage des points d'entraînement
plt.scatter(X_train, y_train, color="blue", label="Données d'entraînement")

# droite de régression
plt.plot(X_train, modele.predict(X_train), color="red", label="Droite de régression")

plt.title("Régression linéaire : expérience vs salaire")
plt.xlabel("Années d'expérience")
plt.ylabel("Salaire")
plt.legend()
plt.show()

# -----------------------------
# 2ème visualisation :
# Vraies valeurs vs prédictions (ensemble test)
# -----------------------------
plt.figure(figsize=(8, 6))

plt.scatter(X_test, y_test, color="green", label="Valeurs réelles")
plt.scatter(X_test, y_pred, color="red", label="Prédictions")

plt.title("Comparaison valeurs réelles vs prédictions (données test)")
plt.xlabel("Années d'expérience")
plt.ylabel("Salaire")
plt.legend()
plt.show()
