# -----------------------------
# JOUR 5 Evaluation du modèle
# -----------------------------

# 1. importation des bibliothèques
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv(
    "C:/Users/ocura/Documents/machine_learning_pratique/SALARY_PREDICTION-ML/data/Salary_Data.csv"
)

# 3. Séparation X/y
X = df[["YearsExperience"]]
y = df["Salary"]

# 4. division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. création et entrainement du modèle
modele = LinearRegression()
modele.fit(X_train, y_train)

# 6. prediction sur l'ensemble de test
y_pred = modele.predict(X_test)

# 7. calcul des métriques d'évaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n Evaluation du modèle : ")
print("R² : ", r2)
print("Root Mean Square Error : ", rmse)

# 8. Affichages des valeurs réelles vs prédites
resultats = pd.DataFrame(
    {
        "Experience": X_test["YearsExperience"],
        "Salaire_reel": y_test,
        "Salaire_prevu": y_pred,
    }
)

print("\nApperçu test vs prédictions : ")
print(resultats.head())
