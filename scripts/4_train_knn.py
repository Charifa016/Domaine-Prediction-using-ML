import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Charger les données d'entraînement et de test
with open("data/processed/train_data.pkl", 'rb') as f:
    X_train, y_train = pickle.load(f)

with open("data/processed/test_data.pkl", 'rb') as f:
    X_test, y_test = pickle.load(f)

# Charger le vectorizer
with open("models/vectorizer.pkl", 'rb') as f:
    vectorizer = pickle.load(f)

# Entraîner le modèle KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Évaluer le modèle
y_pred = knn.predict(X_test)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Sauvegarder le modèle
with open("models/knn_model.pkl", 'wb') as f:
    pickle.dump(knn, f)

# Sauvegarder les résultats
with open("results/model_comparison_report.txt", 'a') as f:
    f.write("KNN Model:\n")
    f.write(report + "\n")

import matplotlib.pyplot as plt
import seaborn as sns

# Sauvegarder la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion KNN")
plt.savefig("results/confusion_matrix_knn.png")
plt.close()
