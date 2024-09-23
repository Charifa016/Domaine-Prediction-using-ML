import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Charger les données d'entraînement et de test
with open("data/processed/train_data.pkl", 'rb') as f:
    X_train, y_train = pickle.load(f)

with open("data/processed/test_data.pkl", 'rb') as f:
    X_test, y_test = pickle.load(f)

# Entraîner le modèle Arbre de Décision
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Évaluer le modèle
y_pred = decision_tree.predict(X_test)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Sauvegarder le modèle
with open("models/decision_tree_model.pkl", 'wb') as f:
    pickle.dump(decision_tree, f)

# Sauvegarder les résultats
with open("results/model_comparison_report.txt", 'a') as f:
    f.write("Decision Tree Model:\n")
    f.write(report + "\n")

import matplotlib.pyplot as plt
import seaborn as sns

# Sauvegarder la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion Arbre de Décision")
plt.savefig("results/confusion_matrix_decision_tree.png")
plt.close()
