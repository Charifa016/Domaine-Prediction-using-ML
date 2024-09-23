import pickle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Charger les données d'entraînement et de test
with open("data/processed/train_data.pkl", 'rb') as f:
    X_train, y_train = pickle.load(f)

with open("data/processed/test_data.pkl", 'rb') as f:
    X_test, y_test = pickle.load(f)

# Entraîner le modèle SVM
svm = SVC()
svm.fit(X_train, y_train)

# Évaluer le modèle
y_pred = svm.predict(X_test)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Sauvegarder le modèle
with open("models/svm_model.pkl", 'wb') as f:
    pickle.dump(svm, f)

# Sauvegarder les résultats
with open("results/model_comparison_report.txt", 'a') as f:
    f.write("SVM Model:\n")
    f.write(report + "\n")

import matplotlib.pyplot as plt
import seaborn as sns

# Sauvegarder la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Matrice de confusion SVM")
plt.savefig("results/confusion_matrix_svm.png")
plt.close()
