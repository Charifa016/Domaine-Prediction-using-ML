import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Charger les résultats
results = {}
with open("results/model_comparison_report.txt", 'r') as f:
    lines = f.readlines()

current_model = None
for line in lines:
    if "Model:" in line:
        current_model = line.split(" ")[0]
        results[current_model] = ""
    else:
        if current_model:
            results[current_model] += line

# Visualisation des précisions des modèles
model_names = list(results.keys())
accuracies = []

for model in model_names:
    report = results[model]
    accuracy_line = [line for line in report.split("\n") if "accuracy" in line.lower()]
    accuracy = float(accuracy_line[0].split()[-1]) if accuracy_line else 0
    accuracies.append(accuracy)

plt.figure(figsize=(10, 7))
sns.barplot(x=model_names, y=accuracies, palette="viridis")
plt.xlabel("Modèle")
plt.ylabel("Précision")
plt.title("Comparaison des Précisions des Modèles")
plt.savefig("visualizations/model_accuracy_comparison.png")
plt.close()

# Visualisation combinée des matrices de confusion
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Liste des noms des fichiers de matrices de confusion
confusion_matrix_files = [
    "results/confusion_matrix_decision_tree.png",
    "results/confusion_matrix_random_forest.png",
    "results/confusion_matrix_knn.png",
     "results/confusion_matrix_svm.png" # Ajoutez d'autres noms si nécessaires
]

for i, file_path in enumerate(confusion_matrix_files):
    if os.path.exists(file_path):
        cm = plt.imread(file_path)
        ax = axes.flatten()[i]
        ax.imshow(cm)
        ax.title.set_text(f"Matrice de confusion {os.path.basename(file_path).split('.')[0]}")
    else:
        print(f"Fichier de matrice de confusion non trouvé : {file_path}")
        ax = axes.flatten()[i]
        ax.text(0.5, 0.5, 'Fichier non trouvé', horizontalalignment='center', verticalalignment='center', fontsize=12)
        ax.title.set_text(f"Matrice de confusion {os.path.basename(file_path).split('.')[0]}")

plt.tight_layout()
plt.savefig("visualizations/confusion_matrices_combined.png")
plt.close()





