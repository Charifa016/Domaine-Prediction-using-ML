import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Exemple de données
data = {
    'Model': ['KNN', 'Decision Tree', 'SVM', 'Random Forest'],
    'Precision': [0.11, 0.58, 0.16, 0.53],
    'Recall': [0.12, 0.68, 0.15, 0.58],
    'F1-Score': [0.02, 0.61, 0.11, 0.57]
}

df = pd.DataFrame(data)

# Création des graphiques
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

# Bar Chart pour Precision
sns.barplot(x='Model', y='Precision', data=df, ax=ax[0], palette='viridis')
ax[0].set_title('Precision by Model')

# Bar Chart pour Recall
sns.barplot(x='Model', y='Recall', data=df, ax=ax[1], palette='viridis')
ax[1].set_title('Recall by Model')

# Bar Chart pour F1-Score
sns.barplot(x='Model', y='F1-Score', data=df, ax=ax[2], palette='viridis')
ax[2].set_title('F1-Score by Model')

# Affichage
plt.tight_layout()
plt.show()

