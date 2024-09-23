import matplotlib.pyplot as plt
import pickle

# Chargement du modèle Random Forest
with open('models/random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

# Extraction des importances des caractéristiques
importances = random_forest_model.feature_importances_

# Génération d'un graphique des importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances)
plt.title('Importance des caractéristiques - Random Forest')
plt.xlabel('Index des caractéristiques')
plt.ylabel('Importance')
plt.savefig('visualizations/feature_importance.png')
plt.show()
