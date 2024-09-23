import pickle

# Charger le fichier train_data.pkl
with open('data/processed/train_data.pkl', 'rb') as file:
    data = pickle.load(file)

# Afficher le type et un aperçu des éléments du tuple
print("Type de données :", type(data))

if isinstance(data, tuple) and len(data) == 2:
    print("Premier élément du tuple (Données) :")
    print(data[0])  # Affiche les données

    print("\nDeuxième élément du tuple (Labels) :")
    print(data[1])  # Affiche les labels
else:
    print("Le format des données est inconnu ou le tuple ne contient pas deux éléments.")


