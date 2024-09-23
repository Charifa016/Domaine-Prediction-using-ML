

from sklearn.preprocessing import LabelEncoder
import joblib

# Liste complète des labels
all_labels = ['Civil', 'Mécanique', 'électrique', 'Ressources humaines', 'économique', 'Informatique','Juridique']  # Ajoutez tous les labels possibles

# Créer et entraîner le LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Sauvegarder le LabelEncoder mis à jour
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print("LabelEncoder sauvegardé avec succès.")



