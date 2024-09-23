import joblib

# Charger le LabelEncoder et le modèle
label_encoder = joblib.load('models/label_encoder.pkl')
decision_tree_model = joblib.load('models/decision_tree_model.pkl')

# Afficher les labels connus
print("Labels connus par le LabelEncoder :", label_encoder.classes_)
