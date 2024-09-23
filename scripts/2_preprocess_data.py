import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

def clean_text(text, noisy_words):
    if not isinstance(text, str):  # Vérifier que text est une chaîne de caractères
        text = str(text) if text is not None else ""
    text = re.sub(r'\b\w+@\w+\.\w+\b', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in noisy_words])
    return text

def preprocess_data(input_file, output_csv, output_txt_dir, noisy_words):
    df = pd.read_csv(input_file)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('french'))
    
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    
    cleaned_data = []

    for index, row in df.iterrows():
        cleaned_text = clean_text(row["Texte"], noisy_words)
        cleaned_text = ' '.join([lemmatizer.lemmatize(word) 
                                 for word in cleaned_text.split() 
                                 if word not in stop_words])

        cleaned_data.append([row["Domaine"], cleaned_text])

        domain_txt_file = os.path.join(output_txt_dir, f"{row['Domaine']}_cleaned.txt")
        with open(domain_txt_file, 'a', encoding='utf-8') as domain_file:
            domain_file.write(f"\n--- CV: {index} ---\n{cleaned_text}\n")

    cleaned_df = pd.DataFrame(cleaned_data, columns=["Domaine", "Texte"])
    cleaned_df.to_csv(output_csv, index=False)

noisy_words = [
    "cv", "compétences", "permis", "formation", "fonction", "expérience", 
    "adresse", "téléphone", "email", "projet", "stage", "diplôme", "année",
    "Rue", "Nom", "français", "arabe", "espagnol", "allemand", "italien", "anglais", 
    "portugais", "russe", "chinois", "japonais", "hindi", "coréen", 
    "le", "la", "les", "un", "une", "des", "du", "de", "dans", "pour", 
    "et", "ou", "avec", "sans", "au", "aux", "par", "en", "à", "ce", 
    "cet", "cette", "sur", "sous", "son", "sa", "ses", "leur", "leurs",
    "nous", "vous", "ils", "elles", "rabat", "casablanca", "marrakech", 
    "fès", "meknès", "tanger", "agadir", "oujda", "tétouan", "kenitra", 
    "safi", "el jadida", "nador", "beni mellal", "taza", "chefchaouen", 
    "larache", "khouribga", "settat", "guelmim", "taroudant", "errachidia", 
    "essaouira", "midelt","Projets réalisés","Expérience professionnelle ","Compétences ","Lieu de résidence ",
    "Poste","Langue Maternelle","Courant","Intermédiaire"
]

# Fichier d'entrée et de sortie
input_file = "data/extracted_texts/cv_data.csv"
output_csv = "data/processed/cleaned_cv_data.csv"
output_txt_dir = "data/processed/domain_texts_cleaned/"

# Exécuter le prétraitement
preprocess_data(input_file, output_csv, output_txt_dir, noisy_words)



