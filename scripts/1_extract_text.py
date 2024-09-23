import os
from PyPDF2 import PdfReader
import pandas as pd

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as f:
        reader = PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text
def extract_texts_from_directory(directory_path, output_csv, output_txt_dir):
    data = []
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                domain = os.path.basename(root)  # Le nom du dossier est le domaine
               
                data.append([domain, text])

                domain_txt_file = os.path.join(output_txt_dir, f"{domain}.txt")
                with open(domain_txt_file, 'a', encoding='utf-8') as domain_file:
                    domain_file.write(f"\n--- CV: {file} ---\n{text}\n")

    df = pd.DataFrame(data, columns=["Domaine", "Texte"])
    df.to_csv(output_csv, index=False)

# Chemins d'entrée et de sortie
directory_path = "data/raw/pdfs/"
output_csv = "data/extracted_texts/cv_data.csv"
output_txt_dir = "data/extracted_texts/domain_texts/"
extract_texts_from_directory(directory_path, output_csv, output_txt_dir)



