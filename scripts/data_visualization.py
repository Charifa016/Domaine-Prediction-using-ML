import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assure-toi que tes données sont déjà chargées et prétraitées
data = pd.read_csv('C:\\Users\\chari\\Desktop\\project\\data\\extracted_texts\\cv_data.csv')

# Visualisation de la répartition des données par domaine
plt.figure(figsize=(10,6))
sns.countplot(x='Domaine', data=data, palette='viridis')
plt.title('Distribution des CVs par domaine')
plt.xticks(rotation=45)
plt.savefig('visualizations/data_distribution.png')
plt.show()

