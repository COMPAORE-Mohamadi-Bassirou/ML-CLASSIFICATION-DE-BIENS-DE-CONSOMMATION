
import pandas as pd
import numpy as np
import re
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import warnings
warnings.filterwarnings("ignore")

# Téléchargement des données NLTK si ce n'est pas déjà fait
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Fonction de prétraitement du texte (adaptée du notebook)
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower() # Convertir en minuscules
    text = re.sub(r'[^a-z\s]', '', text) # Supprimer les caractères non alphabétiques
    text = re.sub(r'\s+', ' ', text).strip() # Supprimer les espaces multiples
    return text

# Fonction pour extraire les catégories principales (adaptée du dashboard)
def extract_main_category(category_tree):
    if pd.isna(category_tree):
        return "Inconnu"
    try:
        categories = category_tree.strip('[]"').split(' >> ')
        return categories[0] if categories else "Inconnu"
    except:
        return "Inconnu"

# 1. Chargement des données
df = pd.read_csv("/home/ubuntu/flipkart_com-ecommerce_sample_1050.csv") # Chemin corrigé

# 2. Prétraitement des données
df['description_processed'] = df['description'].apply(preprocess_text)

# 3. TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df['description_processed'])

# 4. K-Means Clustering
kmeans_model = KMeans(n_clusters=5, random_state=42, n_init='auto') # Utilisation de 5 clusters comme exemple, à ajuster si nécessaire
df['cluster'] = kmeans_model.fit_predict(X_tfidf)

# 5. PCA pour la visualisation
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

# 6. Ajout de la catégorie principale au DataFrame
df['main_category'] = df['product_category_tree'].apply(extract_main_category)

# 7. Sauvegarde des modèles et données traitées
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_model, f)

with open('X_pca.pkl', 'wb') as f:
    pickle.dump(X_pca, f)

# Sauvegarde du DataFrame traité (avec les clusters et catégories principales)
df_processed = df[['product_category_tree', 'description', 'product_name', 'cluster', 'brand', 'retail_price', 'main_category']]
with open('df_processed.pkl', 'wb') as f:
    pickle.dump(df_processed, f)

print("Modèles et données sauvegardés avec succès.")


