
# 📦 Classification Automatique de Biens de Consommation

## 📚 Contexte du Projet

La "Place de Marché" est une plateforme d’e-commerce sur laquelle les vendeurs publient des articles à vendre en fournissant une **description textuelle** et une **photo**. Actuellement, la catégorisation des articles est **manuelle**, sujette à erreurs, incohérente, et fastidieuse.

🎯 **Objectif** : Automatiser la classification des articles (produits) dans les bonnes catégories, en exploitant **le texte et/ou l’image**, afin de :

- Garantir une **meilleure fiabilité** et **cohérence** de la catégorisation.
- **Optimiser l’expérience utilisateur** lors de la mise en ligne ou de la recherche de produits.

---

## 📁 Structure du Répertoire

```
classification-produits/
│
├── data/
│   ├── raw/               # Données brutes (descriptions, images)
│   ├── processed/         # Données après nettoyage et transformations
│   └── features/          # Vecteurs de features extraits (TF-IDF, SIFT, etc.)
│
├── notebooks/
│   ├── 01_preprocessing_text.ipynb     # Nettoyage des textes
│   ├── 02_preprocessing_images.ipynb   # Traitement des images
│   ├── 03_feature_extraction.ipynb     # Extraction des features textuelles et visuelles
│   ├── 04_dim_reduction.ipynb          # Réduction de dimension (PCA, t-SNE)
│   ├── 05_clustering.ipynb             # Algorithmes de clustering (TF-IDF, SIFT)
│   └── 06_topic_modeling.ipynb         # LDA / NMF pour exploration de topics
│
├── models/
│   └── cnn_transfer_learning.py        # Script pour test CNN (Transfer Learning)
│
├── results/
│   ├── figures/                        # Graphiques t-SNE, wordclouds, etc.
│   └── metrics/                        # Silhouette score, ARI, etc.
│
├── app/
│   └── demo_app.py                     # Démo de l’application de classification
│
├── requirements.txt                    # Librairies nécessaires
├── README.md                           # Ce fichier
└── .gitignore                          # Fichiers à ignorer par Git
```

---

## 🔍 Détails Techniques

### 📊 Jeu de Données

- 1050 produits, répartis dans **7 catégories de niveau 0**.
- Types de données :
  - **Textuelles** : nom, description, marque, catégorie.
  - **Visuelles** : images isolées sur fond blanc.

---

### 🔧 Prétraitement

#### Texte
- Suppression de la ponctuation
- Tokenisation
- Filtrage (stopwords, tokens < 2 lettres)
- Stemming & lemmatisation
- Vectorisation : **Bag of Words** (BoW), **TF-IDF**

#### Images
- Correction d'exposition & contraste
- Réduction du bruit
- Passage en niveaux de gris
- Redimensionnement : `224x224`

---

### 📈 Extraction des Caractéristiques (Features)

- **Textuelles** : TF-IDF, BoW
- **Visuelles** : 
  - **SIFT** (bag of visual words)
  - **CNN** préentraîné sur ImageNet (via Transfer Learning)

---

### 🧬 Réduction de Dimension

- **PCA** : Conserve jusqu’à 99% de la variance
- **t-SNE** : Projection 2D pour visualisation

---

### 📌 Clustering

- Algorithmes : K-Means
- Évaluation : 
  - **Silhouette Score**
  - **ARI (Adjusted Rand Index)**

| Méthode | Donnée       | Silhouette | ARI   |
|---------|--------------|------------|-------|
| KMeans  | TF-IDF       | 0.47       | 0.43  |
| KMeans  | SIFT         | 0.35       | -0.0008 |

---

### 🧠 Analyse Thématique (Topic Modeling)

- Méthodes : **LDA**, **NMF**
- Résultat : Certains topics correspondent bien à des catégories de produits (ex. Watches, Baby Care…)

---

### 🚀 Application

Une démonstration simple est disponible via le script `app/demo_app.py`.

---

## 🔧 Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
nltk
opencv-python
tensorflow / keras
scikit-image
gensim
umap-learn
```

---

## 👥 Équipe

- **Chaka KONE**
- **Mahamadi Bassirou COMPAORE**
- **Moussa DIAKITE**

**Superviseure** : Mme DIAW Mously – Lead ML Engineer, Experte IA & MLOps

---

## 📬 Contact

Pour toute question, contactez-nous via [GitHub].
