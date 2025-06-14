import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
import re
from PIL import Image
import io
import base64

# Configuration de la page
st.set_page_config(
    page_title="Dashboard ML - Classification de Biens de Consommation",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        border: 6px solid #0047AB;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        overflow-x: hidden;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 5px;
        border-left: 3px solid #3498db;
        border: 5px solid #0047AB;
        text-align: center;
        overflow-x: hidden;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        background-color: #a84bd1;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: black;
        border: 4px solid #0047AB;
    }
    
    html, body, [data-testid="stApp"] {
        overflow-x: hidden;
    }

    .main .block-container {
        padding-bottom: 1rem;
    }

    .dataframe-container {
        overflow-x: auto;
        width: 100%;
    }
    
    .dataframe {
        width: 100% !important;
        table-layout: auto;
    }
    
    .insight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    section[data-testid="stSidebar"] {
    position: fixed;
    top: 0;
    left: 0;
    height: 100vh;
    overflow-y: auto;
    border-right: fixed solid #dee2e6;

    }

</style>
""", unsafe_allow_html=True)

# Fonction pour charger les données et modèles
@st.cache_data
def load_data():
    try:
        with open('df_processed.pkl', 'rb') as f:
            df = pickle.load(f)
        return df
    except FileNotFoundError:
        st.error("Fichier de données non trouvé. Veuillez exécuter le script de préparation des modèles.")
        return pd.DataFrame()

@st.cache_resource
def load_models():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            tfidf_vectorizer = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        with open('X_pca.pkl', 'rb') as f:
            X_pca = pickle.load(f)
        return tfidf_vectorizer, kmeans_model, X_pca
    except FileNotFoundError:
        st.error("Modèles non trouvés. Veuillez exécuter le script de préparation des modèles.")
        return None, None, None

# Fonction pour extraire les catégories principales
def extract_main_category(category_tree):
    if pd.isna(category_tree):
        return "Inconnu"
    try:
        categories = category_tree.strip('[]"').split(' >> ')
        return categories[0] if categories else "Inconnu"
    except:
        return "Inconnu"

# Fonction pour créer un nuage de mots
def create_wordcloud(text_data, title="Nuage de mots"):
    text = ' '.join(text_data.dropna().astype(str))
    if text.strip():
        wordcloud = WordCloud(width=800, height=400, background_color='white', 
                            colormap='viridis', max_words=100).generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'Aucune donnée textuelle disponible', 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig

# Fonction de prédiction
def predict_category(text, tfidf_vectorizer, kmeans_model):
    if tfidf_vectorizer is None or kmeans_model is None:
        return "Modèles non disponibles"
    
    # Prétraitement du texte
    text_processed = str(text).lower()
    text_processed = re.sub(r'[^a-z\s]', '', text_processed)
    text_processed = re.sub(r'\s+', ' ', text_processed).strip()
    
    # Vectorisation
    text_tfidf = tfidf_vectorizer.transform([text_processed])
    
    # Prédiction du cluster
    cluster = kmeans_model.predict(text_tfidf)[0]
    
    return cluster

# Fonction pour analyser les insights business
def generate_business_insights(df):
    insights = []
    
    if 'retail_price' in df.columns:
        # Analyse des prix par catégorie
        price_by_category = df.groupby('main_category')['retail_price'].agg(['mean', 'count']).sort_values('mean', ascending=False)
        top_expensive = price_by_category.head(1)
        insights.append(f"La catégorie la plus chère en moyenne est '{top_expensive.index[0]}' avec {top_expensive['mean'].iloc[0]:.0f}€")
    
    # Analyse de la distribution des clusters
    if 'cluster' in df.columns:
        cluster_distribution = df['cluster'].value_counts()
        dominant_cluster = cluster_distribution.index[0]
        insights.append(f"Le cluster {dominant_cluster} est dominant avec {cluster_distribution.iloc[0]} produits ({cluster_distribution.iloc[0]/len(df)*100:.1f}%)")
    
    # Analyse des catégories
    category_distribution = df['main_category'].value_counts()
    top_category = category_distribution.index[0]
    insights.append(f"'{top_category}' représente la catégorie principale avec {category_distribution.iloc[0]} produits")
    
    return insights

# Chargement des données et modèles
df = load_data()
tfidf_vectorizer, kmeans_model, X_pca = load_models()

if df.empty:
    st.stop()

# Ajout de la catégorie principale si pas déjà présente
if 'main_category' not in df.columns:
    df['main_category'] = df['product_category_tree'].apply(extract_main_category)

# Header principal
st.markdown('<div class="main-header">CLASSIFICATION DE BIENS DE CONSOMMATION</div>', 
            unsafe_allow_html=True)

# Sidebar pour les filtres
st.sidebar.markdown("## 🔍 Filtres")

# Filtres
categories = ['Toutes'] + sorted(list(df['main_category'].unique()))
selected_category = st.sidebar.selectbox("Catégorie", categories)

if 'cluster' in df.columns:
    clusters = ['Tous'] + sorted(list(df['cluster'].unique()))
    selected_cluster = st.sidebar.selectbox("Cluster", clusters)
else:
    selected_cluster = 'Tous'

# Filtrage des données
filtered_df = df.copy()
if selected_category != 'Toutes':
    filtered_df = filtered_df[filtered_df['main_category'] == selected_category]
if selected_cluster != 'Tous':
    filtered_df = filtered_df[filtered_df['cluster'] == selected_cluster]

# Onglets principaux
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Résumé des Données", 
    "Visualisation des Clusters", 
    "Performance des Modèles", 
    "Analyse Textuelle", 
    "Prédiction",
    "Insights Business",
    "Exploration des Données",
    "À propos / Contact"
])

with tab1:
    st.markdown('<div class="section-header">Résumé des Données</div>', unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Nombre total d'observations", len(df))
    
    with col2:
        st.metric("Nombre de catégories", df['main_category'].nunique())
    
    with col3:
        if 'cluster' in df.columns:
            st.metric("Nombre de clusters", df['cluster'].nunique())
        else:
            st.metric("Nombre de clusters", "N/A")
    
    with col4:
        st.metric("Nombre de variables", len(df.columns))

    st.markdown("---")

    # Graphiques de distribution
    col1, col2 = st.columns(2)

    with col1:
        # Distribution des catégories
        category_counts = df['main_category'].value_counts().head(10)
        fig_cat = px.bar(
            x=category_counts.values, 
            y=category_counts.index,
            orientation='h',
            title="Top 10 des Catégories",
            labels={'x': 'Nombre de produits', 'y': 'Catégorie'},
            color=category_counts.values,
            color_continuous_scale='viridis'
        )
        fig_cat.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    with col2:
        # Distribution des clusters si disponible
        if 'cluster' in df.columns:
            cluster_counts = df['cluster'].value_counts().sort_index()
            fig_cluster = px.pie(
                values=cluster_counts.values,
                names=[f"Cluster {i}" for i in cluster_counts.index],
                title="Distribution des Clusters"
            )
            fig_cluster.update_layout(height=400)
            st.plotly_chart(fig_cluster, use_container_width=True)
        else:
            st.info("Données de clustering non disponibles")

    st.markdown("### Aperçu des données filtrées")
    st.dataframe(filtered_df.head(100), use_container_width=True)

with tab2:
    st.markdown('<div class="section-header">Visualisation des Clusters</div>', unsafe_allow_html=True)
    
    if X_pca is not None and len(X_pca) > 0:
        # Visualisation PCA
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Réduction de dimension - PCA")
            
            # Créer un DataFrame pour la visualisation
            pca_df = pd.DataFrame(X_pca[:len(df)], columns=['PC1', 'PC2'])
            if 'cluster' in df.columns:
                pca_df['Cluster'] = df['cluster'].values[:len(pca_df)]
                pca_df['Catégorie'] = df['main_category'].values[:len(pca_df)]
                
                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', 
                    color='Cluster',
                    hover_data=['Catégorie'],
                    title="Visualisation des Clusters (PCA)",
                    color_continuous_scale='viridis'
                )
            else:
                pca_df['Catégorie'] = df['main_category'].values[:len(pca_df)]
                fig_pca = px.scatter(
                    pca_df, x='PC1', y='PC2', 
                    color='Catégorie',
                    title="Visualisation par Catégorie (PCA)"
                )
            
            fig_pca.update_layout(height=500)
            st.plotly_chart(fig_pca, use_container_width=True)
        
        with col2:
            st.markdown("#### Métriques de Clustering")
            
            if 'cluster' in df.columns:
                # Calculer le score de silhouette
                try:
                    X_sample = X_pca[:min(1000, len(X_pca))]
                    clusters_sample = df['cluster'].values[:len(X_sample)]
                    silhouette_avg = silhouette_score(X_sample, clusters_sample)
                    
                    st.metric("Score de Silhouette", f"{silhouette_avg:.3f}")
                    
                    # Interprétation du score
                    if silhouette_avg > 0.7:
                        st.success("Excellent clustering")
                    elif silhouette_avg > 0.5:
                        st.info("Bon clustering")
                    elif silhouette_avg > 0.25:
                        st.warning("Clustering acceptable")
                    else:
                        st.error("Clustering faible")
                        
                except Exception as e:
                    st.error(f"Erreur dans le calcul du score de silhouette: {e}")
                
                # Distribution des tailles de clusters
                cluster_sizes = df['cluster'].value_counts().sort_index()
                fig_sizes = px.bar(
                    x=[f"Cluster {i}" for i in cluster_sizes.index],
                    y=cluster_sizes.values,
                    title="Taille des Clusters",
                    color=cluster_sizes.values,
                    color_continuous_scale='plasma'
                )
                fig_sizes.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_sizes, use_container_width=True)
            else:
                st.info("Données de clustering non disponibles")
    else:
        st.warning("Données de réduction de dimension non disponibles")

with tab3:
    st.markdown('<div class="section-header">Performance des Modèles</div>', unsafe_allow_html=True)
    
    # Métriques de performance réelles basées sur le clustering
    if 'cluster' in df.columns and X_pca is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Score de silhouette
            try:
                silhouette_avg = silhouette_score(X_pca[:min(1000, len(X_pca))], 
                                                df['cluster'].values[:min(1000, len(X_pca))])
                st.metric("Score de Silhouette", f"{silhouette_avg:.3f}")
            except:
                st.metric("Score de Silhouette", "N/A")
        
        with col2:
            # Inertie du modèle K-means
            if kmeans_model is not None:
                st.metric("Inertie K-means", f"{kmeans_model.inertia_:.0f}")
            else:
                st.metric("Inertie K-means", "N/A")
        
        with col3:
            # Nombre de clusters
            st.metric("Nombre de clusters", df['cluster'].nunique())
    
    # Analyse de la distribution des clusters par catégorie
    if 'cluster' in df.columns:
        st.markdown("#### Distribution des Clusters par Catégorie")
        
        # Créer une matrice de confusion entre clusters et catégories
        cluster_category_matrix = pd.crosstab(df['cluster'], df['main_category'])
        
        fig_heatmap = px.imshow(
            cluster_category_matrix.values,
            x=cluster_category_matrix.columns,
            y=[f"Cluster {i}" for i in cluster_category_matrix.index],
            title="Matrice Cluster vs Catégorie",
            color_continuous_scale='Blues',
            aspect='auto'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Tableau de correspondance
        st.markdown("#### Correspondance Clusters-Catégories")
        st.dataframe(cluster_category_matrix, use_container_width=True)

with tab4:
    st.markdown('<div class="section-header">Analyse Textuelle</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Nuage de mots - Toutes catégories")
        if 'description' in df.columns:
            fig_wordcloud = create_wordcloud(df['description'], "Mots les plus fréquents")
            st.pyplot(fig_wordcloud)
        else:
            st.info("Données textuelles non disponibles")
    
    with col2:
        st.markdown("#### Nuage de mots - Catégorie sélectionnée")
        if selected_category != 'Toutes' and 'description' in filtered_df.columns:
            fig_wordcloud_filtered = create_wordcloud(
                filtered_df['description'], 
                f"Mots fréquents - {selected_category}"
            )
            st.pyplot(fig_wordcloud_filtered)
        else:
            st.info("Sélectionnez une catégorie spécifique pour voir son nuage de mots")
    
    # Analyse des termes les plus fréquents par cluster
    if 'cluster' in df.columns and 'description' in df.columns:
        st.markdown("#### Analyse par Cluster")
        
        selected_cluster_analysis = st.selectbox(
            "Sélectionnez un cluster pour l'analyse textuelle:",
            sorted(df['cluster'].unique())
        )
        
        cluster_data = df[df['cluster'] == selected_cluster_analysis]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"##### Nuage de mots - Cluster {selected_cluster_analysis}")
            fig_cluster_wordcloud = create_wordcloud(
                cluster_data['description'],
                f"Cluster {selected_cluster_analysis}"
            )
            st.pyplot(fig_cluster_wordcloud)
        
        with col2:
            st.markdown(f"##### Statistiques - Cluster {selected_cluster_analysis}")
            st.metric("Nombre de produits", len(cluster_data))
            
            # Top catégories dans ce cluster
            top_categories = cluster_data['main_category'].value_counts().head(5)
            st.markdown("**Top 5 catégories:**")
            for cat, count in top_categories.items():
                st.write(f"- {cat}: {count} produits")

with tab5:
    st.markdown('<div class="section-header">Prédiction de Cluster</div>', unsafe_allow_html=True)
    
    if tfidf_vectorizer is not None and kmeans_model is not None:
        st.markdown("### Prédire le cluster d'un nouveau produit")
        
        # Interface de prédiction
        user_input = st.text_area(
            "Entrez la description d'un produit:",
            placeholder="Ex: Smartphone avec écran OLED, 128GB de stockage, appareil photo 48MP...",
            height=100
        )
        
        if st.button("Prédire le cluster", type="primary"):
            if user_input.strip():
                predicted_cluster = predict_category(user_input, tfidf_vectorizer, kmeans_model)
                
                st.success(f"Cluster prédit: **{predicted_cluster}**")
                
                # Afficher des informations sur le cluster prédit
                cluster_info = df[df['cluster'] == predicted_cluster]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Informations sur ce cluster")
                    st.metric("Nombre de produits", len(cluster_info))
                    
                    # Top catégories dans ce cluster
                    top_categories = cluster_info['main_category'].value_counts().head(3)
                    st.markdown("**Principales catégories:**")
                    for cat, count in top_categories.items():
                        percentage = (count / len(cluster_info)) * 100
                        st.write(f"- {cat}: {percentage:.1f}%")
                
                with col2:
                    st.markdown("#### Exemples de produits similaires")
                    sample_products = cluster_info.sample(min(5, len(cluster_info)))
                    for _, product in sample_products.iterrows():
                        st.write(f"• {product['product_name'][:80]}...")
            else:
                st.warning("Veuillez entrer une description de produit.")
        
        # Section d'exemples
        st.markdown("### Exemples de descriptions")
        
        examples = [
            "Smartphone Android avec écran 6.5 pouces, 64GB stockage, double caméra",
            "T-shirt en coton 100% pour homme, taille M, couleur bleue",
            "Livre de cuisine française avec 200 recettes traditionnelles",
            "Casque audio sans fil Bluetooth avec réduction de bruit",
            "Robe d'été pour femme en lin, taille S, motif floral"
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"Tester: {example[:50]}...", key=f"example_{i}"):
                predicted_cluster = predict_category(example, tfidf_vectorizer, kmeans_model)
                st.info(f"Cluster prédit pour cet exemple: **{predicted_cluster}**")
    
    else:
        st.error("Modèles de prédiction non disponibles. Veuillez vérifier que les modèles ont été correctement chargés.")

with tab6:
    st.markdown('<div class="section-header">Insights Business</div>', unsafe_allow_html=True)
    
    # Génération d'insights automatiques
    insights = generate_business_insights(df)
    
    st.markdown("### Insights Automatiques")
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Analyse des prix si disponible
    if 'retail_price' in df.columns:
        st.markdown("### Analyse des Prix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des prix par catégorie
            price_stats = df.groupby('main_category')['retail_price'].agg(['mean', 'median', 'count']).round(2)
            price_stats = price_stats.sort_values('mean', ascending=False).head(10)
            
            fig_price = px.bar(
                x=price_stats['mean'],
                y=price_stats.index,
                orientation='h',
                title="Prix moyen par catégorie (Top 10)",
                labels={'x': 'Prix moyen (€)', 'y': 'Catégorie'}
            )
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            # Box plot des prix
            top_categories = df['main_category'].value_counts().head(5).index
            df_top = df[df['main_category'].isin(top_categories)]
            
            fig_box = px.box(
                df_top, 
                x='main_category', 
                y='retail_price',
                title="Distribution des prix (Top 5 catégories)"
            )
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # Recommandations business
    st.markdown("### Recommandations Business")
    
    recommendations = [
        "Optimiser l'inventaire en se concentrant sur les catégories dominantes",
        "Analyser les clusters pour identifier des opportunités de cross-selling",
        "Utiliser la classification automatique pour améliorer la recherche produit",
        "Développer des stratégies marketing ciblées par cluster de produits"
    ]
    
    for rec in recommendations:
        st.markdown(f'<div class="recommendation-box">{rec}</div>', unsafe_allow_html=True)

with tab7:
    st.markdown('<div class="section-header">Exploration des Données</div>', unsafe_allow_html=True)
    
    # Filtres avancés
    st.markdown("### 🔍 Filtres Avancés")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'brand' in df.columns:
            brands = ['Toutes'] + sorted([str(brand) for brand in df['brand'].unique() if pd.notna(brand)])
            selected_brand = st.selectbox("Marque", brands)
        else:
            selected_brand = 'Toutes'
    
    with col2:
        if 'retail_price' in df.columns:
            price_range = st.slider(
                "Gamme de prix",
                min_value=float(df['retail_price'].min()),
                max_value=float(df['retail_price'].max()),
                value=(float(df['retail_price'].min()), float(df['retail_price'].max()))
            )
        else:
            price_range = None
    
    with col3:
        # Recherche textuelle
        search_term = st.text_input("Recherche dans les descriptions", "")
    
    # Application des filtres
    explore_df = df.copy()
    
    if selected_brand != 'Toutes' and 'brand' in df.columns:
        explore_df = explore_df[explore_df['brand'] == selected_brand]
    
    if price_range and 'retail_price' in df.columns:
        explore_df = explore_df[
            (explore_df['retail_price'] >= price_range[0]) & 
            (explore_df['retail_price'] <= price_range[1])
        ]
    
    if search_term and 'description' in df.columns:
        explore_df = explore_df[
            explore_df['description'].str.contains(search_term, case=False, na=False)
        ]
    
    # Affichage des résultats
    st.markdown(f"### Résultats de l'exploration ({len(explore_df)} produits)")
    
    if len(explore_df) > 0:
        # Statistiques rapides
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Produits trouvés", len(explore_df))
        
        with col2:
            st.metric("Catégories uniques", explore_df['main_category'].nunique())
        
        with col3:
            if 'brand' in explore_df.columns:
                st.metric("Marques uniques", explore_df['brand'].nunique())
        
        with col4:
            if 'retail_price' in explore_df.columns:
                st.metric("Prix moyen", f"{explore_df['retail_price'].mean():.0f}€")
        
        # Tableau des résultats
        st.dataframe(explore_df, use_container_width=True)
        
        # Option de téléchargement
        csv = explore_df.to_csv(index=False)
        st.download_button(
            label="Télécharger les résultats (CSV)",
            data=csv,
            file_name="exploration_results.csv",
            mime="text/csv"
        )
    else:
        st.warning("Aucun produit ne correspond aux critères de recherche.")

with tab8:
    st.markdown('<div class="section-header">À propos du Projet</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Description du Projet
        
        Ce dashboard présente un système de **classification automatisée de biens de consommation** 
        basé sur l'analyse de texte et le machine learning.
        
        **Objectifs:**
        - Classifier automatiquement les produits en catégories
        - Analyser les patterns dans les descriptions de produits
        - Fournir des insights sur la distribution des produits
        
        **Technologies utilisées:**
        - **Python** pour le traitement des données
        - **Scikit-learn** pour le machine learning
        - **Streamlit** pour l'interface web
        - **Plotly** pour les visualisations
        """)
    
    with col2:
        st.markdown("""
        ###  Équipe de Développement
        
        **Auteur:** Bassirou Chaka Moussa  
        **Statut:** Élève ingénieur statisticien économiste
        
        ### Méthodologie
        
        1. **Prétraitement des données**
           - Nettoyage des descriptions textuelles
           - Extraction des catégories principales
        
        2. **Modélisation**
           - Vectorisation TF-IDF
           - Clustering K-means
           - Réduction de dimension (PCA)
        
        3. **Évaluation**
           - Score de silhouette
           - Analyse de la cohérence des clusters
        """)
    
    st.markdown("---")
    
    # Statistiques du dataset
    st.markdown("### Statistiques du Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Produits", f"{len(df):,}")
    
    with col2:
        st.metric("Catégories Uniques", df['main_category'].nunique())
    
    with col3:
        if 'brand' in df.columns:
            st.metric("Marques Uniques", df['brand'].nunique())
        else:
            st.metric("Marques Uniques", "N/A")
    
    with col4:
        if 'cluster' in df.columns:
            st.metric("Clusters Générés", df['cluster'].nunique())
        else:
            st.metric("Clusters Générés", "N/A")
    
    # Contact et support
    st.markdown("""
    ### 📞 Contact et Support
    
    Pour toute question ou suggestion concernant ce projet, n'hésitez pas à nous contacter.
    
    **Note:** Ce dashboard est un projet académique dans le cadre de la formation en 
    ingénierie statistique et économique.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Dashboard ML - Classification de Biens de Consommation | "
    "Développé avec ❤️ par Bassirou Chaka Moussa"
    "</div>", 
    unsafe_allow_html=True
)

