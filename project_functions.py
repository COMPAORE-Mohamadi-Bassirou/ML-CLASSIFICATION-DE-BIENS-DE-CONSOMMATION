############### PROJET Machine learning #################

import time
import pandas as pd
import numpy as np
import string as st
import re
import os
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns
import matplotlib.cm as cm
from sklearn import (preprocessing,
                     manifold,
                     decomposition)
from sklearn import metrics
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MeanShift, estimate_bandwidth
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from sklearn.metrics.cluster import adjusted_rand_score
from math import radians, cos, sin, asin, sqrt
import nltk
from nltk.corpus import stopwords
from nltk import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.preprocessing import FunctionTransformer
from scipy.optimize import linear_sum_assignment

from PIL import Image, ImageOps, ImageFilter
import cv2 as cv
from sklearn.metrics import pairwise_distances_argmin_min
import warnings

def describe_dataset(source_files):
    '''
        Outputs a presentation pandas dataframe for the dataset.

        Parameters
        ----------------
        sourceFiles     : dict with :
                            - keys : the names of the files
                            - values : a list containing two values :
                                - the dataframe for the data
                                - a brief description of the file

        Returns
        ---------------
        presentation_df : pandas dataframe :
                            - a column "Nom du fichier" : the name of the file
                            - a column "Nb de lignes"   : the number of rows per file
                            - a column "Nb de colonnes" : the number of columns per file
                            - a column "Description"    : a brief description of the file
    '''

    print("Les données se décomposent en {} fichier(s): \n".format(len(source_files)))

    filenames = []
    files_nb_lines = []
    files_nb_columns = []
    files_descriptions = []

    for filename, file_data in source_files.items():
        filenames.append(filename)
        files_nb_lines.append(len(file_data[0]))
        files_nb_columns.append(len(file_data[0].columns))
        files_descriptions.append(file_data[1])

    # Create a dataframe for presentation purposes
    presentation_df = pd.DataFrame({'Nom du fichier':filenames,
                                    'Nb de lignes':files_nb_lines,
                                    'Nb de colonnes':files_nb_columns,
                                    'Description': files_descriptions})

    presentation_df.index += 1

    return presentation_df

#------------------------------------------

def get_missing_values_percent_per(data):
    '''
        Calculates the mean percentage of missing values
        in a given pandas dataframe per unique value
        of a given column

        Parameters
        ----------------
        data                : pandas dataframe
                              The dataframe to be analyzed

        Returns
        ---------------
        missing_percent_df  : A pandas dataframe containing:
                                - a column "column"
                                - a column "Percent Missing" containing the percentage of
                                  missing value for each value of column
    '''

    missing_percent_df = pd.DataFrame({'Percent Missing':data.isnull().sum()/len(data)*100})
    missing_percent_df['Percent Filled'] = 100 - missing_percent_df['Percent Missing']
    missing_percent_df['Total'] = 100

    return missing_percent_df


#------------------------------------------

def plot_percentage_missing_values_for(data, long, larg):
    '''
        Plots the proportions of filled / missing values for each unique value
        in column as a horizontal bar chart.

        Parameters
        ----------------
        data : pandas dataframe with:
                - a column column
                - a column "Percent Filled"
                - a column "Percent Missing"
                - a column "Total"

       long : int
            The length of the figure for the plot

        larg : int
               The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    data_to_plot = get_missing_values_percent_per(data)\
                     .sort_values("Percent Filled").reset_index()

    # Constants for the plot
    TITLE_SIZE = 60
    TITLE_PAD = 100
    TICK_SIZE = 50
    TICK_PAD = 20
    LABEL_SIZE = 50
    LABEL_PAD = 50
    LEGEND_SIZE = 50

    sns.set(style="whitegrid")

    #sns.set_palette(sns.dark_palette("purple", reverse=True))

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title("PROPORTIONS DE VALEURS RENSEIGNÉES / NON-RENSEIGNÉES PAR COLONNE",
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    handle_plot_1 = sns.barplot(x="Total", y="index", data=data_to_plot,
                                label="non renseignées", color="thistle", alpha=0.3)

    handle_plot_1.set_xticklabels(handle_plot_1.get_xticks(), size=TICK_SIZE)
    _, ylabels = plt.yticks()
    handle_plot_1.set_yticklabels(ylabels, size=TICK_SIZE)


    # Plot the Percent Filled values
    handle_plot_2 = sns.barplot(x="Percent Filled", y="index", data=data_to_plot,
                                label="renseignées", color="darkviolet")

    handle_plot_2.set_xticklabels(handle_plot_2.get_xticks(), size=TICK_SIZE)
    handle_plot_2.set_yticklabels(ylabels, size=TICK_SIZE)


    # Add a legend and informative axis label
    axis.legend(bbox_to_anchor=(1.04, 0), loc="lower left",
                borderaxespad=0, ncol=1, frameon=True, fontsize=LEGEND_SIZE)

    axis.set(ylabel="Colonnes", xlabel="Pourcentage de valeurs (%)")

    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    axis.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,
                                                               pos: '{:2d}'.format(int(x)) + '%'))
    axis.tick_params(axis='both', which='major', pad=TICK_PAD)

    sns.despine(left=True, bottom=True)

    # Display the figure
    plt.show()

#------------------------------------------

def plot_clusters_repartition(clusters_labels):
    '''
    Affiche la répartition par cluster
    Parameters
    ----------
    clusters_labels : la séries des labels des clusters, obligatoire.
    Returns
    -------
    None.
    '''
    ax1 = plt.gca()

    # DataFrame de travail
    series_client_cluster = pd.Series(clusters_labels).value_counts()
    nb_client = series_client_cluster.sum()
    df_visu_client_cluster = pd.DataFrame(
        {'Clusters': series_client_cluster.index,
         'Nombre': series_client_cluster.values})
    #df_visu_client_cluster['%'] = round(
     #   (df_visu_client_cluster['Nombre']) * 100 / nb_client, 2)
    df_visu_client_cluster = df_visu_client_cluster.sort_values(by='Clusters')
    display(df_visu_client_cluster.style.hide_index())

    # Barplot de la distribution
    sns.set_style('white')
    sns.barplot(x='Clusters', y='Nombre',
                data=df_visu_client_cluster, color='SteelBlue', ax=ax1)
    ax1.set_ylabel('Nombre', fontsize=12)
    ax1.set_xlabel('Clusters', fontsize=12)
    ax1.set_title('Répartition par clusters', fontsize=16, fontweight='bold')
    plt.gcf().set_size_inches(6, 4)
    plt.grid(False)
    plt.show()

#------------------------------------------

def plot_stacked_bar_clust_vs_cat(ser_clust, ser_cat, data, figsize=(8,4),
                                  palette='tab10', ylim=(0,250),
                                  bboxtoanchor=None):
    
    # pivot = data.drop(columns=['description','image'])
    pivot = pd.DataFrame()
    pivot['label']=ser_clust
    pivot['category']=ser_cat
    pivot['count']=1
    pivot = pivot.groupby(by=['label','category']).count().unstack().fillna(0)
    pivot.columns=pivot.columns.droplevel()
    
    colors = sns.color_palette(palette, ser_clust.shape[0]).as_hex()
    pivot.plot.bar(width=0.8,stacked=True,legend=True,figsize=figsize,
                   color=colors, ec='k')

    row_data=data.shape[0]

    if ser_clust.nunique() > 15:
        font = 8 
    else : 
        font = 12

    for index, value in enumerate(ser_clust.value_counts().sort_index(ascending=True)):
        percentage = np.around(value/row_data*100,1)   
        plt.text(index-0.25, value+2, str(percentage)+' %',fontsize=font)

    plt.gca().set(ylim=ylim)
    plt.xticks(rotation=0) 

    plt.xlabel('Clusters',fontsize=14)
    plt.ylabel('Nombre de produits', fontsize=14)
    plt.title('Répartition des vraies catégories par cluster',
              fontweight='bold', fontsize=18)

    if bboxtoanchor is not None:
        plt.legend(bbox_to_anchor=bboxtoanchor)
        
    plt.show()    
    
    return pivot
#------------------------------------------

def conf_matr_max_diago(true_cat, clust_lab, normalize=False):

    '''
    Takes two series giving for each row :
    - the true label
    - the cluster label
    Then computes the matrix counting each pair of true category/ cluster label.
    Then reorder the lines and columns in order to have maximum diagonal.
    The best bijective correspondance between categories and clusters is obtained by
     list(zip(result.columns, result.index))
    '''

    ### Count the number of articles in eact categ/clust pair
    cross_tab = pd.crosstab(true_cat, clust_lab,
                            normalize=normalize)

    ### Rearrange the lines and columns to maximize the diagonal values sum
    # Take the invert values in the matrix
    func = lambda x: 1/(x+0.0000001)
    inv_func = lambda x: (1/x) - 0.0000001
    funct_trans = FunctionTransformer(func, inv_func)
    inv_df = funct_trans.fit_transform(cross_tab)

    # Use hungarian algo to find ind and row order that minimizes inverse
    # of the diag vals -> max diag vals
    row_ind, col_ind = linear_sum_assignment(inv_df.values)
    inv_df = inv_df.loc[inv_df.index[row_ind],
                        inv_df.columns[col_ind]]

    # Take once again inverse to go back to original values
    cross_tab = funct_trans.inverse_transform(inv_df)
    result = cross_tab.copy(deep='True')

    if normalize == False:
        result = result.round(0).astype(int)

    return result

#------------------------------------------

def plot_conf_matrix_cat_vs_clust(true_cat, clust_lab, normalize=False,
                                  margins_sums=False, margins_score=False):
    '''
    Takes two series giving for each row :
    - the true label
    - the cluster label
    Then computes the matrix counting each pair of true category / cluster label.
    Then reorders the lines and columns to have a maximum diagonal.
    '''

    cross_tab = conf_matr_max_diago(true_cat, clust_lab, normalize=normalize)
    result = cross_tab.copy(deep=True)

    if margins_sums:
        marg_sum_vert = cross_tab.sum(axis=1)
        result = result.assign(cat_sum=marg_sum_vert)

        marg_sum_hor = cross_tab.sum(axis=0)
        clust_sum_row = pd.DataFrame([marg_sum_hor], columns=cross_tab.columns, index=['clust_sum'])
        result = pd.concat([result, clust_sum_row])

    if margins_score:
        li_cat_clust = list(zip(cross_tab.index, cross_tab.columns))
        li_cat_corresp_score, li_clust_corresp_score = [], []

        for i, (cat, clust) in enumerate(li_cat_clust):
            match = result.loc[cat, clust]
            cat_corr_score = 100 * match / cross_tab.sum(axis=1).iloc[i]
            clust_corr_score = 100 * match / cross_tab.sum(axis=0).iloc[i]
            li_cat_corresp_score.append(cat_corr_score)
            li_clust_corresp_score.append(clust_corr_score)

        if margins_sums:
            li_cat_corresp_score.append('-')
            li_clust_corresp_score.append('-')

        marg_vert = pd.Series(li_cat_corresp_score, index=result.index, name='cat_matching_score_pct')
        result = result.assign(cat_matching_score_pct=marg_vert)

        marg_hor = pd.Series(li_clust_corresp_score + ['-'], index=result.columns, name='clust_matching_score_pct')
        clust_score_row = pd.DataFrame([marg_hor], columns=result.columns, index=['clust_matching_score_pct'])
        result = pd.concat([result, clust_score_row])

    result = result.fillna('-')

    return result

# -------------------------------------------------------------------

#------------------------------------------

def plot_heatmap(corr, title, figsize=(8, 4), vmin=-1, vmax=1, center=0,
                 palette=sns.color_palette("coolwarm", 20), shape='rect',
                 fmt='.2f', robust=False, fig=None, ax=None):
    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_subplot(111) if ax is None else ax

    if shape == 'rect':
        mask = None
    elif shape == 'tri':
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
    else:
        print('ERROR : this type of heatmap does not exist')

    ax = sns.heatmap(corr, mask=mask, cmap=palette, vmin=vmin, vmax=vmax,
                     center=center, annot=True, annot_kws={"size": 10}, fmt=fmt,
                     square=False, linewidths=.5, linecolor='white',
                     cbar_kws={"shrink": .9, 'label': None}, robust=robust,
                     xticklabels=corr.columns, yticklabels=corr.index,
                     ax=ax)
    ax.tick_params(labelsize=10, top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    ax.collections[0].colorbar.ax.tick_params(labelsize=10)
    plt.ylabel('Categories',fontsize=14)
    plt.xlabel('Clusters',fontsize=14)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    ax.set_title(title, fontweight='bold', fontsize=18)

#------------------------------------------



#------------------------------------------
#TEXTES
#------------------------------------------


def remove_punct(text):
    '''
    Remove all punctuations from the text
    
    '''
    return ("".join([ch for ch in text if ch not in st.punctuation]))
#------------------------------------------

def tokenize(text):
    '''
    Convert text to lower case tokens. Here, split() is applied on white-spaces. But,
    it could be applied on special characters, tabs or any other string based on which
    text is to be seperated into tokens.
    
    '''
    text = re.split('\s+' ,text)
    return [x.lower() for x in text]

#------------------------------------------

def remove_small_words(text):
    '''
    Remove tokens of length less than 2
    '''
    return [x for x in text if len(x) > 1 ]

#------------------------------------------

def remove_stopwords(text):
    ''' 
    Remove stopwords. Here, NLTK corpus list is used for a match. However, a customized
    user-defined list could be created and used to limit the matches in input text. 

    '''
    return [word for word in text if word not in nltk.corpus.stopwords.words('english')]

#------------------------------------------
 
def stemming(text):
    '''
    Apply stemming to get root words
    '''
    ps = PorterStemmer()
    return [ps.stem(word) for word in text]

#------------------------------------------

def lemmatize(text):
    '''
    Apply lemmatization on tokens
    '''
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]

#------------------------------------------

def return_sentences(tokens):
    '''
    Create sentences to get clean text as input for vectors
    '''
    return " ".join([word for word in tokens])

#------------------------------------------
def get_most_freq(data_series, nb):
    '''
        Counts the occurrences of each words in data_series
        and returns the nb most frequent with their associated
        count.
        
        Parameters
        ----------------
        data_series : pandas series
                      The corpus of documents

        - nb        : int
                      The number of most frequent words to
                      return

        Returns
        ---------------
        df   : pandas dataframe
               The nb most frequent words with their associated
               count.
    '''
    
    all_words = []

    for word_list in data_series:
        all_words += word_list
        
    freq_dict = nltk.FreqDist(all_words)

    df = pd.DataFrame.from_dict(freq_dict, orient='index').rename(columns={0:"freq"})

    return df.sort_values(by="freq", ascending=False).head(nb)

#------------------------------------------
def plot_freq_dist(data_df, title, long, larg):
    '''
        Displays a bar chart showing the frequency of the modalities
        for each column of data.

        Parameters
        ----------------
        data  : dataframe
                Working data containing exclusively qualitative data
               
        title : string
                The title to give the plot

        long  : int
                The length of the figure for the plot

        larg  : int
                The width of the figure for the plot

        Returns
        ---------------
        -
    '''

    TITLE_SIZE = 40
    TITLE_PAD = 80
    TICK_SIZE = 12
    LABEL_SIZE = 30
    LABEL_PAD = 20

    # Initialize the matplotlib figure
    _, axis = plt.subplots(figsize=(long, larg))

    plt.title(title,
              fontweight="bold",
              fontsize=TITLE_SIZE, pad=TITLE_PAD)

    # Plot the Total values
    data_to_plot = data_df.reset_index().rename(columns={"index":"words"})
    handle_plot_1 = sns.barplot(x="words", y="freq", data=data_to_plot,
                                label="non renseignées", color="darkviolet", alpha=1)

    _, xlabels = plt.xticks()
    _ = handle_plot_1.set_xticklabels(xlabels, size=TICK_SIZE, rotation=45)
    
    x_label = axis.get_xlabel()
    axis.set_xlabel(x_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

    y_label = axis.get_ylabel()
    axis.set_ylabel(y_label, fontsize=LABEL_SIZE, labelpad=LABEL_PAD, fontweight="bold")

#------------------------------------------

def plot_wordclouds_from_gb(gb, n_top_words, n_rows=1, figsize=(18,8),
                             backgnd_color='black', cmap='Dark2',
                            random_state=None):
    '''Takes a groupby made on a series of texts (non tokenized),
    (-> for example : gb = df_desc_cat.groupby('category')['desc_clean'])
    extracts the n top words and plots a wordcloud of the (n_top_words)
    top words for each topic.
    '''

    fig = plt.figure(figsize=figsize)

    for i, tup in enumerate(gb,1):
        n_topic, ser_texts = tup
        # creation of a corpus of all the cleaned descriptions and product_names
        corpus = ' '.join(ser_texts.values)
        # tokenizing the words in the cleaned corpus
        tokenizer = nltk.RegexpTokenizer(r'[a-z]+')
        li_words = tokenizer.tokenize(corpus.lower())
        # counting frequency of each word
        ser_freq = pd.Series(nltk.FreqDist(li_words))

        wc = WordCloud(stopwords=None, background_color=backgnd_color,
                        colormap=cmap, max_font_size=150,
                        random_state=14)
        ser_topic = ser_freq\
            .sort_values(ascending=False)[0:n_top_words]
        wc.generate(' '.join(list(ser_topic.index)))

        n_tot = len(gb)
        n_cols = (n_tot//n_rows)+((n_tot%n_rows)>0)*1
        ax = fig.add_subplot(n_rows,n_cols,i)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.title(n_topic, fontweight='bold')
        
#------------------------------------------

def print_top_words(model, feature_names, n_top_words):
    '''
        Prints the n_top_words found by a text analysis
        model (LDA or NMF)

        Parameters
        ----------------
        - model         : a fitted LDA or NMF model

        - feature_names : list
                          The ordered name of the features
                          used by the data the model
                          was fitted on

        Returns
        ---------------
        _
    '''
    
    for topic_idx, topic in enumerate(model.components_):
        print("Catégorie #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))

    print()
    
#------------------------------------------

def get_categories_per_topic(n_topics, model, text_data, product_data):
    '''
        Returns the proportion of products from each category
        for each topic.
        
        Parameters
        ----------------
        - n_topics     : int
                         Number of topics
        
        - model        : fitted model
        - text_data    : pandas dataframe
                         input data
        - product_data : pandas dataframe
                         the dataframe containing the
                         original data, including the
                         categories

        Returns
        ---------------
        - categories_topics : pandas dataframe
                              the proportion of products
                              from each category for each topic
    '''
    
    doc_topic = model.transform(text_data)
    
    products_topic = product_data.copy()
    
    topics = []
    
    for n in range(doc_topic.shape[0]):
        topic_most_pr = doc_topic[n].argmax()
        topics.append(topic_most_pr)
        
    products_topic["topics"] = topics
    
    percent_predicted = pd.DataFrame(columns=["Topic",
                                          "% Home Furnishing",
                                          "% Baby Care",
                                          "% Watches",
                                          "% Home Decor & Festive Needs",
                                          "% Kitchen & Dining",
                                          "% Beauty and Personal Care",
                                          "% Computers"])

    for topic, data_topic in products_topic.groupby("topics"):
        
        row = [topic,
               (len(data_topic[data_topic["category"]=="Home Furnishing"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Baby Care"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Watches"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Home Decor & Festive Needs"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Kitchen & Dining"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Beauty and Personal Care"])*100)/len(data_topic),
               (len(data_topic[data_topic["category"]=="Computers"])*100)/len(data_topic)]
        
        percent_predicted.loc[len(percent_predicted)] = row
    
    percent_predicted = percent_predicted.set_index("Topic")

    return percent_predicted

#------------------------------------------

def creer_vecteur_moyen_par_mot(data, text_dim, w2v_model):

    vect_moy = np.zeros((text_dim,), dtype='float32')
    num_words = 0.

    for word in data.split():
        if word in w2v_model.wv.vocab:
            vect_moy = np.add(vect_moy, w2v_model[word])
            num_words += 1.

    if num_words != 0.:
        vect_moy = np.divide(vect_moy, num_words)

    return vect_moy
#------------------------------------------

def word2vec_vectorisation(data, text_dim, w2v_model):
    '''
    Vectorisation.
    Parameters
    ----------
    data : variable à vectoriser, obligatoire.
    text_dim : taille du vecteur, obligatoire.
    w2v_model : modèle Word2Vec entraîné, obligatoire.
    Returns
    -------
    w2v_vector : les words vectorisés.
    '''
    w2v_vector = np.zeros((data.shape[0], text_dim), dtype='float32')

    for i in range(len(data)):
        w2v_vector[i] = creer_vecteur_moyen_par_mot(
            data[i], text_dim, w2v_model)

    return w2v_vector
#------------------------------------------


def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features





#------------------------------------------
#IMAGES
#------------------------------------------

def afficher_image_histopixel(image, titre):
    '''
    Afficher côte à côte l'image et l'histogramme de répartiton des pixels.
    Parameters
    ----------
    image : image à afficher, obligatoire.
    Returns
    -------
    None.
    '''
    plt.figure(figsize=(40, 10))
    plt.subplot(131)
    plt.grid(False)
    plt.title(titre, fontsize=30)
    plt.imshow(image, cmap='gray')
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.subplot(132)
    plt.title('Histogramme de répartition des pixels', fontsize=30)
    hist, bins = np.histogram(np.array(image).flatten(), bins=256)
    plt.bar(range(len(hist[0:255])), hist[0:255])
    plt.xlabel('Niveau de gris', fontsize=30)
    plt.ylabel('Nombre de pixels', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.subplot(133)
    plt.title('Histogramme cumulé des pixels', fontsize=30)
    plt.hist(np.array(image).flatten(), bins=range(256), cumulative=True, 
                           histtype='stepfilled')
    plt.xlabel('Niveau de gris', fontsize=24)
    plt.ylabel('Fréquence cumulée de pixels', fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    plt.show()

#------------------------------------------

def preprocess_image(image):
    '''
    le pré-traitement suivant est appliqué sur toutes les images :
    - Correction de l'exposition (étirement d'histogramme) 
    - Correction du contraste (égalisation d'histogramme)    
    - Réduction du bruit 
    - Conversion en niveau de gris de l'image (ORB, SIFT...).
    - Réduction de dimension
    Parameters
    ----------
    image : image localisée dans un répertoire, obligatoire.
    Returns
    -------
    None
    '''
    # Variables locales
    dim = (224, 224)
    dir_images_transformed = r"C:\Users\hp\Desktop\Projet_ML\Base\Images"
    
    # Nom de l'image
    file_dir = os.path.split(image)

    # Chargement de l'image originale
    img = Image.open(image)

    # Correction de l'exposition PILS (étirement de l'histogramme)
    img = ImageOps.autocontrast(img, 1)

    # Correction du contraste (égalisation de l'histogramme)
    img = ImageOps.equalize(img)
    
    # Réduction du bruit 
    img = img.filter(ImageFilter.BoxBlur(1))
    
    # Conversion en niveau de gris de l'image
    img = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)    
    
    # Redimensionnement en 224 * 224
    img = cv.resize(np.array(img), dim, interpolation=cv.INTER_AREA)

    # Sauvegarde de l'image dans le répertoire data/Images_process
    #cv.imwrite('../DataSet/Flipkart/Images_process/' + file_dir[1], img)

    # Transforme un array en Image
    img = Image.fromarray(img) 
    
    # Sauvegarde de l'image dans le répertoire ../DataSet/Flipkart/Images_process/
    img.save(os.path.join(dir_images_transformed, file_dir[1]))    
    
    return r"C:\Users\hp\Desktop\Projet_ML\Base\Images" + file_dir[1]

#------------------------------------------

def gen_sift_features(gray_img):
    sift = cv.SIFT_create()
    # kp is the keypoints
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

#------------------------------------------

def show_sift_features(gray_img, color_img, kp):

    return plt.imshow(cv.drawKeypoints(gray_img, kp, color_img.copy()))

#------------------------------------------

def sift_features(images):
    '''
    Extraire les descripteurs et keypoints avec SIFT.
    Parameters
    ----------
    images : les images dont on veut extraire les descripteurs et keypoints
    
    Returns
    -------
    list des descripteurs et des vecteurs SIFT.
    '''
    sift_vectors = {}
    descriptor_list = []
    sift = cv.SIFT_create()
    for key, value in images.items():
        features = []
        kp, des = sift.detectAndCompute(value, None)
        descriptor_list.extend(des)
        # in case no descriptor
        des = [np.zeros((128,))] if des is None else des
        features.append(des)
        sift_vectors[key] = features
    return [descriptor_list, sift_vectors]

#------------------------------------------

def load_image_in_dict(repertoire):
    # Chargement des images pré-traitées dans un dictionnaire d'images
    images = {}
    for filename in os.listdir(repertoire):
        path = repertoire + "/" + filename
        img = cv.imread(path, 0)
        images[filename] = img
    return images

#------------------------------------------

def kmeans(k, descriptor_list):
    kmeans = KMeans(n_clusters = k, n_init=10)
    kmeans.fit(descriptor_list)
    visual_words = kmeans.cluster_centers_ 
    return visual_words

#------------------------------------------

# Takes 2 parameters. The first one is a dictionary that holds the descriptors that are separated class by class 
# And the second parameter is an array that holds the central points (visual words) of the k means clustering
# Returns a dictionary that holds the histograms for each images that are separated class by class. 
def image_class(all_bovw, centers):
    dict_feature = {}
    for key,value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = find_index(each_feature, centers)
                histogram[ind] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature

#------------------------------------------

# Find the index of the closest central point to the each sift descriptor.
# Takes 2 parameters the first one is a sift descriptor and the second one is the array of central points in k means
# Returns the index of the closest central point.
def find_index(image, center):
    count = 0
    ind = 0
    for i in range(len(center)):
        if(i == 0):
            count = distance.euclidean(image, center[i])
            #count = L1_dist(image, center[i])
        else:
            dist = distance.euclidean(image, center[i])
            #dist = L1_dist(image, center[i])
            if(dist < count):
                ind = i
                count = dist
    return ind

#------------------------------------------

def conf_mat_transform(y_true,y_pred) :
    '''
    Associe l'étiquette (la catégory produit) la plus probable 
    à chaque cluster dans le modèle KMeans
    '''
    conf_mat = metrics.confusion_matrix(y_true,y_pred)
    
    corresp = np.argmax(conf_mat, axis=0)
    print ("Correspondance des clusters : ", corresp)
    # y_pred_transform = np.apply_along_axis(correspond_fct, 1, y_pred)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    
    return labels['y_pred_transform']

#------------------------------------------

import cv2

def sift_features_batch(images, batch_size=100):
    sift = cv2.SIFT_create()
    descriptor_list = []

    items = list(images.items())
    total = len(items)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        print(f"[🔄] Traitement des images {start} à {end} sur {total}")

        batch = items[start:end]
        for key, img in batch:
            try:
                kp, des = sift.detectAndCompute(img, None)
                if des is not None:
                    descriptor_list.extend(des)
            except Exception as e:
                print(f"[⚠️] Erreur sur l'image {key} : {e}")

    return descriptor_list

#-----------------------------------------------------------------

