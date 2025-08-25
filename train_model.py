"""Script d'entraînement du modèle de classification de sentiments pour les commentaires de films.

Ce script :
1. Charge un dataset de commentaires de films avec leurs sentiments (positif/neutre/négatif)
2. Transforme les textes en vecteurs numériques avec Word2Vec
3. Entraîne un réseau de neurones (CNN + LSTM) pour prédire le sentiment
4. Sauvegarde les modèles entraînés pour une utilisation ultérieure
"""

import os

# Configuration pour réduire les messages d'information de TensorFlow
# IMPORTANT : Ces lignes DOIVENT être avant l'import de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 3 = Afficher seulement les erreurs (pas les warnings/infos)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Désactive les optimisations oneDNN (réduit les messages)

# Imports des bibliothèques nécessaires
import numpy as np  # Pour les calculs numériques et les tableaux
import pandas as pd  # Pour manipuler les données sous forme de tableaux
import matplotlib.pyplot as plt  # Pour créer des graphiques (pas utilisé actuellement)
import tensorflow as tf  # Framework de deep learning pour créer le réseau de neurones

# Imports des modules spécifiques
from dotenv import load_dotenv  # Pour charger les variables d'environnement depuis le fichier .env
from gensim.models import Word2Vec  # Pour créer des embeddings de mots (Word2Vec)
from keras import layers  # Pour construire les couches du réseau de neurones
from spacy.tokenizer import Tokenizer  # Pour découper le texte (pas utilisé actuellement)
from spacy.lang.fr import French  # Pour le traitement du texte en français
from sklearn.model_selection import train_test_split  # Pour diviser les données en train/test
from keras.callbacks import TensorBoard  # Pour visualiser l'entraînement avec TensorBoard

def tokenize_corpus(comments):
    """Transforme une liste de commentaires en liste de mots (tokens).
    
    Cette fonction prépare le texte pour Word2Vec :
    - Met tout en minuscules pour uniformiser
    - Découpe chaque commentaire en mots individuels
    - Enlève la ponctuation (virgules, points, etc.)
    
    Args:
        comments: Liste de commentaires (textes bruts)
    
    Returns:
        Liste de listes de mots. Chaque sous-liste = les mots d'un commentaire
        Exemple: [["super", "film"], ["très", "mauvais", "scénario"]]
    """
    tokens = []  # Liste qui contiendra tous les commentaires tokenisés
    nlp = French()  # Créer un objet pour analyser le français
    
    for comment in comments:
        comment = comment.lower()  # Convertir en minuscules (Film → film)
        # nlp(comment) découpe le texte et analyse chaque mot
        # x.text récupère le mot, x.is_punct vérifie si c'est de la ponctuation
        tokens.append([x.text for x in nlp(comment) if not x.is_punct])
    return tokens

def fit_word2vec(tokens):
    """Entraîne un modèle Word2Vec pour transformer les mots en vecteurs numériques.
    
    Word2Vec apprend à représenter chaque mot comme un vecteur de nombres.
    Les mots similaires auront des vecteurs proches (ex: "excellent" et "super").
    
    Args:
        tokens: Liste de listes de mots provenant de tokenize_corpus
    
    Returns:
        Modèle Word2Vec entraîné
    """
    # Création et entraînement du modèle Word2Vec
    w2v = Word2Vec(
        sentences=tokens,  # Les commentaires tokenisés
        vector_size=W2V_SIZE,  # Taille des vecteurs (100 dimensions par défaut)
        min_count=W2V_MIN_COUNT,  # Ignore les mots qui apparaissent moins de 3 fois
        window=5,  # Contexte : regarde 5 mots avant et après pour apprendre
        workers=2  # Utilise 2 threads pour accélérer l'entraînement
    )
    
    # Sauvegarder le modèle pour pouvoir le réutiliser plus tard
    w2v.wv.save("models/w2v.wv")
    
    # Afficher des statistiques sur le vocabulaire appris
    print(f"\n📊 Informations sur le modèle Word2Vec:")
    print(f"- Taille du vocabulaire: {len(w2v.wv)} mots uniques")
    
    return w2v

def comment2vec(tokens, w2v):
    """Convertit chaque commentaire en une matrice de vecteurs Word2Vec.
    
    Transforme les mots en nombres pour que le réseau de neurones puisse les comprendre.
    Chaque commentaire devient une matrice de taille fixe (100 x 64).
    
    Args:
        tokens: Liste de listes de mots
        w2v: Modèle Word2Vec entraîné
    
    Returns:
        Array numpy 3D de forme (nb_commentaires, 100, 64)
        - Dimension 1: Chaque commentaire
        - Dimension 2: Les 100 dimensions du vecteur Word2Vec
        - Dimension 3: Les 64 mots maximum par commentaire
    """
    X = []  # Liste qui contiendra toutes les matrices
    
    # Pour chaque commentaire tokenisé
    for i, row_tokens in enumerate(tokens):
        # Créer une matrice vide (100 lignes x 64 colonnes) remplie de zéros
        # 100 lignes = dimensions Word2Vec, 64 colonnes = mots maximum
        row = np.zeros((W2V_SIZE, MAX_LENGTH))
        
        # Remplir la matrice avec les vecteurs des mots (max 64 mots)
        for j in range(min(MAX_LENGTH, len(row_tokens))):
            try:
                # Mettre le vecteur du mot j dans la colonne j
                row[:, j] = w2v.wv[row_tokens[j]]
            except KeyError:
                # Si le mot n'est pas dans le vocabulaire, on laisse des zéros
                continue
        
        X.append(row)
        
        # Afficher la progression tous les 1000 commentaires
        if i % 1000 == 0:
            print(f"{(i * 100) / len(tokens):.1f}% effectué.")
    
    # Convertir la liste en un seul tableau numpy 3D
    X = np.asarray(X)
    return X

def merge_feelings(data):
    """Prépare les étiquettes (labels) pour l'entraînement.
    
    Transforme les scores de sentiment en catégories one-hot encoded.
    Le dataset contient 3 colonnes de scores (negative, neutral, positive).
    On garde seulement le sentiment dominant pour chaque commentaire.
    
    Args:
        data: DataFrame avec colonnes 'negative', 'neutral', 'positive'
    
    Returns:
        Array one-hot encoded (3 colonnes : une seule vaut 1, les autres 0)
        Exemple: [0, 0, 1] = positif, [1, 0, 0] = négatif, [0, 1, 0] = neutre
    """
    # Extraire les 3 colonnes de sentiments
    y = data[["negative", "neutral", "positive"]]
    
    # argmax trouve l'indice de la valeur maximum (0=neg, 1=neutre, 2=pos)
    # get_dummies transforme en one-hot encoding
    y = pd.get_dummies(np.argmax(y.values, axis=1))
    return y

def create_rnn():
    """Crée l'architecture du réseau de neurones pour la classification.
    
    Architecture hybride CNN + LSTM :
    - CNN (Convolution) : Détecte des patterns locaux dans le texte
    - LSTM : Comprend les séquences et le contexte
    - Dense : Couches finales pour la classification
    
    Returns:
        Modèle Keras non compilé
    """
    return tf.keras.Sequential([
        # Couche d'entrée : attend des matrices de taille (100, 64)
        layers.Input(shape=(W2V_SIZE, MAX_LENGTH)),
        
        # Convolution 1D : détecte des patterns de 3 mots consécutifs
        # 32 filtres différents apprennent 32 patterns différents
        layers.Convolution1D(32, kernel_size=3, padding='same', activation='relu'),
        
        # MaxPooling : réduit la taille de moitié en gardant les valeurs importantes
        layers.MaxPool1D(2),
        
        # LSTM : réseau récurrent qui comprend les séquences
        # 128 neurones, return_sequences=True pour garder toute la séquence
        layers.LSTM(128, activation="tanh", return_sequences=True),
        
        # Dropout : désactive aléatoirement 10% des neurones (évite le surapprentissage)
        layers.Dropout(0.1),
        
        # Flatten : transforme la matrice en vecteur 1D pour les couches Dense
        layers.Flatten(),
        
        # Dense : couche classique avec 64 neurones
        layers.Dense(64, activation="tanh"),
        
        # Couche de sortie : 3 neurones (négatif, neutre, positif)
        # Softmax garantit que la somme des probabilités = 1
        layers.Dense(3, activation="softmax")
    ])
            
def main():
    """Fonction principale qui orchestre tout l'entraînement."""
    
    # ÉTAPE 1 : Chargement des données
    # Lit le fichier CSV et supprime les lignes avec des valeurs manquantes
    data = pd.read_csv("data/text_sentiment.csv").dropna()
    
    # Prend un échantillon aléatoire (toujours le même avec random_state=42)
    data = data.sample(NB_COMMENT, random_state=42)
    
    # ÉTAPE 2 : Préparation du texte
    # Transforme les commentaires en listes de mots
    tokens = tokenize_corpus(data["comment"])
    
    # ÉTAPE 3 : Création des embeddings Word2Vec
    # Apprend à représenter chaque mot comme un vecteur
    w2v = fit_word2vec(tokens)
    
    # ÉTAPE 4 : Vectorisation des commentaires
    # Transforme chaque commentaire en matrice de vecteurs
    X = comment2vec(tokens, w2v)
    
    # ÉTAPE 5 : Préparation des labels
    # Transforme les sentiments en format one-hot
    y = merge_feelings(data)
    
    # ÉTAPE 6 : Division train/test
    # 75% pour l'entraînement, 25% pour le test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # ÉTAPE 7 : Création et configuration du modèle
    rnn = create_rnn()
    
    # Compilation : définit comment le modèle va apprendre
    rnn.compile(
        optimizer='adam',  # Algorithme d'optimisation (ajuste les poids)
        loss="categorical_crossentropy",  # Fonction de perte pour classification multi-classes
        metrics=['categorical_accuracy']  # Métrique à surveiller (% de bonnes prédictions)
    )
    
    # Affiche un résumé de l'architecture du modèle
    rnn.summary()
    
    # ÉTAPE 8 : Entraînement
    # TensorBoard permet de visualiser l'entraînement en temps réel
    tensorboard_callback = TensorBoard("logs/rnn_movies_comments")
    
    # Entraînement du modèle
    rnn.fit(
        x=X_train, y=y_train,  # Données d'entraînement
        validation_data=(X_test, y_test),  # Données de validation
        epochs=30,  # Nombre de passes sur les données
        batch_size=32,  # Traite 32 commentaires à la fois
        callbacks=[tensorboard_callback]  # Pour le monitoring
    )
    
    # ÉTAPE 9 : Sauvegarde
    # Sauvegarde le modèle entraîné pour utilisation future
    rnn.save("models/comment_sentiment_rnn.keras")
    print("\n✅ Modèle sauvegardé dans models/comment_sentiment_rnn.keras")

# Point d'entrée du script
if __name__ == "__main__":
    # Charge les variables d'environnement depuis le fichier .env
    load_dotenv()
    
    # Configuration des hyperparamètres (avec valeurs par défaut si non définies)
    W2V_SIZE = int(os.getenv("W2V_SIZE", 100))  # Dimension des vecteurs Word2Vec
    W2V_MIN_COUNT = int(os.getenv("W2V_MIN_COUNT", 3))  # Fréquence minimale des mots
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", 64))  # Nombre max de mots par commentaire
    NB_COMMENT = int(os.getenv("NB_COMMENT", 10000))  # Nombre de commentaires à utiliser
    
    # Lance le processus d'entraînement
    main()