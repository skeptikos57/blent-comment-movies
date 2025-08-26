# 🎬 Analyse de Sentiments - Commentaires de Films

Un système de deep learning pour analyser automatiquement le sentiment (positif, neutre, négatif) des commentaires de films en français.

## 📋 Description

Ce projet utilise :
- **Word2Vec** pour transformer les mots en vecteurs numériques
- **CNN + LSTM** pour comprendre le contexte et classifier les sentiments
- **TensorFlow/Keras** pour le deep learning
- **Spacy** pour le traitement du texte en français

## 🏗️ Architecture

```
Commentaire texte
    ↓
Tokenisation (Spacy)
    ↓
Vectorisation (Word2Vec)
    ↓
CNN (détection de patterns)
    ↓
LSTM (compréhension du contexte)
    ↓
Classification (Positif/Neutre/Négatif)
```

## 📁 Structure du Projet

```
commentaires-film/
├── data/                      # Données
│   └── text_sentiment.csv    # Dataset des commentaires
├── models/                    # Modèles sauvegardés
│   ├── w2v.wv                # Modèle Word2Vec
│   └── comment_sentiment_rnn.keras  # Modèle RNN
├── logs/                      # Logs TensorBoard
├── train_model.py            # Script d'entraînement
├── predict.py                # Script de prédiction
├── test_gpu.py              # Test de configuration GPU
├── requirements.txt         # Dépendances Python
└── .env                     # Configuration

```

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone <votre-repo>
cd commentaires-film
```

### 2. Créer un environnement virtuel (recommandé)

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger le modèle de langue française

```bash
python -m spacy download fr_core_news_sm
```

### 5. Configuration (optionnel)

Modifier le fichier `.env` pour ajuster les paramètres :

```env
W2V_SIZE=100        # Dimension des vecteurs Word2Vec
W2V_MIN_COUNT=3     # Fréquence minimale des mots
MAX_LENGTH=64       # Nombre max de mots par commentaire
NB_COMMENT=10000    # Nombre de commentaires pour l'entraînement
```

## 💻 Utilisation

### Entraîner le modèle

```bash
python train_model.py
```

Ce script va :
1. Charger les données depuis `data/text_sentiment.csv`
2. Entraîner un modèle Word2Vec pour créer des embeddings
3. Entraîner un réseau CNN+LSTM pour la classification
4. Sauvegarder les modèles dans le dossier `models/`

**Temps d'entraînement estimé** : 
- CPU : ~30-45 minutes pour 10,000 commentaires
- GPU : ~5-10 minutes pour 10,000 commentaires

### Faire une prédiction

```bash
python predict.py "Ce film était vraiment excellent, je le recommande !"
```

**Exemple de sortie** :
```
✅ Modèle RNN chargé depuis models/comment_sentiment_rnn.keras
✅ Modèle Word2Vec chargé depuis models/w2v.wv

📊 Analyse du commentaire:
Commentaire: "Ce film était vraiment excellent, je le recommande !"
Sentiment prédit: Positif (confiance: 92.3%)
Probabilités détaillées:
  - Négatif: 5.2%
  - Neutre: 2.5%
  - Positif: 92.3%
```

### Tester la configuration GPU (optionnel)

```bash
python test_gpu.py
```

## 📊 Dataset

Le fichier `data/text_sentiment.csv` doit contenir les colonnes suivantes :
- `comment` : Le texte du commentaire
- `negative` : Score de négativité (0-1)
- `neutral` : Score de neutralité (0-1)
- `positive` : Score de positivité (0-1)

## 🔧 Personnalisation

### Modifier l'architecture du réseau

Dans `train_model.py`, fonction `create_rnn()` :

```python
def create_rnn():
    return tf.keras.Sequential([
        layers.Input(shape=(W2V_SIZE, MAX_LENGTH)),
        layers.Convolution1D(32, kernel_size=3, ...),  # Ajuster les filtres
        layers.LSTM(128, ...),  # Ajuster les neurones LSTM
        layers.Dense(64, ...),  # Ajuster la couche dense
        layers.Dense(3, activation="softmax")
    ])
```

### Choisir l'optimiseur

Dans `train_model.py`, lors de la compilation du modèle :

```python
rnn.compile(
    optimizer='adam',  # Optimiseur par défaut (recommandé)
    loss="categorical_crossentropy",
    metrics=['categorical_accuracy']
)
```

**Optimiseurs disponibles et cas d'usage :**

| Optimiseur | Utilisation | Avantages | Cas d'usage idéal |
|------------|-------------|-----------|-------------------|
| **adam** (défaut) | 90% des cas | Adaptatif, converge rapidement, robuste | NLP, vision, réseaux profonds, analyse de sentiments |
| **sgd** | Modèles simples | Simple, généralise bien | Fine-tuning, convergence finale précise |
| **rmsprop** | RNN, LSTM | Bon pour gradients instables | Séries temporelles, problèmes avec gradients variables |
| **adagrad** | Données éparses | Adapte le learning rate par paramètre | Embeddings de mots, texte avec vocabulaire large |
| **adamax** | Gradients bruités | Plus stable qu'Adam | Modèles avec beaucoup de bruit |
| **nadam** | Convergence rapide | Adam + momentum Nesterov | Alternative à Adam pour convergence plus rapide |
| **adadelta** | Sans hyperparamètres | Pas besoin de learning rate | Prototypage rapide |

**Configuration avancée (optionnel) :**

```python
from keras.optimizers import Adam

# Personnaliser l'optimiseur
optimizer = Adam(
    learning_rate=0.001,  # Taux d'apprentissage
    beta_1=0.9,          # Momentum exponentiel pour le gradient
    beta_2=0.999         # Momentum exponentiel pour le carré du gradient
)

rnn.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=['categorical_accuracy']
)
```

**Recommandations pour votre cas (analyse de sentiments) :**
- ✅ **Adam** : Excellent choix par défaut, fonctionne très bien pour l'analyse de sentiments
- **Alternative 1** : **Nadam** si la convergence est lente
- **Alternative 2** : **RMSprop** si vous observez des instabilités pendant l'entraînement
- **Alternative 3** : **SGD avec momentum** pour un fine-tuning final après Adam

💡 **Conseil** : Commencez toujours avec Adam. Changez d'optimiseur uniquement si vous rencontrez des problèmes spécifiques de convergence ou de performance.

### Choisir la fonction de perte

Dans `train_model.py`, lors de la compilation du modèle :

```python
rnn.compile(
    optimizer='adam',
    loss="categorical_crossentropy",  # Fonction de perte par défaut pour multi-classes
    metrics=['categorical_accuracy']
)
```

**Fonctions de perte disponibles et cas d'usage :**

#### Pour la classification

| Fonction de perte | Cas d'usage | Activation finale | Exemple d'utilisation |
|------------------|-------------|-------------------|----------------------|
| **categorical_crossentropy** (défaut) | Classification multi-classes avec one-hot encoding | softmax | Sentiments (positif/neutre/négatif), catégories de films |
| **sparse_categorical_crossentropy** | Classification multi-classes avec labels entiers | softmax | Même cas mais labels [0,1,2] au lieu de [[1,0,0],[0,1,0],[0,0,1]] |
| **binary_crossentropy** | Classification binaire ou multi-label | sigmoid | Bon/mauvais film, tags multiples (action ET comédie) |
| **focal_crossentropy** | Classification avec classes déséquilibrées | softmax/sigmoid | Dataset avec 90% positifs, 10% négatifs |

#### Pour la régression (si vous prédisez des scores)

| Fonction de perte | Cas d'usage | Activation finale | Exemple d'utilisation |
|------------------|-------------|-------------------|----------------------|
| **mean_squared_error** (MSE) | Prédiction de valeurs continues | linear/None | Note de 0 à 10, score de sentiment 0-100% |
| **mean_absolute_error** (MAE) | Régression robuste aux outliers | linear/None | Scores avec données bruitées |
| **huber** | Hybride MSE/MAE | linear/None | Robuste mais différentiable |
| **mean_squared_logarithmic_error** | Valeurs avec large échelle | linear/None | Prédictions où l'erreur relative compte plus |

#### Pour des cas spécialisés

| Fonction de perte | Cas d'usage | Activation finale | Exemple d'utilisation |
|------------------|-------------|-------------------|----------------------|
| **cosine_similarity** | Similarité entre vecteurs | None | Comparaison d'embeddings de commentaires |
| **kullback_leibler_divergence** | Divergence entre distributions | softmax | Modèles génératifs, autoencoders |
| **poisson** | Comptage d'événements | exponential | Nombre de likes, vues |

**Configuration pour votre cas (analyse de sentiments 3 classes) :**

```python
# Option 1 : One-hot encoding (votre configuration actuelle) ✅
# Labels : [[1,0,0], [0,1,0], [0,0,1]]
rnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # Parfait pour votre cas
    metrics=['categorical_accuracy']
)

# Option 2 : Labels entiers (plus économe en mémoire)
# Labels : [0, 1, 2]
rnn.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy']
)

# Option 3 : Si vous aviez des classes déséquilibrées
from tensorflow.keras.losses import CategoricalFocalCrossentropy
rnn.compile(
    optimizer='adam',
    loss=CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0),
    metrics=['categorical_accuracy']
)
```

**Fonction de perte personnalisée (avancé) :**

```python
import tensorflow as tf

def custom_weighted_loss(y_true, y_pred):
    """Perte personnalisée avec poids différents par classe"""
    # Poids : négatif=2, neutre=1, positif=1.5
    weights = tf.constant([2.0, 1.0, 1.5])
    
    # Categorical crossentropy pondérée
    cce = tf.keras.losses.CategoricalCrossentropy()
    base_loss = cce(y_true, y_pred)
    
    # Appliquer les poids selon la vraie classe
    class_weights = tf.reduce_sum(y_true * weights, axis=-1)
    weighted_loss = base_loss * class_weights
    
    return tf.reduce_mean(weighted_loss)

rnn.compile(
    optimizer='adam',
    loss=custom_weighted_loss,
    metrics=['categorical_accuracy']
)
```

**Recommandations pour votre projet :**
- ✅ **categorical_crossentropy** : Excellent choix pour 3 classes équilibrées
- **Alternative 1** : **sparse_categorical_crossentropy** si vous voulez économiser de la mémoire
- **Alternative 2** : **focal_crossentropy** si vos classes sont très déséquilibrées
- **Alternative 3** : Fonction personnalisée si certains sentiments sont plus importants à détecter

💡 **Conseil** : La fonction de perte doit correspondre à votre problème ET à votre activation finale. Pour la classification multi-classes, utilisez toujours softmax + categorical_crossentropy (ou sa variante sparse).

### Ajuster les hyperparamètres d'entraînement

Dans `train_model.py`, fonction `main()` :

```python
rnn.fit(
    epochs=30,      # Nombre d'époques
    batch_size=32,  # Taille des batchs
    ...
)
```

## 📈 Monitoring avec TensorBoard

Pour visualiser l'entraînement en temps réel :

```bash
tensorboard --logdir=logs/rnn_movies_comments
```

Puis ouvrir : http://localhost:6006

## 🐛 Résolution de problèmes

### Erreur CUDA/GPU

Si TensorFlow ne détecte pas votre GPU :

1. Vérifier les drivers NVIDIA :
```bash
nvidia-smi
```

2. Installer TensorFlow avec support CUDA :
```bash
pip install tensorflow[and-cuda]
```

### Erreur de mémoire

Si vous manquez de mémoire, réduire :
- `NB_COMMENT` dans `.env`
- `batch_size` dans `train_model.py`

### Mot hors vocabulaire

Si un mot n'est pas reconnu lors de la prédiction, il est ignoré (remplacé par des zéros). Pour améliorer la couverture :
- Augmenter `NB_COMMENT` pour l'entraînement
- Diminuer `W2V_MIN_COUNT` dans `.env`

## 📝 Dépendances principales

- `tensorflow[and-cuda]` : Framework de deep learning
- `gensim` : Pour Word2Vec
- `spacy` : Traitement du texte français
- `pandas` : Manipulation des données
- `numpy` : Calculs numériques
- `scikit-learn` : Division train/test
- `python-dotenv` : Gestion de la configuration

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation

## 📄 Licence

MIT

## 👤 Auteur

Raphaël Léo

---

**Note** : Ce projet est à but éducatif pour apprendre le deep learning et le NLP.