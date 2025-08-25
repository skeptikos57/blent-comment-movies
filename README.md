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