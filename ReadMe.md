# Emotion Detection API

## 📋 Description du Projet

API d'analyse émotionnelle basée sur l'IA qui détecte automatiquement les visages dans des images et prédit les émotions correspondantes. Ce projet a été développé comme prototype pour valider la faisabilité d'un futur produit SaaS destiné aux tests produits et aux expériences UX.

## 🎯 Fonctionnalités

- **Détection automatique de visages** : Utilisation d'OpenCV avec Haar Cascade
- **Prédiction d'émotions** : Modèle CNN entraîné sur plusieurs émotions (happy, sad, angry, surprised, etc.)
- **API REST** : Endpoints FastAPI pour la prédiction et l'historique
- **Persistance des données** : Stockage des prédictions dans PostgreSQL
- **Tests automatisés** : Tests unitaires avec intégration CI/CD via GitHub Actions

## 🛠️ Technologies Utilisées

- **Deep Learning** : TensorFlow/Keras
- **Computer Vision** : OpenCV
- **API Framework** : FastAPI
- **Base de données** : PostgreSQL
- **ORM** : SQLAlchemy
- **Tests** : pytest
- **CI/CD** : GitHub Actions

## 📦 Installation

### Prérequis

- Python 3.8+
- PostgreSQL
- pip

### Configuration de l'environnement

1. **Cloner le repository**
```bash
git clone https://github.com/Khaoula1025/Facial_Emotion_Detection.git
cd emotion-detection-api
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer la base de données PostgreSQL**
```bash
# Créer la base de données
createdb emotion_detection

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos identifiants PostgreSQL
```

5. **Télécharger le fichier Haar Cascade**
```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

## 🚀 Utilisation

### 1. Entraînement du modèle CNN

```bash
# Exécuter le notebook Jupyter
jupyter notebook ml/notebooks/train_cnn_model.ipynb
```

Le notebook effectue :
- Chargement et prétraitement des données
- Augmentation des images
- Entraînement du modèle CNN
- Sauvegarde du modèle (`.h5` ou `.keras`)
- Visualisation des métriques

### 2. Test de détection et prédiction standalone

```bash
python detect_and_predict.py 
```

### 3. Lancement de l'API

```bash
uvicorn main:app --reload 
```

L'API sera accessible à : `http://localhost:8000`

Documentation interactive : `http://localhost:8000/docs`

## 📡 Endpoints API

### POST `/predict_emotion`

Prédit l'émotion d'un visage dans une image.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict_emotion" 
```

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.92,
  "id": 1,
  "created_at": "2025-11-14T10:30:00"
}
```

### GET `/history`

Récupère l'historique des prédictions.

**Request:**
```bash
curl -X GET "http://localhost:8000/history"
```

**Response:**
```json
[
  {
    "id": 1,
    "emotion": "happy",
    "confidence": 0.92,
    "created_at": "2025-11-14T10:30:00"
  },
  {
    "id": 2,
    "emotion": "sad",
    "confidence": 0.87,
    "created_at": "2025-11-14T10:35:00"
  }
]
```

## 🗂️ Structure du Projet

```
Facial_emotion-detection/
│
├── app/
│   ├── db/
|   |    ├── database.py
|   |    └── models.py
|   |
|   ├──schemas/
|   |      └── prediction.py
|   ├──utils/
|   |      └── predictionScript.py
|   └── main.py
│
├── ml/
│   ├──images
|   ├──ml_models/
|   |     ├── emotion_detection.keras
|   |     └── haarcascade_frontface_default.xml
|   ├──notebooks/
|   |       └── train_cnn_model.ipynb
|   └── detect_and_predict.py 
|       
├── data/
│   └── test/
|   └── train/ 
│
├── test_ml.py             
├── test_api.py               
│
├── .github/
│   └── workflows/
│          └── python_test.yml              
|
├── requirements.txt              
├── .gitignore
└── README.md
```

## 🧪 Tests

### Exécuter les tests unitaires

```bash
pytest tests/ -v
```

### Tests couverts

- ✅ Sauvegarde et rechargement du modèle
- ✅ Format des prédictions
- ✅ Endpoints API
- ✅ Détection de visages
- ✅ Connexion à la base de données

### Tests automatisés (CI/CD)

Les tests sont automatiquement exécutés via GitHub Actions à chaque push/PR.

## 🗄️ Schéma de Base de Données

**Table: predictions**

| Colonne      | Type      | Description                    |
|--------------|-----------|--------------------------------|
| id           | SERIAL    | Clé primaire auto-incrémentée  |
| emotion      | VARCHAR   | Émotion prédite                |
| confidence   | FLOAT     | Score de confiance (0-1)       |
| created_at   | TIMESTAMP | Date et heure de la prédiction |

## 📊 Modèle CNN

### Architecture

```
Input (48x48x1 ou 48x48x3)
    ↓
Conv2D(32) + ReLU + MaxPooling
    ↓
Conv2D(64) + ReLU + MaxPooling
    ↓
Conv2D(128) + ReLU + MaxPooling
    ↓
Flatten
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(num_classes) + Softmax
```

### Paramètres d'entraînement

- **Optimiseur** : Adam
- **Fonction de perte** : Categorical Crossentropy
- **Métriques** : Accuracy
- **Augmentation** : Rotation, Zoom, Flip horizontal

## 🔧 Configuration

### Variables d'environnement (.env)

```env
DATABASE_URL=postgresql://user:password@localhost:5432/emotion_detection
MODEL_PATH=models/emotion_model.h5
CASCADE_PATH=haarcascade_frontalface_default.xml
```

## 📈 Performances

- **Accuracy** : ~85-90% sur le dataset de validation
- **Temps de prédiction** : <100ms par image
- **Détection de visages** : Fonctionne sur des visages frontaux bien éclairés

## 🚧 Limitations et Améliorations Futures

### Limitations actuelles
- Détection limitée aux visages frontaux
- Performance variable selon l'éclairage
- Dataset limité à certaines émotions de base

### Améliorations prévues
- [ ] Utilisation de modèles de détection plus robustes (MTCNN, RetinaFace)
- [ ] Fine-tuning avec des modèles pré-entraînés (VGG, ResNet)
- [ ] Support de plusieurs visages dans une image
- [ ] Ajout d'une interface web
- [ ] Système d'authentification
- [ ] Export des données en CSV/Excel
- [ ] Monitoring et logging avancés

## 👥 Contributeurs

- **Votre Nom** - Développeur IA

## 📝 Licence

Ce projet est sous licence MIT.

## 📞 Contact

Pour toute question ou suggestion :
- Email: votre.email@example.com
- GitHub: [@votre-username](https://github.com/votre-username)

## 🙏 Remerciements

- OpenCV pour les outils de vision par ordinateur
- TensorFlow/Keras pour le framework de deep learning
- FastAPI pour le framework web moderne et performant
- La communauté open-source pour les datasets d'émotions

---

**Note** : Ce projet est un prototype développé à des fins éducatives et de validation de concept. Pour une utilisation en production, des améliorations de sécurité, de performance et de robustesse sont nécessaires.