import cv2
import numpy as np
import tensorflow as tf

# Configuration
MODEL_PATH = '././ml/ml_models/emotion_detection.keras'
CASCADE_PATH = '././ml/ml_models/haarcascade_frontalface_default.xml'

# Étiquettes d’émotions (doivent correspondre à ton modèle)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Charger le modèle et le détecteur de visage (une seule fois)
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def predict_emotions(image_bytes: bytes):
    """
    Prédit l’émotion à partir d’une image envoyée sous forme de bytes.
    Retourne un dictionnaire avec l’émotion et la confiance.
    """

    # Convertir les bytes en image OpenCV
    npimg = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Image invalide ou non lisible"}

    # Convertir en niveaux de gris pour la détection du visage
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return {"error": "Aucun visage détecté"}

    # Utiliser le premier visage détecté
    (x, y, w, h) = faces[0]
    face_roi = image[y:y+h, x:x+w]

    # Prétraitement de l'image du visage
    face_resized = cv2.resize(face_roi, (48, 48))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

    # Pas de normalisation si ton modèle a une couche Rescaling
    face_input = np.expand_dims(face_rgb, axis=0)

    # Prédiction
    preds = model.predict(face_input, verbose=0)[0]
    emotion_idx = np.argmax(preds)
    emotion = emotion_labels[emotion_idx]
    confidence = float(preds[emotion_idx])
    # Retourner les résultats
    return {
        "emotion": emotion,
        "confidence": confidence,
    }
