import cv2
import numpy as np
import tensorflow as tf

# Configuration
MODEL_PATH = 'ml/ml_models/emotion_detection.keras'
CASCADE_PATH = 'ml/ml_models/haarcascade_frontalface_default.xml'
IMAGE_PATH = 'ml/images/test4.jpeg'

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load model and classifier
model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Load image
image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Only for face detection

# Detect faces (use grayscale for detection)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Process each face
for (x, y, w, h) in faces:
    # Extract face from ORIGINAL COLOR IMAGE (not grayscale!)
    face_roi = image[y:y+h, x:x+w]  # BGR format
    
    # Resize to 48x48
    face_resized = cv2.resize(face_roi, (48, 48))
    
    # Convert BGR to RGB (TensorFlow expects RGB)
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Just add batch dimension
    face_input = np.expand_dims(face_rgb, axis=0)
    
    # Prediction
    predictions = model.predict(face_input, verbose=0)
    
    # Debug: show all predictions
    print(f"All predictions: {predictions[0]}")
    
    emotion_idx = np.argmax(predictions[0])
    emotion = emotion_labels[emotion_idx]
    confidence = predictions[0][emotion_idx]
    
    print(f"Émotion: {emotion} ({confidence*100:.2f}%)")
    
    # Display on image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"{emotion}: {confidence*100:.1f}%"
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Display
cv2.imshow('Detection et Prediction', image)
cv2.waitKey(0)
cv2.destroyAllWindows()