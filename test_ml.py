import pytest 
import tensorflow as tf 
import os 
model_path="ml/ml_models/emotion_detection.keras"
def test_modelAvalibility():
    assert os.path.exists(model_path)

def test_modelChargement():
    model =tf.keras.models.load_model(model_path)
    assert model!=  None