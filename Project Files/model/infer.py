from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np, os

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'eye_disease_model.h5')

_model_cache = None

def load_trained_model():
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Train the model first.")
        _model_cache = load_model(MODEL_PATH)
        # Stash class names for convenience (assuming they were saved as JSON)
        labels_path = os.path.join(os.path.dirname(__file__), 'labels.txt')
        if os.path.isfile(labels_path):
            with open(labels_path) as f:
                _model_cache.class_names = [ln.strip() for ln in f.readlines()]
        else:
            _model_cache.class_names = [str(i) for i in range(_model_cache.output_shape[-1])]
    return _model_cache

def preprocess(path):
    img = image.load_img(path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    return np.expand_dims(x, axis=0)

def predict_image(model, path):
    x = preprocess(path)
    preds = model.predict(x)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = model.class_names[idx]
    return label, confidence