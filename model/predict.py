from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('saved_models/crop_disease_model.h5')

def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)

    confidence = np.max(preds) * 100
    class_idx = np.argmax(preds, axis=1)[0]
    
    class_labels = ['BrownSpot', 'LeafBlast', 'Healthy', 'Rust', 'BacterialBlight']  # Your actual classes

    if confidence < 60:
        return f"Prediction uncertain. Please upload a clearer image. (Confidence: {confidence:.2f}%)"
    else:
        return f"{class_labels[class_idx]} (Confidence: {confidence:.2f}%)"
