from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = tf.keras.models.load_model('model/crop_disease_model.h5')

def predict_crop_disease(img_path):
    img = image.load_img(img_path, target_size=(128,128))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_names = ['cotton', 'rice', 'wheat']
    return class_names[np.argmax(predictions)]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            result = predict_crop_disease(filepath)
            return render_template('index.html', prediction=result)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
