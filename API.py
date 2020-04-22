import base64
import numpy as np
import io as io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder="templates")

def get_model():
    global model
    model = load_model('Cheque.h5')
    print("Model Loaded")

def preprocess_image(image, target_size):
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    return image

print("** Loading Keras Model **")
get_model()

@app.route('/')
def home():
    return render_template('Predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image =  preprocess_image(image, target_size=(150, 150))
    
    prediction = model.predict(processed_image).tolist()
    
    response = {
        'prediction' : {
            'Result' : prediction[0][0],
        }
    }
    return jsonify(response)