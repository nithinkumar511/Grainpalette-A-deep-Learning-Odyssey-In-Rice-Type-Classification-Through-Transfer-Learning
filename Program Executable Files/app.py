from flask import Flask, render_template, request
from tensorflow import keras
import numpy as np
import cv2

app = Flask(__name__)

# Load your trained model
model = keras.models.load_model('rice_model.keras')

# Preprocessing function (adjust as per your model input)
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # change size if needed
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "Empty filename"

    # Read and preprocess image
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    processed = preprocess_image(image)

    # Predict
    prediction = model.predict(processed)
    class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
    predicted_class = class_names[np.argmax(prediction)]
    
    return render_template('results.html', prediction=predicted_class)



if __name__ == '__main__':
    app.run(debug=True)
