from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO


model_path = '/home/aayu/Documents/Bharat Intern/teacnook/new/model/psi.h5'
model = load_model(model_path)

app = Flask(__name__)

def process_prediction(prediction):
   
    predicted_class_index = np.argmax(prediction)
    
    class_labels = ["Class 0", "Class 1", "Class 2", ...]
    
    if 0 <= predicted_class_index < len(class_labels):
        predicted_class_label = class_labels[predicted_class_index]
    else:
        predicted_class_label = "Unknown Class"

    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label

@app.route('/')
def index():
    return render_template('indexx.html')

@app.route('/predict', methods=['POST'])
def predict():

    uploaded_file = request.files['image']
    if uploaded_file.filename != '':
        # Load and preprocess the image
        img = Image.open(uploaded_file)
        img = img.resize((32, 32))  # Resize to match model input size
        img = np.array(img) / 255.0  # Normalize the image

        # Make a prediction using the loaded model
        prediction = model.predict(np.expand_dims(img, axis=0))

        # Process prediction result (e.g., convert to class label)
        class_label = process_prediction(prediction)

        return f'Predicted class: {class_label}'
    else:
        return 'No file uploaded'
    

    # Handle image upload and model prediction here
    return 'Prediction result goes here'

if __name__ == '__main__':
    app.run(debug=True,port=8080)





