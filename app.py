from flask import Flask, request, jsonify, abort
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load your trained model
try:
    model = tf.keras.models.load_model('cifar10_model')
except Exception as e:
    print(f"Error loading model: {e}")
    # If the model can't be loaded, the server shouldn't run
    raise

# Define a dictionary to map numeric class labels to string labels
label_dict = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck"
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the POST request
        if 'img' not in request.files:
            abort(400, description="Image file is required")
            
        img_file = request.files['img']
        
        # Load the image using PIL
        img = Image.open(img_file).convert('RGB')
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0  # Normalize the pixel values
        img_array = img_array.reshape((1, 32, 32, 3))  # Add the batch dimension
        
        # Make the prediction using the loaded model
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        
        # Translate the numeric label to a string label
        class_label = label_dict[class_idx]
        
        # Return the prediction as JSON
        return jsonify({"predicted_class": class_label})
    
    except Exception as e:
        print(f"Error processing prediction: {e}")
        abort(500, description="Internal Server Error")

if __name__ == '__main__':
    app.run(debug=True)
