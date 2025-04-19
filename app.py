from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__)

#
model_path = r'G:\head_count\head_count challenge\Model\model_final.pth'
model = torch.load(model_path)
model.eval()

# Define a transformation for the incoming image (adjust based on your model)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Adjust this size as per your model input size
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is part of the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Load the image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_count = output.item()  
        
        # Return prediction as a JSON response
        return jsonify({'predicted_count': predicted_count})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
