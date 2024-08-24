from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os

# Define the model architecture (AdvancedFashionStyleClassifier)
class AdvancedFashionStyleClassifier(nn.Module):
    def __init__(self, num_classes=15):  # Assuming 15 fashion style classes
        super(AdvancedFashionStyleClassifier, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.base_model(x)
        return F.softmax(x, dim=1)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# Load the model architecture and weights
model = AdvancedFashionStyleClassifier(num_classes=15)
model.load_state_dict(torch.load('fashion_style_detector/fashion_style_classifier_v7.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Prediction function
def predict_fashion_style(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities, predicted_idx = torch.max(outputs, 1)
        predicted_class = classes[predicted_idx.item()]
        probability_percentage = probabilities.item() * 100
        return predicted_class, probability_percentage

# Define the classes
classes = ['athleisure fashion', 'bohemian fashion', 'business fashion', 'casual fashion', 
           'chic fashion', 'edgy fashion', 'glam fashion', 'gothic fashion', 
           'grunge fashion', 'hipster fashion', 'minimalist fashion', 'preppy fashion', 
           'punk fashion', 'streetwear fashion', 'vintage fashion']

# Define descriptions for each class
descriptions = {
    'athleisure fashion': "Athleisure fashion blends athletic wear with leisurewear, designed to be both functional for exercise and stylish for everyday wear.",
    'bohemian fashion': "Bohemian fashion is characterized by free-spirited and artistic clothing, often featuring bold patterns, natural fabrics, and a relaxed, unconventional style.",
    'business fashion': "Business fashion emphasizes professional attire, typically featuring tailored suits, blouses, and conservative accessories, suitable for corporate environments.",
    'casual fashion': "Casual fashion is all about comfort and simplicity, with relaxed clothing like jeans, t-shirts, and sneakers for everyday wear.",
    'chic fashion': "Chic fashion is elegant and stylish, often featuring modern, sophisticated clothing with clean lines, neutral colors, and an overall polished appearance.",
    'edgy fashion': "Edgy fashion is bold and daring, often incorporating dark colors, leather, and unconventional clothing items for a rebellious, avant-garde look.",
    'glam fashion': "Glam fashion is all about glitz and glamour, featuring luxurious fabrics, sparkling accessories, and bold makeup for a high-fashion, eye-catching appearance.",
    'gothic fashion': "Gothic fashion embraces dark, mysterious, and romantic elements, often featuring black clothing, lace, and silver accessories.",
    'grunge fashion': "Grunge fashion is inspired by the 90s music scene, characterized by oversized flannel shirts, ripped jeans, and a generally unkempt, effortless look.",
    'hipster fashion': "Hipster fashion is alternative and non-mainstream, often featuring vintage or thrifted clothing, ironic accessories, and an emphasis on individuality.",
    'minimalist fashion': "Minimalist fashion focuses on simplicity and functionality, with a neutral color palette, clean lines, and a focus on high-quality, timeless pieces.",
    'preppy fashion': "Preppy fashion is classic and polished, often featuring clean-cut clothing like polo shirts, blazers, and loafers, with a nod to Ivy League style.",
    'punk fashion': "Punk fashion is rebellious and DIY-inspired, featuring bold patterns, leather jackets, band t-shirts, and an overall anti-establishment vibe.",
    'streetwear fashion': "Streetwear fashion is casual and trendy, often inspired by urban culture, featuring oversized clothing, sneakers, and bold logos.",
    'vintage fashion': "Vintage fashion is about embracing styles from past decades, often featuring retro clothing, accessories, and an appreciation for timeless fashion."
    
}

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file).convert('RGB')
    
    predicted_class, probability_percentage = predict_fashion_style(image)
    description = descriptions.get(predicted_class, 'No description available.')

    return jsonify({
        'predicted_class': predicted_class,
        'probability': probability_percentage,
        'description': description
    })

# Route to serve the index.html file
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'fashion_style_detector/index.html')

if __name__ == '__main__':
    app.run(debug=True)

