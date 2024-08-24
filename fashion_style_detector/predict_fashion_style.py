import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F

# Model Definition
class AdvancedFashionStyleClassifier(nn.Module):
    def __init__(self, num_classes=15):  # Ensure num_classes=15
        super(AdvancedFashionStyleClassifier, self).__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.base_model(x)
        return F.softmax(x, dim=1)

# Descriptions for Each Fashion Style
fashion_descriptions = {
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

# Prediction Function
def predict(model, image_path, classes, descriptions):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = default_loader(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities, predicted_idx = torch.max(outputs, 1)
        predicted_class = classes[predicted_idx.item()]
        probability_percentage = probabilities.item() * 100
        description = descriptions[predicted_class]
        return predicted_class, probability_percentage, description

# Main Code
if __name__ == "__main__":
    # Update these paths
    model_path = 'fashion_style_detector/fashion_style_classifier_v7.pth'
    test_image_path = 'test.jpeg'

    # Load the classes (ensure this list has 15 classes)
    classes = ['athleisure fashion', 'bohemian fashion', 'business fashion', 'casual fashion', 
               'chic fashion', 'edgy fashion', 'glam fashion', 'gothic fashion', 
               'grunge fashion', 'hipster fashion', 'minimalist fashion', 'preppy fashion', 
               'punk fashion', 'streetwear fashion', 'vintage fashion']

    print("Loading model...")
    model = AdvancedFashionStyleClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path))
    print("Model loaded successfully!")

    print("Predicting fashion style...")
    predicted_class, probability_percentage, description = predict(model, test_image_path, classes, fashion_descriptions)
    print(f"Predicted class: {predicted_class} ({probability_percentage:.2f}%)")
    print(f"Description: {description}")
