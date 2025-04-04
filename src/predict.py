import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import os

# === 1. Load Model ===
def load_model(model_path, num_classes=2, device='cpu'):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === 2. Define Transform ===
def get_transform():
    weights = ResNet18_Weights.DEFAULT
    return weights.transforms()

# === 3. Make Prediction ===
def predict_image(model, image_path, transform, device='cpu'):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
    
    label = "Van Gogh 🎨" if predicted_class == 1 else "Not Van Gogh ❌"
    return label, confidence
