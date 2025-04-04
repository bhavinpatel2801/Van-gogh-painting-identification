from torchvision import transforms
from PIL import Image
import random

import sys
sys.path.append('../src')

def patchify_image(img, patch_size=224):
    width, height = img.size
    patches = []

    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            right = min(left + patch_size, width)
            bottom = min(top + patch_size, height)
            patch = img.crop((left, top, right, bottom))
            
            if patch.size == (patch_size, patch_size):
                patches.append(patch)
    
    return patches

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),              # Ensure patch is 224x224
    transforms.ToTensor(),                      # Convert to tensor
    transforms.Normalize(                       # Normalize RGB channels
        mean=[0.5, 0.5, 0.5],                    # Center pixels around 0
        std=[0.5, 0.5, 0.5]                      # Scale down values
    )
])



def extract_patches_and_labels(images, label, transform, max_patches=20):
    patch_tensors = []
    patch_labels = []

    for img in images:
        patches = patchify_image(img, 224)
        if len(patches) > max_patches:
            patches = random.sample(patches, max_patches)

        for patch in patches:
            tensor = transform(patch)
            patch_tensors.append(tensor)
            patch_labels.append(label)
    
    return patch_tensors, patch_labels


import torch
#from patchify_image import patchify_image  # already defined in preprocessing_patches.py
from torchvision import transforms

# ⚠️ Import transform only once
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image_patches(image, model, device='cpu'):
    model.eval()
    patches = patchify_image(image, patch_size=224)
    tensors = [transform(p).unsqueeze(0) for p in patches]
    inputs = torch.cat(tensors).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    labels = preds.tolist()
    votes = sum(labels)
    total = len(labels)
    confidence = votes / total
    label = 1 if confidence > 0.5 else 0

    return label, confidence
