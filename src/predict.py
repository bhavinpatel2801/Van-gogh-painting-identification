import sys
sys.path.append('../src')  # Add custom source path to Python module search path

# Standard libraries
import os
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# Custom patch function
from preprocessing_patches import patchify_image
#  Load Model
def load_model(model_path, num_classes=2, device='cpu'):
    # Extract the filename from the full model path and convert it to lowercase
    filename = os.path.basename(model_path).lower()

    # === Automatically detect the model architecture from the filename ===
    
    if "efficientnet" in filename:
        # Load EfficientNet-B0 with pretrained weights
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)

        # Replace the final classification layer to match the number of classes
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

    elif "vit" in filename:
        # Load Vision Transformer (ViT-B/16) with pretrained weights
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)

        # Replace the classification head
        model.heads = torch.nn.Sequential(torch.nn.Linear(model.heads[0].in_features, num_classes))

    elif "basiccnn" in filename:
        # Load a custom-defined basic CNN model from trainer.py
        from trainer import BasicCNN
        model = BasicCNN()

    elif "resnet" in filename:
        # Load ResNet18 with pretrained weights
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)

        # Replace the final fully-connected layer
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    else:
        # Raise an error if the architecture cannot be detected
        raise ValueError(f"Unknown architecture in file: {filename}")

    # === Load the trained weights into the model ===
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Move model to specified device (CPU/GPU)
    model.to(device)

    # Set model to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()

    # Return the fully configured and ready model
    return model

#  Define Transform
def get_transform(arch="resnet"):
    # This function returns the appropriate preprocessing pipeline (resize, normalize, etc.)
    #    based on the model architecture. These are aligned with the training config of the pretrained weights.

    # Preprocessing for ResNet18
    if arch == "resnet":
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.DEFAULT  # Load default pretrained weight configuration
        return weights.transforms()         # Return the corresponding transforms (resize, center crop, normalize)

    # Preprocessing for EfficientNet-B0
    elif arch == "efficientnet":
        from torchvision.models import EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT
        return weights.transforms()

    # Preprocessing for Vision Transformer (ViT-B/16)
    elif arch == "vit":
        from torchvision.models import ViT_B_16_Weights
        weights = ViT_B_16_Weights.DEFAULT
        return weights.transforms()

    # Handle unknown model types
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

# Make Prediction
def predict_image(model, image_path, transform, device='cpu'):
    # Load the image from disk and convert it to RGB format
    image = Image.open(image_path).convert('RGB')

    # Apply preprocessing transform (resize, normalize, etc.)
    # unsqueeze(0) adds a batch dimension → shape becomes (1, C, H, W)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Disable gradient tracking to save memory during inference
    with torch.no_grad():
        outputs = model(image_tensor)              # Forward pass through the model
        probs = torch.softmax(outputs, dim=1)      # Convert logits to probabilities
        predicted_class = torch.argmax(probs, dim=1).item()  # Class index with highest probability
        confidence = probs[0][predicted_class].item()        # Extract confidence of that class

    # Map predicted class index to human-readable label
    label = "Van Gogh 🎨" if predicted_class == 1 else "Not Van Gogh ❌"

    # Return the label and its confidence score
    return label, confidence

def predict_image_soft(model, image, transform, device='cpu'):
    # Apply transform (resize, normalize, etc.) and add batch dimension (1, C, H, W)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Disable gradient computation for faster inference
    with torch.no_grad():
        outputs = model(image_tensor)                  # Forward pass
        probs = torch.softmax(outputs, dim=1)          # Convert logits to probabilities

    # Return the probability of class 1 (Van Gogh)
    return probs[0, 1].item()

def predict_patches_soft(model, image, patch_transform, device='cpu'):
    # Split the image into non-overlapping 224x224 patches
    patches = patchify_image(image, patch_size=224)

    # Apply transform and add batch dimension to each patch
    tensors = [patch_transform(p).unsqueeze(0) for p in patches]

    # Concatenate all patch tensors into a single batch
    batch = torch.cat(tensors).to(device)

    # Inference without gradient tracking
    with torch.no_grad():
        outputs = model(batch)                          # Forward pass on all patches
        probs = torch.softmax(outputs, dim=1)[:, 1]     # Get class-1 probability for each patch

    # Return the average confidence across all patches
    return probs.mean().item()

def load_best_models(model_dir="../models"):
    # === Find best full-image model ===
    best_full_model_path = None
    for fname in os.listdir(model_dir):  # Loop through files in model directory
        if fname.endswith("_best_full.pth"):  # Look for full-image model file
            best_full_model_path = os.path.join(model_dir, fname)  # Build full path
            break  # Stop once found

    # === Find best patch-based model ===
    best_patch_model_path = None
    for fname in os.listdir(model_dir):  # Loop through files again
        if fname.endswith("_best_patch.pth"):  # Look for patch-based model file
            best_patch_model_path = os.path.join(model_dir, fname)  # Build full path
            break  # Stop once found

    # === Load both models using load_model function ===
    model_full = load_model(best_full_model_path)    # Load full-image model
    model_patch = load_model(best_patch_model_path)  # Load patch-based model

    # === Return both models and their paths for reference ===
    return model_full, best_full_model_path, model_patch, best_patch_model_path

def predict_from_ensemble(image_path, model_full, best_full_model_path, model_patch, best_patch_model_path):
     # === 1. Handle both string path and in-memory image ===
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")  # if already PIL.Image from Streamlit or upload

    # === 2. Organize models and their respective paths using dictionaries ===
    models = {"full": model_full, "patch": model_patch}
    model_paths = {"full": best_full_model_path, "patch": best_patch_model_path}

    probs = []         # To store predicted probabilities from each model
    model_names = []   # To store human-readable model names for diagnostics

    # === 3. Iterate over both models: full-image and patch-based ===
    for kind in ["full", "patch"]:
        model = models[kind]                                  # Select model
        fname = os.path.basename(model_paths[kind]).lower()   # Extract filename to detect architecture

        # === Detect architecture based on filename content ===
        if "efficientnet" in fname:
            arch = "EfficientNet"
        elif "vit" in fname:
            arch = "ViT"
        elif "resnet" in fname:
            arch = "ResNet"
        elif "basiccnn" in fname:
            arch = "BasicCNN"  # Fallback if no specific model is matched
        else:
            arch = "Unknown"   # Default label if detection fails

        # === Track model name for logging ===
        model_names.append(f"{arch}_{kind.capitalize()}")

        # === Load appropriate preprocessing transform for this model ===
        transform = get_transform(arch.lower())

        # === Make prediction depending on model type ===
        if kind == "full":
            probs.append(predict_image_soft(model, image, transform))   # Soft probability from full image
        else:
            probs.append(predict_patches_soft(model, image, transform)) # Average probability from patches

    # === 4. Print individual model predictions for diagnostic purposes ===
    for name, prob in zip(model_names, probs):
        print(f"🔍 {name} Confidence: {prob:.2%}")

    # === 5. Final Soft Voting (weighted average of model predictions) ===
    model_weights = {"full": 0.25, "patch": 0.75}  # Assign more weight to patch model (patch-focused task)
    final_prob = sum(probs[i] * model_weights[k] for i, k in enumerate(["full", "patch"]))

    # Classify as Van Gogh if probability ≥ 0.5, otherwise not
    final_label = "Van Gogh 🎨" if final_prob >= 0.5 else "Not Van Gogh ❌"

    # Return the predicted label and its associated confidence
    return final_label, final_prob
