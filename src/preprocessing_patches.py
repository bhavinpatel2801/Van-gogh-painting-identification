from torchvision import transforms       # For image transformations like resizing and normalization
from PIL import Image                   # For image loading and manipulation
import random                           # To randomly sample patches

import sys
sys.path.append('../src')              # Add '../src' to the system path to allow importing custom modules

def patchify_image(img, patch_size=224):
    # Get dimensions of the input image
    width, height = img.size
    patches = []

    # Slide a 224x224 window over the image to extract patches
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            # Calculate bottom-right corner of the patch
            right = min(left + patch_size, width)
            bottom = min(top + patch_size, height)
            
            # Crop the patch from the image
            patch = img.crop((left, top, right, bottom))
            
            # Only include patches that are exactly 224x224
            if patch.size == (patch_size, patch_size):
                patches.append(patch)
    
    return patches  # Return list of PIL patches

transform = transforms.Compose([
    transforms.Resize((224, 224)),              # Resize any patch to exactly 224x224
    transforms.ToTensor(),                      # Convert patch to a PyTorch tensor
    transforms.Normalize(                       # Normalize using mean/std = 0.5
        mean=[0.5, 0.5, 0.5],                    # Each RGB channel centered to 0
        std=[0.5, 0.5, 0.5]                      # Scaled to range [-1, 1]
    )
])

def extract_patches_and_labels(images, label, transform, max_patches=20):
    patch_tensors = []  # Store tensor representations of patches
    patch_labels = []   # Corresponding labels (same for all patches from same image)

    for img in images:
        # Convert image to list of 224x224 patches
        patches = patchify_image(img, 224)

        # If too many patches, randomly sample `max_patches`
        if len(patches) > max_patches:
            patches = random.sample(patches, max_patches)

        # Transform each patch and add label
        for patch in patches:
            tensor = transform(patch)
            patch_tensors.append(tensor)
            patch_labels.append(label)
    
    return patch_tensors, patch_labels  # Return list of tensors and their labels
# Avoid redefining transform repeatedly in large files
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Apply same normalization
])

def predict_image_patches(image, model, device='cpu'):
    model.eval()  # Set model to evaluation mode

    # Divide image into 224x224 patches
    patches = patchify_image(image, patch_size=224)

    # Apply transform to each patch and add batch dimension
    tensors = [transform(p).unsqueeze(0) for p in patches]

    # Concatenate all patch tensors into a single batch (N, C, H, W)
    inputs = torch.cat(tensors).to(device)

    with torch.no_grad():  # No gradients needed for inference
        outputs = model(inputs)                   # Forward pass on batch
        probs = torch.softmax(outputs, dim=1)     # Apply softmax for class probabilities
        preds = torch.argmax(probs, dim=1)        # Get predicted class for each patch

    labels = preds.tolist()  # Convert tensor to list of predicted class indices
    votes = sum(labels)      # Count how many patches voted "Van Gogh" (label 1)
    total = len(labels)      # Total number of patches
    confidence = votes / total  # Confidence = proportion of Van Gogh votes

    # Final label based on majority vote
    label = 1 if confidence > 0.5 else 0

    return label, confidence  # Return binary label and associated confidence score
