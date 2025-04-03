# src/data_preprocessing.py

from torchvision import transforms
import torch

def basic_transform(size=(224, 224)):
    """
    Returns a torchvision transform that resizes and converts images to tensors.
    """
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

def apply_transform(images, transform):
    """
    Applies a given transform to a list of PIL images.
    Returns a list of transformed image tensors.
    """
    return [transform(img) for img in images]

def create_dataset(image_tensors, label_list):
    """
    Creates a PyTorch TensorDataset from image tensors and labels.
    """
    return torch.utils.data.TensorDataset(torch.stack(image_tensors), torch.tensor(label_list))
