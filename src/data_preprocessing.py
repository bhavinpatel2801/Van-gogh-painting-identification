# Import torchvision transforms for resizing and converting images to tensors
from torchvision import transforms
# Import torch for tensor operations and dataset creation
import torch

def basic_transform(size=(224, 224)):
    """
    Returns a torchvision transform that resizes and converts images to tensors.
    """
    return transforms.Compose([
        transforms.Resize(size),      # Resize the input image to the given size (default: 224x224)
        transforms.ToTensor()         # Convert the PIL image to a PyTorch tensor (and normalize to [0,1])
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
    # torch.stack combines a list of image tensors into a single 4D tensor (N, C, H, W)
    # torch.tensor(label_list) converts the list of labels to a tensor of shape (N,)
    return torch.utils.data.TensorDataset(torch.stack(image_tensors), torch.tensor(label_list))
