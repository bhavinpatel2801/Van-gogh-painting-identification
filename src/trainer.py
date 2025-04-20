import torch                         # Core tensor library
import torch.nn as nn                # Neural network components
import torch.nn.functional as F      # Activation functions like relu
import torchvision.models as models  # Pretrained CNN architectures
import wandb                         # For logging experiments
import torch.optim as optim          # Optimizers like Adam, SGD
from sklearn.metrics import f1_score # For F1 score calculation
from torchvision.models import vit_b_16, ViT_B_16_Weights # Pretrained ViT model

#  Basic CNN
class BasicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BasicCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)    # Output: 16 x 224 x 224
        self.pool = nn.MaxPool2d(2, 2)                 # Downsamples by 2
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)   # Output: 32 x 112 x 112
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)   # Output: 64 x 56 x 56

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 28 * 28, 128)        # Flatten and reduce to 128
        self.fc2 = nn.Linear(128, num_classes)         # Output logits for classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))           # Conv -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)                      # Flatten
        x = F.relu(self.fc1(x))                        # FC -> ReLU
        x = self.fc2(x)                                # Final logits
        return x

#  ResNet18 Transfer Learning
def get_resnet18_transfer_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Load pretrained ResNet18

    # Freeze all layers except the last block (layer4) and fc
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Replace final FC layer with our desired number of output classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_and_evaluate(model, train_loader, test_loader, num_epochs=15, lr=0.001, patience=5, device='cpu', model_name="BasicCNN"):
    # Move model to target device (CPU or GPU)
    model.to(device)

    # Define the loss function for classification
    criterion = nn.CrossEntropyLoss()

    # Use Adam optimizer for gradient updates
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler to reduce LR by gamma after every 'step_size' epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Setup for early stopping
    best_loss = float('inf')          # Track best loss observed
    epochs_no_improve = 0             # Counter for early stopping
    best_model_state = None           # Store best model parameters
    min_delta = 0.005                 # Minimum improvement to reset patience

    # Log gradients and parameters to Weights & Biases
    wandb.watch(model, log="all", log_freq=100)

    # ================= Training Loop =================
    for epoch in range(num_epochs):
        model.train()                 # Set model to training mode
        running_loss = 0.0           # Accumulate total loss per epoch

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()    # Clear previous gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()          # Backpropagation
            optimizer.step()         # Update model parameters

            running_loss += loss.item() * inputs.size(0)  # Add batch loss

        # Step the learning rate scheduler
        scheduler.step()

        # Compute average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Log loss to W&B
        wandb.log({f"{model_name}/train_loss": epoch_loss, "epoch": epoch + 1})

        # ================= Early Stopping =================
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()  # Save current best weights
        else:
            epochs_no_improve += 1
            print(f"⚠️  No significant improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("⏹️  Early stopping triggered.")
                break

    # Restore the best model weights (based on lowest loss)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ================= Final Evaluation =================
    model.eval()                     # Set model to evaluation mode
    correct, total = 0, 0            # Accuracy counters
    y_true, y_pred = [], []          # Store true and predicted labels for F1

    with torch.no_grad():           # Disable gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Take class with highest prob

            correct += (predicted == labels).sum().item()  # Count correct preds
            total += labels.size(0)                        # Count total samples

            y_true.extend(labels.cpu().numpy())            # Store true labels
            y_pred.extend(predicted.cpu().numpy())         # Store predictions

    # Calculate final accuracy and F1 score
    accuracy = correct / total
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"✅ Final Accuracy: {accuracy:.4f}")
    print(f"✅ Final F1 Score: {f1:.4f}")

    # Log metrics to W&B
    wandb.log({
        f"{model_name}/final_accuracy": accuracy,
        f"{model_name}/final_f1_score": f1
    })

    return model, accuracy, f1  # Return trained model and metrics

def get_vit_model(num_classes=2):
    # Load default pretrained weights for ViT-B/16 (Vision Transformer with 16x16 patch size)
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # Selectively unfreeze layers for fine-tuning
    # Only update: 
    # - LayerNorm (`encoder.ln`) for stability,
    # - One of the deeper transformer blocks (`encoder.layers.10`),
    # - Classification head (`heads`)
    for name, param in model.named_parameters():
        if "encoder.ln" in name or "encoder.layers.10" in name or "heads" in name:
            param.requires_grad = True   # Allow gradients (fine-tuning)
        else:
            param.requires_grad = False  # Keep all other layers frozen

    # Replace the final classification head to match the number of target classes
    in_features = model.heads[0].in_features  # Get input features to the head
    model.heads = nn.Sequential(nn.Linear(in_features, num_classes))  # New head layer

    return model  # Return customized ViT model

def get_model(name='resnet', num_classes=2):
    # Select and initialize a pretrained model based on the 'name' argument
    if name == 'resnet':
        # Load pretrained ResNet18 with default weights
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Get number of input features to the final FC layer
        in_features = model.fc.in_features
        
        # Replace final FC layer to output logits for `num_classes`
        model.fc = nn.Linear(in_features, num_classes)

    elif name == 'vgg':
        # Load pretrained VGG16 with default weights
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Get input size of the final linear classifier layer
        in_features = model.classifier[6].in_features

        # Replace it with new linear layer for our task
        model.classifier[6] = nn.Linear(in_features, num_classes)

    elif name == 'efficientnet':
        # Load pretrained EfficientNet-B0 with default weights
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

        # Get input features for its classifier
        in_features = model.classifier[1].in_features

        # Replace final classification layer
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        # Raise error if model name is invalid
        raise ValueError("Model name not recognized.")

    # Return the customized model ready for training or inference
    return model

def patch_train_and_evaluate(model, train_loader, test_loader, num_epochs=15, lr=1e-3, patience=5, device='cpu', model_name="ResNet18_Patch"):
    # Move model to the target device (CPU or GPU)
    model.to(device)

    # Loss function for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Early stopping setup
    best_loss = float('inf')          # Best loss seen so far
    epochs_no_improve = 0             # Number of epochs without significant improvement
    best_model_state = None           # To store the best model parameters
    min_delta = 0.005                 # Minimum improvement in loss to reset patience

    # Log model gradients and parameters to Weights & Biases
    wandb.watch(model, log="all", log_freq=100)

    # ================= Training Loop =================
    for epoch in range(num_epochs):
        model.train()                 # Set model to training mode
        running_loss = 0.0            # Accumulate batch losses

        for inputs, labels in train_loader:
            # Move inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()     # Clear previous gradients
            outputs = model(inputs)   # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()           # Backpropagation
            optimizer.step()          # Update model parameters

            # Add loss weighted by batch size
            running_loss += loss.item() * inputs.size(0)

        # Compute average loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # Log training loss to W&B
        wandb.log({f"{model_name}/train_loss": epoch_loss, "epoch": epoch + 1})

        # ================= Early Stopping =================
        if epoch_loss < best_loss - min_delta:
            # Improvement: save best model
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            # No improvement
            epochs_no_improve += 1
            print(f"⚠️  No significant improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                # Stop if patience exceeded
                print("⏹️  Early stopping triggered.")
                break

    # Restore best weights before evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ================= Final Evaluation (F1 Score) =================
    model.eval()                      # Set model to evaluation mode
    y_true, y_pred = [], []          # Lists to store ground truth and predictions

    with torch.no_grad():            # Disable gradient tracking
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted class index

            # Accumulate true and predicted labels
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute weighted F1 score
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"✅ Final F1 Score: {f1:.4f}")

    # Log final F1 score to W&B
    wandb.log({f"{model_name}/final_f1_score": f1})

    # Return the trained model and its F1 score
    return model, f1
