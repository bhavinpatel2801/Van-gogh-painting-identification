import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

#  Basic CNN
class BasicCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#  ResNet18 Transfer Learning
def get_resnet18_transfer_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Unfreeze last few layers
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


import torch.nn as nn
import torch.optim as optim

def train_and_evaluate(model, train_loader, test_loader, num_epochs=15, lr=0.001, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ✅ Initialize scheduler only once
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        # ✅ Step scheduler after each epoch
        scheduler.step()

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

    # ✅ Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Final Test Accuracy: {accuracy:.4f}")

    return model, accuracy


from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn

from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.nn as nn

def get_vit_model(num_classes=2):
    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    # ✅ Unfreeze selected layers for fine-tuning
    for name, param in model.named_parameters():
        if "encoder.ln" in name or "encoder.layers.10" in name or "heads" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # ✅ Replace the classification head
    in_features = model.heads[0].in_features
    model.heads = nn.Sequential(nn.Linear(in_features, num_classes))

    return model





def get_model(name='resnet', num_classes=2):
    if name == 'resnet':
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == 'vgg':
        from torchvision.models import vgg16, VGG16_Weights
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif name == 'efficientnet':
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Model name not recognized.")
    
    return model

from sklearn.metrics import f1_score
import torch.nn as nn
import torch

def patch_train_and_evaluate(model, train_loader, test_loader, num_epochs=10, lr=1e-3, patience=3, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    min_delta = 0.02 # only count as improvement if this much better

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        # ✅ Early Stopping Check (INSIDE loop)
        if epoch_loss < best_loss - min_delta:
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            print(f"⚠️  Minor improvment improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("⏹️  Early stopping triggered.")
                break

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ✅ Final Evaluation (F1)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"✅ Final F1 Score: {f1:.4f}")
    return model, f1
