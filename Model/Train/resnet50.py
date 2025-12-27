import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parents[1]

X_PATH = Path(os.getenv("TRAIN_X_PATH", str(REPO_ROOT / "ecg_images_array.npy")))
Y_PATH = Path(os.getenv("TRAIN_Y_PATH", str(REPO_ROOT / "all_labels.npy")))

X = np.load(str(X_PATH))  # shape (N, 224, 224, 3)
y = np.load(str(Y_PATH))  # shape (N, num_classes), normalized

# Chia train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.34, random_state=48)

# Chuyển đổi dữ liệu để phù hợp với PyTorch (N, C, H, W)
X_train = np.transpose(X_train, (0, 3, 1, 2))  # (N, 3, 224, 224)
X_val = np.transpose(X_val, (0, 3, 1, 2))

# Normalize dữ liệu (0-1)
X_train = X_train.astype(np.float32) / 255.0
X_val = X_val.astype(np.float32) / 255.0

# Chuyển đổi sang tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val)

# Tạo DataLoader
batch_size = 120
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Number of classes: {y.shape[1]}")

# Định nghĩa model ResNet50
class ECGResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ECGResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        
        # Thay đổi fully connected layer cuối
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # Multi-label classification
        )
    
    def forward(self, x):
        return self.resnet(x)

# Khởi tạo model
num_classes = y.shape[1]
model = ECGResNet50(num_classes).to(device)

# Loss function và optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy cho multi-label
optimizer = optim.Adam(model.parameters())

# Training function
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Tính accuracy (threshold = 0.5 cho sigmoid)
        predicted = (output > 0.5).float()
        total += target.size(0) * target.size(1)
        correct += (predicted == target).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# Validation function
def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            # Tính accuracy
            predicted = (output > 0.5).float()
            total += target.size(0) * target.size(1)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc

# Training loop
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_acc = 0.0
best_model_path = Path(os.getenv("TRAIN_BEST_MODEL_PATH", str(MODEL_DIR / "best_resnet50_ecg_model.pth")))


for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    print('-' * 50)
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Validate
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    # Lưu metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    # Lưu model tốt nhất
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, str(best_model_path))
        print(f'New best model saved! Validation Accuracy: {best_val_acc:.2f}%')

print(f'\nTraining completed!')
print(f'Best validation accuracy: {best_val_acc:.2f}%')
print(f'Best model saved as: {best_model_path}')



# Lưu training history
training_history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'train_accuracies': train_accuracies,
    'val_accuracies': val_accuracies,
    'best_val_acc': best_val_acc
}

history_path = Path(os.getenv("TRAIN_HISTORY_PATH", str(MODEL_DIR / "resnet50_training_history.npy")))
np.save(str(history_path), training_history)
print(f"Training history saved as '{history_path}'")

# Load và test model đã lưu
print("\nLoading best model for final evaluation...")
checkpoint = torch.load(str(best_model_path))
model.load_state_dict(checkpoint['model_state_dict'])

final_val_loss, final_val_acc = validate_epoch(model, val_loader, criterion, device)
print(f"Final validation accuracy: {final_val_acc:.2f}%")

# Hàm để load model sau này
def load_trained_model(model_path, num_classes, device):
    """
    Load trained ResNet50 model
    """
    model = ECGResNet50(num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['best_val_acc']
