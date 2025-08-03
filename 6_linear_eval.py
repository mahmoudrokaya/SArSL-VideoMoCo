import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 30
feature_dim = 512  # Adjust based on your 3D CNN output
num_classes = 50   # Update this to match number of sign classes
data_path = r"D:\Exams\45\451\NewProject\Programs\DataSets\KARSL-502"

# === Dummy Dataset (Replace with real one) ===
class LabeledVideoDataset(Dataset):
    def __init__(self, feature_path, label_path):
        self.features = np.load(feature_path)
        self.labels = np.load(label_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), self.labels[idx]

# === Classifier Head ===
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

# === Load Pre-Extracted Features and Labels ===
# Replace these paths with actual .npy files of features/labels
train_features_path = os.path.join(data_path, "train_features.npy")
train_labels_path = os.path.join(data_path, "train_labels.npy")
val_features_path = os.path.join(data_path, "val_features.npy")
val_labels_path = os.path.join(data_path, "val_labels.npy")

train_dataset = LabeledVideoDataset(train_features_path, train_labels_path)
val_dataset = LabeledVideoDataset(val_features_path, val_labels_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# === Initialize Classifier ===
model = LinearClassifier(input_dim=feature_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Loss = {train_loss/len(train_loader):.4f}, Accuracy = {acc:.2f}%")

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100. * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

# === Save Classifier ===
os.makedirs(r"E:\Mahmoud\Exams\46\461\New-Papers\Paper3-ِAtlam\Model", exist_ok=True)
torch.save(model.state_dict(), r"E:\Mahmoud\Exams\46\461\New-Papers\Paper3-ِAtlam\Model\linear_classifier.pth")
