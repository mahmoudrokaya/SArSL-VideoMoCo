import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm

# === Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
num_epochs = 25
num_classes = 50  # Update based on your dataset
video_frames = 16  # Number of frames per clip
input_channels = 3  # RGB
video_size = 112  # Height and width
learning_rate = 1e-4
dataset_path = r"D:\Exams\45\451\NewProject\Programs\DataSets\KARSL-502"
output_path = r"E:\Mahmoud\Exams\46\461\New-Papers\Paper3-ِAtlam\Model"

# === Dataset Class ===
class SignLanguageVideoDataset(Dataset):
    def __init__(self, clips_path, labels_path):
        self.clips = np.load(clips_path)  # shape: (N, C, T, H, W)
        self.labels = np.load(labels_path)  # shape: (N,)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Already tensor if saved in .npy
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        clip = torch.tensor(self.clips[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return clip, label

# === Load Data ===
train_clips = os.path.join(dataset_path, "train_clips.npy")
train_labels = os.path.join(dataset_path, "train_labels.npy")
val_clips = os.path.join(dataset_path, "val_clips.npy")
val_labels = os.path.join(dataset_path, "val_labels.npy")

train_dataset = SignLanguageVideoDataset(train_clips, train_labels)
val_dataset = SignLanguageVideoDataset(val_clips, val_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === Load Pretrained Backbone (R3D / I3D) ===
from torchvision.models.video import r3d_18

class SignRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(SignRecognitionModel, self).__init__()
        self.backbone = r3d_18(pretrained=False)
        self.backbone.stem[0] = nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                                          stride=(1, 2, 2), padding=(1, 3, 3),
                                          bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

model = SignRecognitionModel(num_classes=num_classes).to(device)

# === Optional: Load Pretrained Weights ===
pretrained_weights = os.path.join(output_path, "pretrained_backbone.pth")
if os.path.exists(pretrained_weights):
    model.load_state_dict(torch.load(pretrained_weights), strict=False)
    print("Loaded pretrained backbone weights.")

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Training Loop ===
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for clips, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        clips, labels = clips.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = 100. * correct / total
    print(f"Train Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}, Accuracy = {acc:.2f}%")

    # === Validation ===
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for clips, labels in val_loader:
            clips, labels = clips.to(device), labels.to(device)
            outputs = model(clips)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100. * val_correct / val_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

# === Save Final Model ===
os.makedirs(output_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(output_path, "finetuned_sign_model.pth"))
