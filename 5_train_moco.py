import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime

from video_moco import VideoMoCo
from dataloader import VideoContrastiveDataset
from r3d_backbone import R3DBackbone

# Paths
DATA_PATH = r"D:\Exams\45\451\NewProject\Programs\DataSets\KARSL-502"
OUTPUT_PATH = r"E:\Mahmoud\Exams\46\461\New-Papers\Paper3-ِAtlam\Model"
CHECKPOINT_PATH = os.path.join(OUTPUT_PATH, "checkpoints")
LOG_FILE = os.path.join(OUTPUT_PATH, "training_log.txt")

# Training Config
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output dirs
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Transforms
video_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor()
])

# Dataset and Dataloader
train_dataset = VideoContrastiveDataset(DATA_PATH, transform=video_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Model & Optimizer
model = VideoMoCo(base_encoder=R3DBackbone).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# Training Loop
def train():
    model.train()
    with open(LOG_FILE, "w") as log_f:
        for epoch in range(NUM_EPOCHS):
            total_loss, total_correct, total_samples = 0.0, 0, 0
            for im_q, im_k in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                im_q, im_k = im_q.to(DEVICE), im_k.to(DEVICE)
                logits, labels = model(im_q, im_k)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * im_q.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += torch.sum(preds == labels).item()
                total_samples += im_q.size(0)

            avg_loss = total_loss / total_samples
            acc = total_correct / total_samples * 100
            log = f"[{datetime.now()}] Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.2f}%"
            print(log)
            log_f.write(log + "\n")

            torch.save(model.state_dict(), os.path.join(CHECKPOINT_PATH, f"moco_epoch_{epoch+1}.pth"))

train()
