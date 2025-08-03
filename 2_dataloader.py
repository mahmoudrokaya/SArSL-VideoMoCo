import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

class ContrastiveDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_sample=8):
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_sample = frames_per_sample
        self.classes = os.listdir(root_dir)
        self.video_paths = self._gather_videos()

    def _gather_videos(self):
        video_list = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for video in os.listdir(class_dir):
                    video_list.append({
                        'class': class_name,
                        'path': os.path.join(class_dir, video)
                    })
        return video_list

    def _load_frames(self, video_path):
        frame_files = sorted([
            os.path.join(video_path, f) for f in os.listdir(video_path)
            if f.endswith('.jpg')
        ])
        selected = random.sample(frame_files, min(self.frames_per_sample, len(frame_files)))
        frames = [Image.open(f).convert("RGB") for f in selected]
        if self.transform:
            frames = [self.transform(f) for f in frames]
        return torch.stack(frames)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        anchor_info = self.video_paths[idx]
        anchor_class = anchor_info['class']
        anchor_video = anchor_info['path']
        anchor_frames = self._load_frames(anchor_video)

        # Decide on positive or negative sample (50% chance)
        if random.random() > 0.5:
            # Positive pair
            positive_candidates = [v for v in self.video_paths if v['class'] == anchor_class and v['path'] != anchor_video]
            if not positive_candidates:
                positive_candidates = [anchor_info]
            positive_info = random.choice(positive_candidates)
            positive_frames = self._load_frames(positive_info['path'])
            label = 1
        else:
            # Negative pair
            negative_candidates = [v for v in self.video_paths if v['class'] != anchor_class]
            negative_info = random.choice(negative_candidates)
            positive_frames = self._load_frames(negative_info['path'])
            label = 0

        return anchor_frames, positive_frames, torch.tensor(label, dtype=torch.float)

# Example Usage:
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = ContrastiveDataset(
    root_dir=r"E:\Mahmoud\Exams\46\461\New-Papers\Paper3-ِAtlam\Model\preprocessed_frames",
    transform=transform
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Example iteration
for x1, x2, y in dataloader:
    print("Batch:", x1.shape, x2.shape, y)
    break
