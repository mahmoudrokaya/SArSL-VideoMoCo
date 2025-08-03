import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class R3DBackbone(nn.Module):
    def __init__(self, output_dim=512):
        super(R3DBackbone, self).__init__()
        base_model = r3d_18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # Remove last pooling + fc
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        """
        x: shape [B, 3, T, H, W]
        """
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.projection(x)
        return x  # shape: [B, output_dim]
