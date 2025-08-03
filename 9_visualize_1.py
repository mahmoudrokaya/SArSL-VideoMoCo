import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import os

def visualize_embeddings(model, dataloader, class_names, save_dir='visualization', method='tsne', device='cuda'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval().to(device)

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in dataloader:
            videos = videos.to(device)
            feats = model.backbone(videos)  # assumes features before final classifier
            if isinstance(feats, tuple):
                feats = feats[0]
            feats = feats.view(feats.size(0), -1)  # flatten
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    features = torch.cat(all_feats).numpy()
    labels = torch.cat(all_labels).numpy()

    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=30, n_iter=300)
    elif method == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    else:
        raise ValueError("Choose 'tsne' or 'umap'")

    reduced = reducer.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=[class_names[i] for i in labels], palette='tab10', s=50)
    plt.title(f"Clip Embedding Visualization ({method.upper()})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{method}_embedding_visualization.png'))
    plt.close()
