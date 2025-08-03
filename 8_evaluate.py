import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def evaluate_model(model, dataloader, class_names, device='cuda', save_dir='evaluation_results'):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for videos, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_probs = torch.cat(all_probs).numpy()

    # Top-1 Accuracy
    top1_acc = accuracy_score(y_true, y_pred)

    # Top-5 Accuracy
    top5_preds = np.argsort(-y_probs, axis=1)[:, :5]
    top5_correct = sum([y_true[i] in top5_preds[i] for i in range(len(y_true))])
    top5_acc = top5_correct / len(y_true)

    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")

    # F1 Score Per Class
    f1_per_class = f1_score(y_true, y_pred, average=None)
    for i, f1 in enumerate(f1_per_class):
        print(f"F1 Score for {class_names[i]}: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

    # Precision-Recall Curves
    plt.figure(figsize=(12, 8))
    for i in range(len(class_names)):
        precision, recall, _ = precision_recall_curve(y_true == i, y_probs[:, i])
        plt.plot(recall, precision, label=f"{class_names[i]}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'precision_recall_curves.png'))
    plt.close()

    # Detailed Report
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    return {
        'Top-1 Accuracy': top1_acc,
        'Top-5 Accuracy': top5_acc,
        'F1 per class': dict(zip(class_names, f1_per_class))
    }
