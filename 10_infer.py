import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os

def preprocess_video(video_path, clip_len=16, resize=(112, 112)):
    """
    Load and preprocess video into a 4D tensor [C, T, H, W]
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Temporal sampling
    total_frames = len(frames)
    if total_frames < clip_len:
        # Pad with last frame
        frames += [frames[-1]] * (clip_len - total_frames)
    else:
        indices = np.linspace(0, total_frames - 1, clip_len).astype(int)
        frames = [frames[i] for i in indices]

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])
    frames = [transform(frame) for frame in frames]
    clip_tensor = torch.stack(frames, dim=1)  # [C, T, H, W]
    return clip_tensor.unsqueeze(0)  # [1, C, T, H, W]

def predict_sign_label(model, video_path, class_names, device='cuda'):
    """
    Given a new video path, return predicted sign label
    """
    model.eval()
    model = model.to(device)

    input_tensor = preprocess_video(video_path).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = class_names[pred_idx]
        confidence = probs[0, pred_idx].item()
    
    return pred_label, confidence

# Example usage:
if __name__ == '__main__':
    from model import YourModelDefinition  # Replace with actual model import
    class_names = ['Hello', 'Thank you', 'Yes', 'No', 'I love you']  # Replace with actual labels
    
    model_path = 'E:/Mahmoud/Exams/46/461/New-Papers/Paper3-ِAtlam/Model/final_model.pth'
    video_path = 'D:/Exams/45/451/NewProject/Programs/DataSets/KARSL-502/test/hello_sample.mp4'

    model = YourModelDefinition(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    
    label, confidence = predict_sign_label(model, video_path, class_names)
    print(f"Predicted Label: {label} (Confidence: {confidence:.2f})")
