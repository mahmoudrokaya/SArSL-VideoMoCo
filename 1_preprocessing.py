import cv2
import os

input_path = r"D:\Exams\45\451\NewProject\Programs\DataSets\KARSL-502"
output_path = r"E:\Mahmoud\Exams\46\461\New-Papers\Paper3-ِAtlam\Model\preprocessed_frames"

# Create output directory if not exists
os.makedirs(output_path, exist_ok=True)

# Resize parameters
frame_size = (224, 224)

for video_file in os.listdir(input_path):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(input_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        video_output_dir = os.path.join(output_path, os.path.splitext(video_file)[0])
        os.makedirs(video_output_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, frame_size)
            frame_filename = os.path.join(video_output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            frame_count += 1

        cap.release()
print("Preprocessing completed.")
