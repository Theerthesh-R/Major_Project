import os
import sys
import cv2
import torch
from facenet_pytorch import MTCNN
import numpy as np

def crop_faces(input_dir, output_dir):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    detector = MTCNN(keep_all=False, device=device)

    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Looking inside: {input_dir}")

    for file in os.listdir(input_dir):
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(input_dir, file)
            print(f"📸 Processing {img_path}")
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Could not read image: {img_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes, probs = detector.detect(rgb)

            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = [int(v) for v in boxes[0]]

                # Clip coordinates to image dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)

                cropped_face = image[y1:y2, x1:x2]

                # Skip if crop is empty
                if cropped_face.size == 0:
                    print(f"⚠️ Empty crop for {file}, skipping")
                    continue

                out_path = os.path.join(output_dir, file)
                cv2.imwrite(out_path, cropped_face)
                print(f"✅ Saved cropped face: {out_path}")
            else:
                print(f"⚠️ No face detected in {file}")

# --- Main ---
if len(sys.argv) < 2:
    print("Usage: python crop_faces.py <person_name>")
    sys.exit(1)

name = sys.argv[1]
input_dir = f"/home/victus/face_attendance_final/face/backend/dataset/{name}"
output_dir = f"/home/victus/face_attendance_final/face/backend/cropped/{name}"

if not os.path.exists(input_dir):
    print(f"❌ Folder not found: {input_dir}")
else:
    crop_faces(input_dir, output_dir)
