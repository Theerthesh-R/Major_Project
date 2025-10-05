import os
import numpy as np
import torch
import cv2
from facenet_pytorch import InceptionResnetV1

# ── Setup device ────────────────────────────────────────────────
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# ── Load FaceNet (VGGFace2 pretrained) ──────────────────────────
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ── Directories ────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
cropped_path = os.path.join(BASE_DIR, "cropped")
output_path = os.path.join(BASE_DIR, "embeddings")

if not os.path.exists(cropped_path):
    print(f"❌ Cropped folder not found: {cropped_path}")
    exit()

os.makedirs(output_path, exist_ok=True)

# ── Embedding function ─────────────────────────────────────────
def get_embedding(img_bgr):
    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Resize to 160x160 (FaceNet default)
    img_rgb = cv2.resize(img_rgb, (160, 160))

    # Convert to tensor [C,H,W], normalize to ~[-1,1]
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor - 127.5) / 128.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        embedding = embedder(img_tensor)  # [1,512]
    return embedding.squeeze(0).cpu().numpy()

# ── Process all cropped faces per person ────────────────────────
for person in os.listdir(cropped_path):
    person_folder = os.path.join(cropped_path, person)
    if not os.path.isdir(person_folder):
        continue

    print(f"🔹 Processing person: {person}")
    embeddings = []  # collect all embeddings for this person

    for img_name in os.listdir(person_folder):
        if img_name.lower().endswith(".jpg"):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not read {img_path}")
                continue

            emb = get_embedding(img)
            embeddings.append(emb)

    if not embeddings:
        print(f"⚠️ No valid images found for {person}, skipping.")
        continue

    # Average all embeddings → single 512-d vector
    mean_embedding = np.mean(embeddings, axis=0)

    out_file = os.path.join(output_path, f"{person}.npy")
    np.save(out_file, mean_embedding)
    print(f"💾 Saved combined embedding: {out_file}")

print("✅ All combined embeddings saved successfully!")
