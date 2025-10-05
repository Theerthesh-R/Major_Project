import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

# -------------------------------
# Settings
# -------------------------------
EMBEDDINGS_DIR = "embeddings"     # folder with saved embeddings (.npy)
TEST_IMAGES_DIR = "test_images"   # folder with test images (subfolders per person)
THRESHOLD = 0.7                  # similarity threshold
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Load FaceNet
# -------------------------------
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def get_embedding(img_bgr):
    """Convert image to FaceNet embedding (512D)"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160))
    img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
    img_tensor = (img_tensor - 127.5) / 128.0
    img_tensor = img_tensor.to(DEVICE)
    with torch.no_grad():
        embedding = embedder(img_tensor)
    return embedding.squeeze(0).cpu().numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------------------
# Load known embeddings
# -------------------------------
known_embeddings = []
names = []

for file in os.listdir(EMBEDDINGS_DIR):
    if file.endswith(".npy"):
        name = file.split(".")[0]
        emb = np.load(os.path.join(EMBEDDINGS_DIR, file))
        known_embeddings.append(emb)
        names.append(name)

print(f"🔹 Loaded {len(names)} known persons: {names}")

# Create a mapping for flexible name matching
name_mapping = {}
for name in names:
    # Create lowercase version without spaces for flexible matching
    simplified = name.lower().replace(" ", "").replace("_", "").replace("-", "")
    name_mapping[simplified] = name

print(f"🔹 Name mapping: {name_mapping}")

# -------------------------------
# Test each image
# -------------------------------
correct = 0
total = 0

for person_folder in os.listdir(TEST_IMAGES_DIR):
    folder_path = os.path.join(TEST_IMAGES_DIR, person_folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n--- Testing {person_folder} ---")
    
    # Simplify folder name for matching
    folder_simple = person_folder.lower().replace(" ", "").replace("_", "").replace("-", "")
    
    # Check if this person exists in our database
    person_in_database = folder_simple in name_mapping
    expected_name = name_mapping[folder_simple] if person_in_database else None
    
    print(f"Folder '{person_folder}' -> simplified: '{folder_simple}' -> in DB: {person_in_database} -> expected: {expected_name}")
    
    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {img_path}")
            continue

        test_emb = get_embedding(img)
        sims = [cosine_similarity(test_emb, emb) for emb in known_embeddings]
        best_idx = np.argmax(sims)
        best_score = sims[best_idx]
        predicted_name = names[best_idx] if best_score >= THRESHOLD else "Unknown"

        # CORRECT ACCURACY CALCULATION:
        if person_in_database:
            # This person should be recognized with correct name
            is_correct = (predicted_name == expected_name)
            expected_str = expected_name
        else:
            # This person should be rejected as Unknown
            is_correct = (predicted_name == "Unknown")
            expected_str = "Unknown"

        if is_correct:
            correct += 1
            status = "✅ CORRECT"
        else:
            status = "❌ WRONG"

        total += 1

        print(f"[{person_folder}] {img_file} → Predicted: {predicted_name}, Expected: {expected_str}, Score: {best_score:.2f} {status}")

# -------------------------------
# Print accurate results
# -------------------------------
accuracy = (correct / total) * 100 if total > 0 else 0
print(f"\n📊 FINAL RESULTS:")
print(f"Total test images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.2f}%")