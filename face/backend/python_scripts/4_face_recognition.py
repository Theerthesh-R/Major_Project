import os
import sys
import time
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import requests
import mysql.connector
import threading

# --------------------------
# Threaded RTSP Stream Reader
# --------------------------
class RTSPStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()


# --------------------------
# GPU / CUDA setup
# --------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --------------------------
# Config
# --------------------------
USE_COSINE = False
THRESHOLD = 1.0
FRAMES_REQUIRED = 10

# --------------------------
# Face Detection + Embedding
# --------------------------
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --------------------------
# DB connection
# --------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="final_db_for_project"
)
cursor = conn.cursor()

cursor.execute("""
    SELECT s.name, f.embedding FROM face_embeddings f
    JOIN students s ON s.student_id = f.student_id
""")
known_embeddings, names = [], []
for name, emb_str in cursor.fetchall():
    emb = np.array(list(map(float, emb_str.strip('[]').split(','))))
    known_embeddings.append(emb)
    names.append(name)

# --------------------------
# Attendance marking
# --------------------------
already_marked = set()
detection_counts = {}

def mark_attendance(name):
    if name not in already_marked:
        try:
            r = requests.post("http://localhost:3000/attendance", json={"name": name})
            if r.ok:
                already_marked.add(name)
                print(f"✅ Attendance marked: {name}")
            else:
                print(f"❌ API failed ({r.status_code})")
        except Exception as e:
            print("Error:", e)

# --------------------------
# Similarity function
# --------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --------------------------
# Initialize RTSP stream with threading
# --------------------------
url = "rtsp://admin:Sahyadri%401234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
stream = RTSPStream(url)

cv2.namedWindow("Recognition - Dahua RTSP", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Recognition - Dahua RTSP", 800, 600)

fps_time = time.time()

while True:
    ret, frame = stream.read()
    if not ret or frame is None:
        print("⚠️ No frame received, retrying…")
        time.sleep(0.1)
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs = mtcnn.detect(rgb_frame)
    if boxes is not None:
        faces = []
        h, w, _ = rgb_frame.shape
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            face_img = rgb_frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            face_img = cv2.resize(face_img, (160, 160))
            face_tensor = torch.tensor(face_img).permute(2,0,1).unsqueeze(0).float()/255.0
            faces.append(face_tensor)

        if faces:
            faces_tensor = torch.cat(faces).to(device)
            embeddings = resnet(faces_tensor).detach().cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                emb = embeddings[i]
                name = "Unknown"

                if USE_COSINE:
                    max_sim, best_match = -1, None
                    for j, known_emb in enumerate(known_embeddings):
                        sim = cosine_similarity(emb, known_emb)
                        if sim > max_sim:
                            max_sim, best_match = sim, names[j]
                    if max_sim >= THRESHOLD:
                        name = best_match
                else:
                    min_dist, best_match = float('inf'), None
                    for j, known_emb in enumerate(known_embeddings):
                        dist = np.linalg.norm(emb - known_emb)
                        if dist < min_dist:
                            min_dist, best_match = dist, names[j]
                    if min_dist <= THRESHOLD:
                        name = best_match

                # Draw box & label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Attendance logic
                if name != "Unknown":
                    detection_counts[name] = detection_counts.get(name, 0) + 1
                    if detection_counts[name] >= FRAMES_REQUIRED and name not in already_marked:
                        mark_attendance(name)
                        detection_counts[name] = 0
                else:
                    detection_counts["Unknown"] = 0

    # Show FPS
    now = time.time()
    fps = 1 / (now - fps_time)
    fps_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Recognition - Dahua RTSP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()
