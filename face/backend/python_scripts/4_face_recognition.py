# import os
# import sys
# import time
# import cv2
# import numpy as np
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import requests
# import mysql.connector
# import threading

# # --------------------------
# # Threaded RTSP Stream Reader
# # --------------------------
# class RTSPStream:
#     def __init__(self, url):
#         self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#         #self.cap = cv2.VideoCapture(0)
#         self.ret, self.frame = self.cap.read()
#         self.lock = threading.Lock()
#         self.running = True
#         threading.Thread(target=self.update, daemon=True).start()

#     def update(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 with self.lock:
#                     self.ret, self.frame = ret, frame

#     def read(self):
#         with self.lock:
#             return self.ret, self.frame

#     def release(self):
#         self.running = False
#         self.cap.release()

# # --------------------------
# # GPU / CUDA setup
# # --------------------------
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {device}")

# # --------------------------
# # Config
# # --------------------------
# USE_COSINE = True      # True = cosine similarity; False = Euclidean distance
# THRESHOLD = 0.75   # Confidence threshold for recognition
# FRAMES_REQUIRED = 10   # frames before marking attendance

# # --------------------------
# # Face Detection + Embedding
# # --------------------------
# mtcnn = MTCNN(keep_all=True, device=device)
# resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# # --------------------------
# # DB connection
# # --------------------------
# conn = mysql.connector.connect(
#     host="localhost",
#     user="root",
#     password="password",
#     database="majorproject"
# )
# cursor = conn.cursor()

# cursor.execute("""
#     SELECT s.name, f.embedding FROM face_embeddings f
#     JOIN students s ON s.student_id = f.student_id
# """)
# known_embeddings, names = [], []
# for name, emb_str in cursor.fetchall():
#     emb = np.array(list(map(float, emb_str.strip('[]').split(','))))
#     known_embeddings.append(emb)
#     names.append(name)

# # --------------------------
# # Attendance marking
# # --------------------------
# already_marked = set()
# detection_counts = {}

# def mark_attendance(name):
#     if name not in already_marked:
#         try:
#             r = requests.post("http://localhost:3000/attendance", json={"name": name})
#             if r.ok:
#                 already_marked.add(name)
#                 print(f"✅ Attendance marked: {name}")
#             else:
#                 print(f"❌ API failed ({r.status_code})")
#         except Exception as e:
#             print("Error:", e)

# # --------------------------
# # Similarity function
# # --------------------------
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))









# # --------------------------
# # Initialize RTSP stream with threading
# # --------------------------
# url = "rtsp://admin:Sahyadri%401234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
# stream = RTSPStream(url)

# cv2.namedWindow("Recognition - Dahua RTSP", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Recognition - Dahua RTSP", 800, 600)

# fps_time = time.time()

# while True:
#     ret, frame = stream.read()
#     if not ret or frame is None:
#         print("⚠️ No frame received, retrying…")
#         time.sleep(0.1)
#         continue

#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect faces
#     boxes, probs = mtcnn.detect(rgb_frame)
#     if boxes is not None:
#         faces = []
#         h, w, _ = rgb_frame.shape
#         for box in boxes:
#             x1, y1, x2, y2 = [int(b) for b in box]
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)
#             face_img = rgb_frame[y1:y2, x1:x2]
#             if face_img.size == 0:
#                 continue
#             face_img = cv2.resize(face_img, (160, 160))
#             face_tensor = torch.tensor(face_img).permute(2,0,1).unsqueeze(0).float()/255.0
#             faces.append(face_tensor)

#         if faces:
#             faces_tensor = torch.cat(faces).to(device)
#             embeddings = resnet(faces_tensor).detach().cpu().numpy()

#             for i, box in enumerate(boxes):
#                 x1, y1, x2, y2 = [int(b) for b in box]
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(w, x2), min(h, y2)

#                 emb = embeddings[i]
#                 name = "Unknown"
#                 confidence_score = 0.0

#                 if USE_COSINE:
#                     max_sim, best_match = -1, None
#                     for j, known_emb in enumerate(known_embeddings):
#                         sim = cosine_similarity(emb, known_emb)
#                         if sim > max_sim:
#                             max_sim, best_match = sim, names[j]
#                     confidence_score = max_sim
#                     if max_sim >= THRESHOLD:
#                         name = best_match
#                 else:
#                     min_dist, best_match = float('inf'), None
#                     for j, known_emb in enumerate(known_embeddings):
#                         dist = np.linalg.norm(emb - known_emb)
#                         if dist < min_dist:
#                             min_dist, best_match = dist, names[j]
#                     # Convert distance to confidence score (0-1 scale)
#                     # Assuming typical Euclidean distances are between 0-2 for face recognition
#                     confidence_score = max(0, 1 - (min_dist / 2.0))
#                     if confidence_score >= THRESHOLD:
#                         name = best_match

#                 # Determine color based on confidence
#                 if confidence_score >= THRESHOLD:
#                     color = (0, 255, 0)  # Green for recognized
#                 else:
#                     color = (0, 0, 255)  # Red for unknown/low confidence

#                 # Draw box & label + confidence
#                 label = f"{name} ({confidence_score:.2f})"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                 cv2.putText(frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#                 # Attendance logic - only mark if confidence is above threshold
#                 if name != "Unknown" and confidence_score >= THRESHOLD:
#                     detection_counts[name] = detection_counts.get(name, 0) + 1
#                     if detection_counts[name] >= FRAMES_REQUIRED and name not in already_marked:
#                         mark_attendance(name)
#                         detection_counts[name] = 0
#                 else:
#                     # Reset counter for unknown or low confidence detections
#                     if name in detection_counts:
#                         detection_counts[name] = 0

#     # Show FPS and threshold info
#     now = time.time()
#     fps = 1 / (now - fps_time)
#     fps_time = now
    
#     # Display FPS and threshold information
#     cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#     cv2.putText(frame, f"Threshold: {THRESHOLD}", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#     cv2.putText(frame, f"Method: {'Cosine' if USE_COSINE else 'Euclidean'}", (10, 90),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#     cv2.imshow("Recognition - Dahua RTSP", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# stream.release()
# cv2.destroyAllWindows()


import os
import time
import threading
import traceback

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import requests
import mysql.connector

# --------------------------
# Config
# --------------------------
RTSP_URL = "rtsp://admin:Sahyadri%401234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
USE_COSINE = True
THRESHOLD = 0.75
FRAMES_REQUIRED = 10
ATTENDANCE_ENDPOINT = "http://localhost:3000/attendance"
REQUEST_TIMEOUT = 5  # seconds

# --------------------------
# Device & models
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
mtcnn = MTCNN(keep_all=True, device=device)                  # returns aligned tensors
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# --------------------------
# Robust RTSP capture (threaded)
# --------------------------
class RTSPStream:
    def __init__(self, url):
        # try FFMPEG backend first, fallback to default, then to camera 0
        # self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(url)
            if not self.cap.isOpened():
                print("RTSP open failed; falling back to camera 0")
                self.cap = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

# --------------------------
# Load known embeddings from DB (robust)
# --------------------------
def load_known_embeddings():
    names = []
    embs = []
    try:
        conn = mysql.connector.connect(
            host="localhost", user="root", password="password", database="majorproject"
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT s.name, f.embedding FROM face_embeddings f
            JOIN students s ON s.student_id = f.student_id
        """)
        for name, emb_str in cur.fetchall():
            try:
                flat = [float(x) for x in emb_str.strip("[]").split(",") if x.strip()]
                arr = np.array(flat, dtype=np.float32)
                embs.append(arr)
                names.append(name)
            except Exception as e:
                print(f"Skipping bad embedding for {name}: {e}")
        cur.close()
        conn.close()
    except Exception as e:
        print("DB load error:", e)
    if len(embs) == 0:
        return np.empty((0,)), []
    # stack and normalize
    embs = np.vstack(embs)  # (N, D)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    return embs, names

known_embeddings, names = load_known_embeddings()
if known_embeddings.size == 0:
    print("Warning: no known embeddings loaded.")

# --------------------------
# Attendance (async)
# --------------------------
already_marked = set()
detection_counts = {}

def async_mark_attendance(name):
    def _post():
        try:
            r = requests.post(ATTENDANCE_ENDPOINT, json={"name": name}, timeout=REQUEST_TIMEOUT)
            if r.ok:
                already_marked.add(name)
                print("✅ Attendance marked:", name)
            else:
                print("❌ Attendance API failed:", r.status_code)
        except Exception as e:
            print("Attendance post error:", e)
    threading.Thread(target=_post, daemon=True).start()

# --------------------------
# Helper: vectorized matching
# --------------------------
def match_embeddings_batch(embeddings_np, known_embs, names, use_cosine=True):
    # embeddings_np: (n, D) already L2-normalized
    # known_embs: (N, D) already L2-normalized or empty
    n = embeddings_np.shape[0]
    if known_embs.size == 0:
        return ["Unknown"] * n, [ -1.0 ] * n
    if use_cosine:
        sims = np.dot(embeddings_np, known_embs.T)    # (n, N)
        best_idx = np.argmax(sims, axis=1)
        best_sims = sims[np.arange(n), best_idx]
        matched_names = [ names[i] if best_sims[idx] >= THRESHOLD else "Unknown" for idx, i in enumerate(best_idx) ]
        return matched_names, best_sims.tolist()
    else:
        # Euclidean distances vectorized
        # dist^2 = ||a||^2 + ||b||^2 - 2 a.b ; embeddings are normalized so ||a|| ~1, but we compute directly
        dists = np.linalg.norm(embeddings_np[:, None, :] - known_embs[None, :, :], axis=2)  # (n, N)
        best_idx = np.argmin(dists, axis=1)
        best_dists = dists[np.arange(n), best_idx]
        # convert to confidence 0-1 (heuristic)
        confidence = np.clip(1.0 - (best_dists / 2.0), 0.0, 1.0)
        matched_names = [ names[i] if confidence[idx] >= THRESHOLD else "Unknown" for idx, i in enumerate(best_idx) ]
        return matched_names, confidence.tolist()

# --------------------------
# Start stream + main loop
# --------------------------
stream = RTSPStream(RTSP_URL)
cv2.namedWindow("Recognition - Dahua RTSP", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Recognition - Dahua RTSP", 800, 600)

fps_time = time.time()

try:
    while True:
        ret, frame = stream.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get aligned, preprocessed face tensors directly from MTCNN
        try:
            faces = mtcnn(rgb)               # None or tensor (n,3,160,160) in CHW order, float32, normalized
        except Exception as e:
            print("MTCNN error:", e)
            faces = None

        names_detected = []
        confidences = []

        if faces is not None:
            if faces.ndimension() == 3:
                faces = faces.unsqueeze(0)   # (1,3,160,160)
            faces = faces.to(device)
            with torch.no_grad():
                emb_t = resnet(faces)       # (n, 512) on device
            emb_np = emb_t.cpu().numpy()
            # normalize
            norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb_np = emb_np / norms

            # match vectorized
            matched_names, matched_scores = match_embeddings_batch(emb_np, known_embeddings, names, USE_COSINE)
            names_detected = matched_names
            confidences = matched_scores

            # Get boxes for drawing (boxes align with mtcnn(rgb) order)
            boxes, probs = mtcnn.detect(rgb)
            if boxes is None:
                boxes = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(b) for b in box]
                name = names_detected[i] if i < len(names_detected) else "Unknown"
                score = confidences[i] if i < len(confidences) else -1.0
                color = (0,255,0) if score >= THRESHOLD and name != "Unknown" else (0,0,255)
                label = f"{name} ({score:.2f})"
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                cv2.putText(frame, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Attendance counting (non-blocking)
                if name != "Unknown" and score >= THRESHOLD:
                    detection_counts[name] = detection_counts.get(name, 0) + 1
                    if detection_counts[name] >= FRAMES_REQUIRED and name not in already_marked:
                        async_mark_attendance(name)
                        detection_counts[name] = 0
                else:
                    # reset counter for mismatched / unknown
                    if name in detection_counts:
                        detection_counts[name] = 0

        # draw HUD
        now = time.time()
        fps = 1.0 / (now - fps_time) if now != fps_time else 0.0
        fps_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Threshold: {THRESHOLD}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Method: {'Cosine' if USE_COSINE else 'Euclidean'}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Recognition - Dahua RTSP", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass
except Exception:
    traceback.print_exc()
finally:
    stream.release()
    cv2.destroyAllWindows()
