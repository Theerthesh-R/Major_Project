





# import os
# import time
# import threading
# import traceback
# from datetime import datetime, date
# import json
# import socketserver
# import http.server
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# import cv2
# import numpy as np
# import torch
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import requests
# import mysql.connector

# # --------------------------
# # Config
# # --------------------------
# RTSP_URL = "rtsp://admin:Sahyadri%401234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
# USE_COSINE = True
# THRESHOLD = 0.75
# FRAMES_REQUIRED = 10
# ATTENDANCE_ENDPOINT = "http://localhost:3000/faculty/attendance-auto"
# REQUEST_TIMEOUT = 5

# # # Email Configuration
# # EMAIL_ENABLED = True
# # SMTP_SERVER = "smtp.gmail.com"  # Change based on your email provider
# # SMTP_PORT = 587
# # EMAIL_SENDER = "your_email@gmail.com"  # Replace with your email
# # EMAIL_PASSWORD = "your_app_password"  # Replace with your app password
# # EMAIL_SUBJECT = "Attendance Marked Successfully"


# # Email Configuration - UPDATED with your credentials
# EMAIL_ENABLED = True
# SMTP_SERVER = "smtp.gmail.com"
# SMTP_PORT = 587
# EMAIL_SENDER = "adscem2025@gmail.com"  # From your .env file
# EMAIL_PASSWORD = "lyso zloe qkcl ydqs"   # From your .env file - App Password
# EMAIL_SUBJECT = "Attendance Marked Successfully"

# # Global state
# current_status = "Stopped"
# last_error = ""
# attendance_log = []
# show_camera_window = True

# class FaceRecognitionService:
#     def __init__(self):
#         self.is_running = False
#         self.stream = None
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print("Using device:", self.device)
        
#         # Initialize models
#         self.mtcnn = MTCNN(keep_all=True, device=self.device)
#         self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        
#         # Load embeddings
#         self.known_embeddings, self.names, self.student_ids = self.load_known_embeddings()
#         self.name_to_id = dict(zip(self.names, self.student_ids))
#         self.id_to_name = dict(zip(self.student_ids, self.names))
#         self.id_to_email = self.load_student_emails()
        
#         # Attendance tracking
#         self.attendance_session = AttendanceSession()
        
#     def load_known_embeddings(self):
#         names, embs, student_ids = [], [], []
#         try:
#             conn = mysql.connector.connect(
#                 host="localhost", user="root", password="password", database="majorproject"
#             )
#             cur = conn.cursor()
#             cur.execute("""
#                 SELECT s.student_id, s.name, f.embedding 
#                 FROM face_embeddings f
#                 JOIN students s ON s.student_id = f.student_id
#             """)
#             for student_id, name, emb_str in cur.fetchall():
#                 try:
#                     flat = [float(x) for x in emb_str.strip("[]").split(",") if x.strip()]
#                     arr = np.array(flat, dtype=np.float32)
#                     embs.append(arr)
#                     names.append(name)
#                     student_ids.append(student_id)
#                 except Exception as e:
#                     print(f"Skipping bad embedding for {name}: {e}")
#             cur.close()
#             conn.close()
#         except Exception as e:
#             print("DB load error:", e)
        
#         if len(embs) == 0:
#             return np.empty((0,)), [], []
        
#         embs = np.vstack(embs)
#         norms = np.linalg.norm(embs, axis=1, keepdims=True)
#         norms[norms == 0] = 1.0
#         embs = embs / norms
#         return embs, names, student_ids

#     def load_student_emails(self):
#         """Load student emails from database"""
#         id_to_email = {}
#         try:
#             conn = mysql.connector.connect(
#                 host="localhost", user="root", password="password", database="majorproject"
#             )
#             cur = conn.cursor()
#             cur.execute("SELECT student_id, email FROM students WHERE email IS NOT NULL")
#             for student_id, email in cur.fetchall():
#                 id_to_email[student_id] = email
#             cur.close()
#             conn.close()
#             print(f"📧 Loaded {len(id_to_email)} student emails")
#         except Exception as e:
#             print(f"Error loading student emails: {e}")
#         return id_to_email

#     def start_recognition(self):
#         if self.is_running:
#             return {"status": "already_running", "message": "Recognition is already running"}
        
#         self.is_running = True
#         threading.Thread(target=self._recognition_loop, daemon=True).start()
#         return {"status": "started", "message": "Face recognition started with camera display"}

#     def stop_recognition(self):
#         self.is_running = False
#         if self.stream:
#             self.stream.release()
#         try:
#             cv2.destroyAllWindows()
#         except:
#             pass
#         return {"status": "stopped", "message": "Face recognition stopped"}

#     def _recognition_loop(self):
#         global current_status, last_error
        
#         try:
#             self.stream = RTSPStream(RTSP_URL)
#             current_status = "Running"
            
#             if show_camera_window:
#                 cv2.namedWindow("Face Recognition - Live Camera", cv2.WINDOW_NORMAL)
#                 cv2.resizeWindow("Face Recognition - Live Camera", 800, 600)
            
#             fps_time = time.time()
#             frame_count = 0
#             last_session_check = time.time()

#             while self.is_running:
#                 # Update current subject every 10 seconds
#                 if time.time() - last_session_check > 10:
#                     self.attendance_session.update_current_subject()
#                     last_session_check = time.time()

#                 ret, frame = self.stream.read()
#                 if not ret or frame is None:
#                     time.sleep(0.05)
#                     continue

#                 rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#                 # Face detection
#                 try:
#                     faces = self.mtcnn(rgb)
#                 except Exception as e:
#                     print("MTCNN error:", e)
#                     faces = None

#                 names_detected = []
#                 confidences = []

#                 if faces is not None:
#                     if faces.ndimension() == 3:
#                         faces = faces.unsqueeze(0)
#                     faces = faces.to(self.device)
#                     with torch.no_grad():
#                         emb_t = self.resnet(faces)
#                     emb_np = emb_t.cpu().numpy()
#                     norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
#                     norms[norms == 0] = 1.0
#                     emb_np = emb_np / norms

#                     # Match faces
#                     matched_names, matched_scores = self.match_embeddings_batch(
#                         emb_np, self.known_embeddings, self.names, USE_COSINE
#                     )
#                     names_detected = matched_names
#                     confidences = matched_scores

#                     # Get boxes for drawing
#                     boxes, probs = self.mtcnn.detect(rgb)
#                     if boxes is None:
#                         boxes = []

#                     for i, box in enumerate(boxes):
#                         x1, y1, x2, y2 = [int(b) for b in box]
#                         name = names_detected[i] if i < len(names_detected) else "Unknown"
#                         score = confidences[i] if i < len(confidences) else -1.0
                        
#                         # Determine color and status
#                         if score >= THRESHOLD and name != "Unknown":
#                             if self.attendance_session.can_mark_attendance(name):
#                                 color = (0, 255, 0)  # Green - can mark attendance
#                                 status = "READY"
#                             else:
#                                 color = (255, 255, 0)  # Yellow - already marked today
#                                 status = "MARKED"
#                         else:
#                             color = (0, 0, 255)  # Red - unknown/low confidence
#                             status = "UNKNOWN"
                        
#                         # Draw bounding box and label
#                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
#                         label = f"{name} ({score:.2f}) [{status}]"
#                         cv2.putText(frame, label, (x1, max(15, y1 - 10)), 
#                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#                         # Attendance logic
#                         if (name != "Unknown" and score >= THRESHOLD and 
#                             self.attendance_session.can_mark_attendance(name)):
                            
#                             self.attendance_session.detection_counts[name] = (
#                                 self.attendance_session.detection_counts.get(name, 0) + 1
#                             )
                            
#                             if (self.attendance_session.detection_counts[name] >= FRAMES_REQUIRED):
#                                 # Reset counter immediately to prevent multiple triggers
#                                 self.attendance_session.detection_counts[name] = 0
                                
#                                 # Mark attendance
#                                 self.async_mark_attendance(
#                                     name, 
#                                     self.attendance_session.current_subject['subject_id'],
#                                     self.attendance_session.current_subject['subject_name']
#                                 )
#                         else:
#                             # Reset counter for mismatched / unknown / already marked
#                             if name in self.attendance_session.detection_counts:
#                                 self.attendance_session.detection_counts[name] = 0

#                 # Calculate and display FPS
#                 frame_count += 1
#                 now = time.time()
#                 fps = frame_count / (now - fps_time) if now != fps_time else 0
                
#                 # Draw HUD information on frame
#                 cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 cv2.putText(frame, f"Threshold: {THRESHOLD}", (10, 60), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 cv2.putText(frame, f"Subject: {self.attendance_session.current_subject['subject_name']}", (10, 90), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 120), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 cv2.putText(frame, f"Detected: {len([n for n in names_detected if n != 'Unknown'])}", (10, 150), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
#                 cv2.putText(frame, f"Method: {'Cosine' if USE_COSINE else 'Euclidean'}", (10, 180), 
#                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

#                 # Display the frame
#                 if show_camera_window:
#                     cv2.imshow("Face Recognition - Live Camera", frame)
                
#                 # Check for 'q' key press to stop
#                 if show_camera_window and cv2.waitKey(1) & 0xFF == ord('q'):
#                     self.is_running = False
#                     break

#                 # Reset frame count every second for FPS calculation
#                 if now - fps_time >= 1.0:
#                     frame_count = 0
#                     fps_time = now

#         except Exception as e:
#             last_error = str(e)
#             current_status = f"Error: {last_error}"
#             traceback.print_exc()
#         finally:
#             if self.stream:
#                 self.stream.release()
#             if show_camera_window:
#                 try:
#                     cv2.destroyAllWindows()
#                 except:
#                     pass
#             current_status = "Stopped"
#             self.is_running = False

#     def match_embeddings_batch(self, embeddings_np, known_embs, names, use_cosine=True):
#         n = embeddings_np.shape[0]
#         if known_embs.size == 0:
#             return ["Unknown"] * n, [-1.0] * n
#         if use_cosine:
#             sims = np.dot(embeddings_np, known_embs.T)
#             best_idx = np.argmax(sims, axis=1)
#             best_sims = sims[np.arange(n), best_idx]
#             matched_names = [names[i] if best_sims[idx] >= THRESHOLD else "Unknown" for idx, i in enumerate(best_idx)]
#             return matched_names, best_sims.tolist()
#         else:
#             dists = np.linalg.norm(embeddings_np[:, None, :] - known_embs[None, :, :], axis=2)
#             best_idx = np.argmin(dists, axis=1)
#             best_dists = dists[np.arange(n), best_idx]
#             confidence = np.clip(1.0 - (best_dists / 2.0), 0.0, 1.0)
#             matched_names = [names[i] if confidence[idx] >= THRESHOLD else "Unknown" for idx, i in enumerate(best_idx)]
#             return matched_names, confidence.tolist()

#     def send_attendance_email(self, student_name, student_email, subject_name):
#         """Send email notification to student about attendance"""
#         if not EMAIL_ENABLED:
#             print("📧 Email notifications are disabled")
#             return False
            
#         if not student_email:
#             print(f"❌ No email found for {student_name}")
#             return False
            
#         try:
#             # Create email message
#             message = MIMEMultipart()
#             message["From"] = EMAIL_SENDER
#             message["To"] = student_email
#             message["Subject"] = EMAIL_SUBJECT

#             # Email body
#             body = f"""
#             Dear {student_name},

#             Your attendance has been successfully marked for the subject: {subject_name}

#             Date: {datetime.now().strftime('%Y-%m-%d')}
#             Time: {datetime.now().strftime('%H:%M:%S')}
#             Subject: {subject_name}

#             This is an automated notification. Please do not reply to this email.

#             Best regards,
#             Attendance System
#             """
            
#             message.attach(MIMEText(body, "plain"))

#             # Send email
#             with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#                 server.starttls()
#                 server.login(EMAIL_SENDER, EMAIL_PASSWORD)
#                 server.send_message(message)
            
#             print(f"📧 Email sent successfully to {student_name} ({student_email})")
#             return True
            
#         except Exception as e:
#             print(f"❌ Failed to send email to {student_name}: {e}")
#             return False

#     def async_mark_attendance(self, name, subject_id, subject_name):
#         def _post():
#             try:
#                 student_id = self.name_to_id.get(name)
#                 if not student_id:
#                     error_msg = f"❌ Student ID not found for {name}"
#                     attendance_log.append(error_msg)
#                     print(error_msg)
#                     return
                    
#                 # Check if already marked in our local session before sending to API
#                 attendance_key = f"{student_id}_{self.attendance_session.current_date}_{subject_id}"
#                 if attendance_key in self.attendance_session.already_marked:
#                     print(f"⚠️ Attendance already marked locally for {name} in {subject_name}")
#                     return
                    
#                 # Prepare attendance data
#                 attendance_data = {
#                     "student_id": student_id,
#                     "subject_id": subject_id,
#                     "subject_name": subject_name
#                 }
                
#                 print(f"📤 Sending attendance to {ATTENDANCE_ENDPOINT}: {attendance_data}")
                
#                 # Send to auto-attendance endpoint
#                 r = requests.post(
#                     ATTENDANCE_ENDPOINT, 
#                     json=attendance_data, 
#                     timeout=REQUEST_TIMEOUT,
#                     headers={"Content-Type": "application/json"}
#                 )
                
#                 if r.ok:
#                     data = r.json()
#                     success_msg = f"✅ {datetime.now().strftime('%H:%M:%S')} - {name} marked for {subject_name}"
#                     attendance_log.append(success_msg)
#                     self.attendance_session.mark_attendance_done(name, subject_id)
#                     print(f"✅ Attendance confirmed: {name} for {subject_name}")
                    
#                     # Send email notification after successful attendance marking
#                     student_email = self.id_to_email.get(student_id)
#                     if student_email:
#                         email_thread = threading.Thread(
#                             target=self.send_attendance_email,
#                             args=(name, student_email, subject_name),
#                             daemon=True
#                         )
#                         email_thread.start()
#                     else:
#                         print(f"⚠️ No email found for {name}, skipping email notification")
                        
#                 else:
#                     # Handle "already marked" error gracefully
#                     if r.status_code == 400 and "already marked" in r.text.lower():
#                         print(f"⚠️ Attendance already marked in system for {name} in {subject_name}")
#                         # Still mark it as done in our local session to prevent retries
#                         self.attendance_session.mark_attendance_done(name, subject_id)
#                     else:
#                         error_msg = f"❌ {datetime.now().strftime('%H:%M:%S')} - API Error {r.status_code}: {r.text} for {name}"
#                         attendance_log.append(error_msg)
#                         print(f"❌ API failed: {r.status_code} - {r.text}")
                    
#             except Exception as e:
#                 error_msg = f"❌ {datetime.now().strftime('%H:%M:%S')} - Network Error: {str(e)}"
#                 attendance_log.append(error_msg)
#                 print(f"❌ Network error: {e}")
                
#         threading.Thread(target=_post, daemon=True).start()

#     def get_status(self):
#         return {
#             "status": current_status,
#             "is_running": self.is_running,
#             "error": last_error,
#             "current_subject": self.attendance_session.current_subject,
#             "attendance_log": attendance_log[-10:],
#             "detected_count": len([n for n in self.attendance_session.detection_counts if self.attendance_session.detection_counts[n] > 0]),
#             "email_enabled": EMAIL_ENABLED
#         }

# # ... (Keep the RTSPStream, AttendanceSession, and FaceRecognitionHandler classes exactly the same as in your original code)

# class RTSPStream:
#     def __init__(self, url):
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#             if not self.cap.isOpened():
#                 print("RTSP open failed; falling back to camera 0")
#                 self.cap = cv2.VideoCapture(0)
#         self.lock = threading.Lock()
#         self.ret, self.frame = self.cap.read()
#         self.running = True
#         threading.Thread(target=self.update, daemon=True).start()

#     def update(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 with self.lock:
#                     self.ret, self.frame = ret, frame
#             else:
#                 time.sleep(0.01)

#     def read(self):
#         with self.lock:
#             return self.ret, self.frame

#     def release(self):
#         self.running = False
#         try:
#             self.cap.release()
#         except Exception:
#             pass

# class AttendanceSession:
#     def __init__(self):
#         self.current_date = date.today().isoformat()
#         self.already_marked = set()
#         self.detection_counts = {}
#         self.current_subject = self.get_current_subject()
        
#     def get_current_subject(self):
#         """Get current subject based on timetable - FIXED for case sensitivity"""
#         try:
#             now = datetime.now()
#             current_time = now.strftime("%H:%M:%S")
#             current_day = now.strftime("%A")  # Keep proper case for display
            
#             print(f"🔍 Checking timetable: Day={current_day}, Time={current_time}")
            
#             conn = mysql.connector.connect(
#                 host="localhost", user="root", password="password", database="majorproject"
#             )
#             cur = conn.cursor()
            
#             # FIXED: Check both proper case and lowercase day names
#             cur.execute("""
#                 SELECT t.timetable_id, t.subject_id, s.name, t.day, t.start_time, t.end_time 
#                 FROM timetable t
#                 JOIN subjects s ON t.subject_id = s.subject_id
#                 WHERE (t.day = %s OR LOWER(t.day) = LOWER(%s)) 
#                 AND %s BETWEEN t.start_time AND t.end_time
#                 LIMIT 1
#             """, (current_day, current_day, current_time))
            
#             result = cur.fetchone()
            
#             if result:
#                 timetable_id, subject_id, subject_name, day, start_time, end_time = result
#                 print(f"✅ Found current class: {subject_name} ({start_time} - {end_time})")
#                 cur.close()
#                 conn.close()
#                 return {"subject_id": subject_id, "subject_name": subject_name}
#             else:
#                 print("❌ No current class found in timetable")
#                 cur.close()
#                 conn.close()
#                 return {"subject_id": 0, "subject_name": "No Class"}
                
#         except Exception as e:
#             print(f"❌ Error getting current subject: {e}")
#             return {"subject_id": 0, "subject_name": "Error"}
    
#     def update_current_subject(self):
#         """Update current subject (call this periodically)"""
#         new_subject = self.get_current_subject()
#         if new_subject['subject_id'] != self.current_subject['subject_id']:
#             print(f"🔄 Subject changed: {self.current_subject['subject_name']} -> {new_subject['subject_name']}")
#             self.current_subject = new_subject
#             # Reset detection counts when subject changes
#             self.detection_counts.clear()
    
#     def get_student_id(self, name):
#         try:
#             conn = mysql.connector.connect(
#                 host="localhost", user="root", password="password", database="majorproject"
#             )
#             cur = conn.cursor()
#             cur.execute("SELECT student_id FROM students WHERE name = %s", (name,))
#             result = cur.fetchone()
#             cur.close()
#             conn.close()
#             return result[0] if result else None
#         except Exception as e:
#             print(f"Error getting student ID for {name}: {e}")
#             return None
    
#     def can_mark_attendance(self, name):
#         # Don't mark attendance if no class is scheduled
#         if self.current_subject['subject_id'] == 0:
#             return False
            
#         student_id = self.get_student_id(name)
#         if not student_id:
#             return False
            
#         attendance_key = f"{student_id}_{self.current_date}_{self.current_subject['subject_id']}"
#         return attendance_key not in self.already_marked
    
#     def mark_attendance_done(self, name, subject_id=None):
#         student_id = self.get_student_id(name)
#         if student_id:
#             # Use provided subject_id or fallback to current subject
#             actual_subject_id = subject_id if subject_id is not None else self.current_subject['subject_id']
#             attendance_key = f"{student_id}_{self.current_date}_{actual_subject_id}"
#             self.already_marked.add(attendance_key)
#             print(f"📝 Marked {name} as attended for {self.current_subject['subject_name']} (key: {attendance_key})")
    
#     def reset_if_new_day(self):
#         today = date.today().isoformat()
#         if today != self.current_date:
#             self.current_date = today
#             self.already_marked.clear()
#             self.detection_counts.clear()
#             self.current_subject = self.get_current_subject()
#             print("🔄 New day detected - attendance tracking reset")

# # HTTP API for controlling the service
# class FaceRecognitionHandler(http.server.BaseHTTPRequestHandler):
#     def do_GET(self):
#         if self.path == '/status':
#             self._send_response(service.get_status())
#         else:
#             self._send_response({"error": "Endpoint not found"}, 404)
    
#     def do_POST(self):
#         if self.path == '/start':
#             self._send_response(service.start_recognition())
#         elif self.path == '/stop':
#             self._send_response(service.stop_recognition())
#         else:
#             self._send_response({"error": "Endpoint not found"}, 404)
    
#     def _send_response(self, data, status=200):
#         self.send_response(status)
#         self.send_header('Content-type', 'application/json')
#         self.send_header('Access-Control-Allow-Origin', '*')
#         self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
#         self.send_header('Access-Control-Allow-Headers', 'Content-Type')
#         self.end_headers()
#         self.wfile.write(json.dumps(data).encode())

#     def do_OPTIONS(self):
#         self.send_response(200)
#         self.send_header('Access-Control-Allow-Origin', '*')
#         self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
#         self.send_header('Access-Control-Allow-Headers', 'Content-Type')
#         self.end_headers()

# # Initialize service
# service = FaceRecognitionService()

# if __name__ == "__main__":
#     port = 5000
#     server = socketserver.TCPServer(("", port), FaceRecognitionHandler)
#     print(f"🎯 Face Recognition Service running on port {port}")
#     print("📡 Endpoints:")
#     print("  GET  /status - Get current status")
#     print("  POST /start  - Start face recognition")
#     print("  POST /stop   - Stop face recognition")
#     print(f"\n📋 Using attendance endpoint: {ATTENDANCE_ENDPOINT}")
#     print(f"📚 Current Subject: {service.attendance_session.current_subject['subject_name']}")
#     print(f"📅 Date: {service.attendance_session.current_date}")
#     print(f"📧 Email Notifications: {'ENABLED' if EMAIL_ENABLED else 'DISABLED'}")
#     if EMAIL_ENABLED:
#         print(f"📧 Email Sender: {EMAIL_SENDER}")
#     print("\n📷 Camera window will open when recognition starts.")
#     print("⏹️  Press 'q' in camera window or use Stop button to exit.")
    
#     try:
#         server.serve_forever()
#     except KeyboardInterrupt:
#         print("🛑 Shutting down...")
#         service.stop_recognition()
#         server.shutdown()


import os
import time
import threading
import traceback
from datetime import datetime, date
import json
import socketserver
import http.server
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import requests
import mysql.connector
import mediapipe as mp  # NEW

# --------------------------
# Config
# --------------------------
RTSP_URL = "rtsp://admin:Sahyadri%401234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
USE_COSINE = True
THRESHOLD = 0.75
FRAMES_REQUIRED = 5  # ↓ reduced for faster marking (was 10)
ATTENDANCE_ENDPOINT = "http://localhost:3000/faculty/attendance-auto"
REQUEST_TIMEOUT = 5

# Email Configuration
EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "adscem2025@gmail.com"
EMAIL_PASSWORD = "lyso zloe qkcl ydqs"
EMAIL_SUBJECT = "Attendance Marked Successfully"

# Global state
current_status = "Stopped"
last_error = ""
attendance_log = []
show_camera_window = True


# ==========================
# Blink-based Liveness (tuned)
# ==========================
class BlinkDetector:
    """
    Blink-based liveness detector using MediaPipe FaceMesh + Eye Aspect Ratio (EAR).
    Uses hysteresis and open->closed->open pattern.
    """

    def __init__(self,
                 ear_threshold_closed=0.23,   # relaxed from 0.21
                 ear_threshold_open=0.26,     # relaxed from 0.27
                 consec_closed_frames=2):     # relaxed from 3
        # Below this => considered closed
        self.ear_threshold_closed = ear_threshold_closed
        # Above this => considered open
        self.ear_threshold_open = ear_threshold_open
        # How many consecutive closed frames needed
        self.consec_closed_frames = consec_closed_frames

        self.closed_counter = 0
        self.last_state_open = True  # assume starts open
        self.last_ear = None

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmark indices (MediaPipe FaceMesh)
        self.left_eye = [33, 160, 158, 133, 153, 144]
        self.right_eye = [263, 387, 385, 362, 380, 373]

    def _euclidean_dist(self, p1, p2):
        return np.linalg.norm(np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32))

    def _eye_aspect_ratio(self, eye):
        A = self._euclidean_dist(eye[1], eye[5])
        B = self._euclidean_dist(eye[2], eye[4])
        C = self._euclidean_dist(eye[0], eye[3])
        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    def detect_blink(self, frame):
        """
        Returns True once when a blink is completed (open -> closed -> open),
        otherwise False.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            # No face -> reset state
            self.closed_counter = 0
            self.last_state_open = True
            self.last_ear = None
            return False

        h, w, _ = frame.shape
        mesh = results.multi_face_landmarks[0]

        left = [(int(mesh.landmark[i].x * w), int(mesh.landmark[i].y * h)) for i in self.left_eye]
        right = [(int(mesh.landmark[i].x * w), int(mesh.landmark[i].y * h)) for i in self.right_eye]

        leftEAR = self._eye_aspect_ratio(left)
        rightEAR = self._eye_aspect_ratio(right)
        ear = (leftEAR + rightEAR) / 2.0

        # print("EAR:", ear)  # debug if needed

        blink_detected = False

        # Closed?
        if ear < self.ear_threshold_closed:
            self.closed_counter += 1
        else:
            # Eye NOT closed this frame
            if (self.closed_counter >= self.consec_closed_frames
                    and self.last_state_open is True
                    and ear > self.ear_threshold_open):
                # We saw: open -> N closed frames -> open again => blink
                blink_detected = True

            # reset
            self.closed_counter = 0

            # mark current state as open if clearly open
            if ear > self.ear_threshold_open:
                self.last_state_open = True
            else:
                # in between zone, don't flip state aggressively
                pass

        self.last_ear = ear
        return blink_detected


class FaceRecognitionService:
    def __init__(self):
        self.is_running = False
        self.stream = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        
        # Initialize models
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        
        # Liveness
        self.blink_detector = BlinkDetector()
        self.LIVE_WINDOW = 8.0          # seconds to look back for recent blinks (was 10)
        self.REQUIRED_BLINKS = 1        # only 1 blink required (was 2)
        self.blink_times = []           # timestamps of recent blinks
        
        # Load embeddings
        self.known_embeddings, self.names, self.student_ids = self.load_known_embeddings()
        self.name_to_id = dict(zip(self.names, self.student_ids))
        self.id_to_name = dict(zip(self.student_ids, self.names))
        self.id_to_email = self.load_student_emails()
        
        # Attendance tracking
        self.attendance_session = AttendanceSession()
        
    def load_known_embeddings(self):
        names, embs, student_ids = [], [], []
        try:
            conn = mysql.connector.connect(
                host="localhost", user="root", password="password", database="majorproject"
            )
            cur = conn.cursor()
            cur.execute("""
                SELECT s.student_id, s.name, f.embedding 
                FROM face_embeddings f
                JOIN students s ON s.student_id = f.student_id
            """)
            for student_id, name, emb_str in cur.fetchall():
                try:
                    flat = [float(x) for x in emb_str.strip("[]").split(",") if x.strip()]
                    arr = np.array(flat, dtype=np.float32)
                    embs.append(arr)
                    names.append(name)
                    student_ids.append(student_id)
                except Exception as e:
                    print(f"Skipping bad embedding for {name}: {e}")
            cur.close()
            conn.close()
        except Exception as e:
            print("DB load error:", e)
        
        if len(embs) == 0:
            return np.empty((0,)), [], []
        
        embs = np.vstack(embs)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = embs / norms
        return embs, names, student_ids

    def load_student_emails(self):
        """Load student emails from database"""
        id_to_email = {}
        try:
            conn = mysql.connector.connect(
                host="localhost", user="root", password="password", database="majorproject"
            )
            cur = conn.cursor()
            cur.execute("SELECT student_id, email FROM students WHERE email IS NOT NULL")
            for student_id, email in cur.fetchall():
                id_to_email[student_id] = email
            cur.close()
            conn.close()
            print(f"📧 Loaded {len(id_to_email)} student emails")
        except Exception as e:
            print(f"Error loading student emails: {e}")
        return id_to_email

    def start_recognition(self):
        if self.is_running:
            return {"status": "already_running", "message": "Recognition is already running"}
        
        self.is_running = True
        threading.Thread(target=self._recognition_loop, daemon=True).start()
        return {"status": "started", "message": "Face recognition started with camera display"}

    def stop_recognition(self):
        self.is_running = False
        if self.stream:
            self.stream.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        return {"status": "stopped", "message": "Face recognition stopped"}

    def _recognition_loop(self):
        global current_status, last_error
        
        try:
            self.stream = RTSPStream(RTSP_URL)
            current_status = "Running"
            
            if show_camera_window:
                cv2.namedWindow("Face Recognition - Live Camera", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Face Recognition - Live Camera", 800, 600)
            
            fps_time = time.time()
            frame_count = 0
            last_session_check = time.time()

            while self.is_running:
                # Update current subject every 10 seconds
                if time.time() - last_session_check > 10:
                    self.attendance_session.update_current_subject()
                    last_session_check = time.time()

                ret, frame = self.stream.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                # ------------ Liveness: multi-blink in time window ------------
                now_ts = time.time()
                blink_now = self.blink_detector.detect_blink(frame)

                if blink_now:
                    # store the timestamp of this blink
                    self.blink_times.append(now_ts)

                # keep only blinks in the last LIVE_WINDOW seconds
                self.blink_times = [t for t in self.blink_times if now_ts - t <= self.LIVE_WINDOW]

                # require at least REQUIRED_BLINKS recent blinks to consider face live
                is_live = len(self.blink_times) >= self.REQUIRED_BLINKS

                liveness_status_text = f"LIVE (blinks={len(self.blink_times)})" if is_live else "NOT LIVE"
                liveness_color = (0, 255, 0) if is_live else (0, 0, 255)

                # Debug if needed:
                # print("blink_now:", blink_now, "recent_blinks:", len(self.blink_times), "is_live:", is_live)
                # ---------------------------------------------------------------

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Face detection
                try:
                    faces = self.mtcnn(rgb)
                except Exception as e:
                    print("MTCNN error:", e)
                    faces = None

                names_detected = []
                confidences = []

                if faces is not None:
                    if faces.ndimension() == 3:
                        faces = faces.unsqueeze(0)
                    faces = faces.to(self.device)
                    with torch.no_grad():
                        emb_t = self.resnet(faces)
                    emb_np = emb_t.cpu().numpy()
                    norms = np.linalg.norm(emb_np, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    emb_np = emb_np / norms

                    # Match faces
                    matched_names, matched_scores = self.match_embeddings_batch(
                        emb_np, self.known_embeddings, self.names, USE_COSINE
                    )
                    names_detected = matched_names
                    confidences = matched_scores

                    # Get boxes for drawing
                    boxes, probs = self.mtcnn.detect(rgb)
                    if boxes is None:
                        boxes = []

                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = [int(b) for b in box]
                        name = names_detected[i] if i < len(names_detected) else "Unknown"
                        score = confidences[i] if i < len(confidences) else -1.0
                        
                        # Determine color and status
                        if score >= THRESHOLD and name != "Unknown":
                            if self.attendance_session.can_mark_attendance(name) and is_live:
                                color = (0, 255, 0)  # Green - can mark attendance
                                status = "READY_LIVE"
                            elif self.attendance_session.can_mark_attendance(name) and not is_live:
                                color = (0, 165, 255)  # Orange - known face, not live
                                status = "NOT_LIVE"
                            else:
                                color = (255, 255, 0)  # Yellow - already marked today
                                status = "MARKED"
                        else:
                            color = (0, 0, 255)  # Red - unknown/low confidence
                            status = "UNKNOWN"
                        
                        # Draw bounding box and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{name} ({score:.2f}) [{status}]"
                        cv2.putText(frame, label, (x1, max(15, y1 - 10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Attendance logic (STRICTLY requires liveness)
                        if (name != "Unknown" 
                            and score >= THRESHOLD 
                            and is_live
                            and self.attendance_session.can_mark_attendance(name)):
                            
                            self.attendance_session.detection_counts[name] = (
                                self.attendance_session.detection_counts.get(name, 0) + 1
                            )
                            
                            if (self.attendance_session.detection_counts[name] >= FRAMES_REQUIRED):
                                # Reset counter immediately to prevent multiple triggers
                                self.attendance_session.detection_counts[name] = 0
                                
                                # Mark attendance
                                self.async_mark_attendance(
                                    name, 
                                    self.attendance_session.current_subject['subject_id'],
                                    self.attendance_session.current_subject['subject_name']
                                )
                        else:
                            # Reset counter for mismatched / unknown / already marked
                            if name in self.attendance_session.detection_counts:
                                self.attendance_session.detection_counts[name] = 0

                # Calculate and display FPS
                frame_count += 1
                now = time.time()
                fps = frame_count / (now - fps_time) if now != fps_time else 0
                
                # Draw HUD information on frame
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Threshold: {THRESHOLD}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Subject: {self.attendance_session.current_subject['subject_name']}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 120), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Detected: {len([n for n in names_detected if n != 'Unknown'])}", (10, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Method: {'Cosine' if USE_COSINE else 'Euclidean'}", (10, 180), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Liveness status
                cv2.putText(frame, f"Liveness: {liveness_status_text}", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, liveness_color, 2)

                # Display the frame
                if show_camera_window:
                    cv2.imshow("Face Recognition - Live Camera", frame)
                
                # Check for 'q' key press to stop
                if show_camera_window and cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                    break

                # Reset frame count every second for FPS calculation
                if now - fps_time >= 1.0:
                    frame_count = 0
                    fps_time = now

        except Exception as e:
            last_error = str(e)
            current_status = f"Error: {last_error}"
            traceback.print_exc()
        finally:
            if self.stream:
                self.stream.release()
            if show_camera_window:
                try:
                    cv2.destroyAllWindows()
                except:
                    pass
            current_status = "Stopped"
            self.is_running = False

    def match_embeddings_batch(self, embeddings_np, known_embs, names, use_cosine=True):
        n = embeddings_np.shape[0]
        if known_embs.size == 0:
            return ["Unknown"] * n, [-1.0] * n
        if use_cosine:
            sims = np.dot(embeddings_np, known_embs.T)
            best_idx = np.argmax(sims, axis=1)
            best_sims = sims[np.arange(n), best_idx]
            matched_names = [names[i] if best_sims[idx] >= THRESHOLD else "Unknown" for idx, i in enumerate(best_idx)]
            return matched_names, best_sims.tolist()
        else:
            dists = np.linalg.norm(embeddings_np[:, None, :] - known_embs[None, :, :], axis=2)
            best_idx = np.argmin(dists, axis=1)
            best_dists = dists[np.arange(n), best_idx]
            confidence = np.clip(1.0 - (best_dists / 2.0), 0.0, 1.0)
            matched_names = [names[i] if confidence[idx] >= THRESHOLD else "Unknown" for idx, i in enumerate(best_idx)]
            return matched_names, confidence.tolist()

    def send_attendance_email(self, student_name, student_email, subject_name):
        """Send email notification to student about attendance"""
        if not EMAIL_ENABLED:
            print("📧 Email notifications are disabled")
            return False
            
        if not student_email:
            print(f"❌ No email found for {student_name}")
            return False
            
        try:
            # Create email message
            message = MIMEMultipart()
            message["From"] = EMAIL_SENDER
            message["To"] = student_email
            message["Subject"] = EMAIL_SUBJECT

            # Email body
            body = f"""
            Dear {student_name},

            Your attendance has been successfully marked for the subject: {subject_name}

            Date: {datetime.now().strftime('%Y-%m-%d')}
            Time: {datetime.now().strftime('%H:%M:%S')}
            Subject: {subject_name}

            This is an automated notification. Please do not reply to this email.

            Best regards,
            Attendance System
            """
            
            message.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(message)
            
            print(f"📧 Email sent successfully to {student_name} ({student_email})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email to {student_name}: {e}")
            return False

    def async_mark_attendance(self, name, subject_id, subject_name):
        def _post():
            try:
                student_id = self.name_to_id.get(name)
                if not student_id:
                    error_msg = f"❌ Student ID not found for {name}"
                    attendance_log.append(error_msg)
                    print(error_msg)
                    return
                    
                # Check if already marked in our local session before sending to API
                attendance_key = f"{student_id}_{self.attendance_session.current_date}_{subject_id}"
                if attendance_key in self.attendance_session.already_marked:
                    print(f"⚠️ Attendance already marked locally for {name} in {subject_name}")
                    return
                    
                # Prepare attendance data
                attendance_data = {
                    "student_id": student_id,
                    "subject_id": subject_id,
                    "subject_name": subject_name
                }
                
                print(f"📤 Sending attendance to {ATTENDANCE_ENDPOINT}: {attendance_data}")
                
                # Send to auto-attendance endpoint
                r = requests.post(
                    ATTENDANCE_ENDPOINT, 
                    json=attendance_data, 
                    timeout=REQUEST_TIMEOUT,
                    headers={"Content-Type": "application/json"}
                )
                
                if r.ok:
                    data = r.json()
                    success_msg = f"✅ {datetime.now().strftime('%H:%M:%S')} - {name} marked for {subject_name}"
                    attendance_log.append(success_msg)
                    self.attendance_session.mark_attendance_done(name, subject_id)
                    print(f"✅ Attendance confirmed: {name} for {subject_name}")
                    
                    # Send email notification after successful attendance marking
                    student_email = self.id_to_email.get(student_id)
                    if student_email:
                        email_thread = threading.Thread(
                            target=self.send_attendance_email,
                            args=(name, student_email, subject_name),
                            daemon=True
                        )
                        email_thread.start()
                    else:
                        print(f"⚠️ No email found for {name}, skipping email notification")
                        
                else:
                    # Handle "already marked" error gracefully
                    if r.status_code == 400 and "already marked" in r.text.lower():
                        print(f"⚠️ Attendance already marked in system for {name} in {subject_name}")
                        # Still mark it as done in our local session to prevent retries
                        self.attendance_session.mark_attendance_done(name, subject_id)
                    else:
                        error_msg = f"❌ {datetime.now().strftime('%H:%M:%S')} - API Error {r.status_code}: {r.text} for {name}"
                        attendance_log.append(error_msg)
                        print(f"❌ API failed: {r.status_code} - {r.text}")
                    
            except Exception as e:
                error_msg = f"❌ {datetime.now().strftime('%H:%M:%S')} - Network Error: {str(e)}"
                attendance_log.append(error_msg)
                print(f"❌ Network error: {e}")
                
        threading.Thread(target=_post, daemon=True).start()

    def get_status(self):
        return {
            "status": current_status,
            "is_running": self.is_running,
            "error": last_error,
            "current_subject": self.attendance_session.current_subject,
            "attendance_log": attendance_log[-10:],
            "detected_count": len([n for n in self.attendance_session.detection_counts if self.attendance_session.detection_counts[n] > 0]),
            "email_enabled": EMAIL_ENABLED
        }


class RTSPStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
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


class AttendanceSession:
    def __init__(self):
        self.current_date = date.today().isoformat()
        self.already_marked = set()
        self.detection_counts = {}
        self.current_subject = self.get_current_subject()
        
    def get_current_subject(self):
        """Get current subject based on timetable - FIXED for case sensitivity"""
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            current_day = now.strftime("%A")  # Keep proper case for display
            
            print(f"🔍 Checking timetable: Day={current_day}, Time={current_time}")
            
            conn = mysql.connector.connect(
                host="localhost", user="root", password="password", database="majorproject"
            )
            cur = conn.cursor()
            
            # FIXED: Check both proper case and lowercase day names
            cur.execute("""
                SELECT t.timetable_id, t.subject_id, s.name, t.day, t.start_time, t.end_time 
                FROM timetable t
                JOIN subjects s ON t.subject_id = s.subject_id
                WHERE (t.day = %s OR LOWER(t.day) = LOWER(%s)) 
                AND %s BETWEEN t.start_time AND t.end_time
                LIMIT 1
            """, (current_day, current_day, current_time))
            
            result = cur.fetchone()
            
            if result:
                timetable_id, subject_id, subject_name, day, start_time, end_time = result
                print(f"✅ Found current class: {subject_name} ({start_time} - {end_time})")
                cur.close()
                conn.close()
                return {"subject_id": subject_id, "subject_name": subject_name}
            else:
                print("❌ No current class found in timetable")
                cur.close()
                conn.close()
                return {"subject_id": 0, "subject_name": "No Class"}
                
        except Exception as e:
            print(f"❌ Error getting current subject: {e}")
            return {"subject_id": 0, "subject_name": "Error"}
    
    def update_current_subject(self):
        """Update current subject (call this periodically)"""
        new_subject = self.get_current_subject()
        if new_subject['subject_id'] != self.current_subject['subject_id']:
            print(f"🔄 Subject changed: {self.current_subject['subject_name']} -> {new_subject['subject_name']}")
            self.current_subject = new_subject
            # Reset detection counts when subject changes
            self.detection_counts.clear()
    
    def get_student_id(self, name):
        try:
            conn = mysql.connector.connect(
                host="localhost", user="root", password="password", database="majorproject"
            )
            cur = conn.cursor()
            cur.execute("SELECT student_id FROM students WHERE name = %s", (name,))
            result = cur.fetchone()
            cur.close()
            conn.close()
            return result[0] if result else None
        except Exception as e:
            print(f"Error getting student ID for {name}: {e}")
            return None
    
    def can_mark_attendance(self, name):
        # Don't mark attendance if no class is scheduled
        if self.current_subject['subject_id'] == 0:
            return False
            
        student_id = self.get_student_id(name)
        if not student_id:
            return False
            
        attendance_key = f"{student_id}_{self.current_date}_{self.current_subject['subject_id']}"
        return attendance_key not in self.already_marked
    
    def mark_attendance_done(self, name, subject_id=None):
        student_id = self.get_student_id(name)
        if student_id:
            # Use provided subject_id or fallback to current subject
            actual_subject_id = subject_id if subject_id is not None else self.current_subject['subject_id']
            attendance_key = f"{student_id}_{self.current_date}_{actual_subject_id}"
            self.already_marked.add(attendance_key)
            print(f"📝 Marked {name} as attended for {self.current_subject['subject_name']} (key: {attendance_key})")
    
    def reset_if_new_day(self):
        today = date.today().isoformat()
        if today != self.current_date:
            self.current_date = today
            self.already_marked.clear()
            self.detection_counts.clear()
            self.current_subject = self.get_current_subject()
            print("🔄 New day detected - attendance tracking reset")


# HTTP API for controlling the service
class FaceRecognitionHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self._send_response(service.get_status())
        else:
            self._send_response({"error": "Endpoint not found"}, 404)
    
    def do_POST(self):
        if self.path == '/start':
            self._send_response(service.start_recognition())
        elif self.path == '/stop':
            self._send_response(service.stop_recognition())
        else:
            self._send_response({"error": "Endpoint not found"}, 404)
    
    def _send_response(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# Initialize service
service = FaceRecognitionService()

if __name__ == "__main__":
    port = 5000
    server = socketserver.TCPServer(("", port), FaceRecognitionHandler)
    print(f"🎯 Face Recognition Service running on port {port}")
    print("📡 Endpoints:")
    print("  GET  /status - Get current status")
    print("  POST /start  - Start face recognition")
    print("  POST /stop   - Stop face recognition")
    print(f"\n📋 Using attendance endpoint: {ATTENDANCE_ENDPOINT}")
    print(f"📚 Current Subject: {service.attendance_session.current_subject['subject_name']}")
    print(f"📅 Date: {service.attendance_session.current_date}")
    print(f"📧 Email Notifications: {'ENABLED' if EMAIL_ENABLED else 'DISABLED'}")
    if EMAIL_ENABLED:
        print(f"📧 Email Sender: {EMAIL_SENDER}")
    print("\n📷 Camera window will open when recognition starts.")
    print("⏹️  Press 'q' in camera window or use Stop button to exit.")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
        service.stop_recognition()
        server.shutdown()
