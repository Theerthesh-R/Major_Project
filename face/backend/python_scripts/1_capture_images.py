import cv2
import os
import sys
import time

url = "rtsp://admin:Sahyadri%401234@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

if len(sys.argv) < 2:
    print("Usage: python capture_faces.py <person_name>")
    sys.exit(1)

name = sys.argv[1]
save_dir = os.path.join("dataset", name)
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
#cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot connect to camera stream")
    sys.exit(1)

cv2.namedWindow("Capture Face - Dahua Camera", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Capture Face - Dahua Camera", 800, 600)

print("Press 's' to save an image. Press 'q' to quit.")

count = 0
frame_skip = 10
frame_num = 0
fps_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame")
        time.sleep(0.2)
        continue

    frame_num += 1
    if frame_num % frame_skip != 0:
        continue  # skip some frames to reduce load

    # Preview only (scaled)
    display_frame = cv2.resize(frame, (800, 600))

    now = time.time()
    fps = 1 / (now - fps_time)
    fps_time = now
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display_frame, f"Saved: {count}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Capture Face - Dahua Camera", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        img_path = os.path.join(save_dir, f"{name}_{count}.jpg")
        cv2.imwrite(img_path, frame)  # full-resolution frame saved
        print(f"✅ Saved: {img_path}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
