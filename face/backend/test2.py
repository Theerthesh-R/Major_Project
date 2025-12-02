import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import pandas as pd
import time

# -------------------------------
# Settings
# -------------------------------
EMBEDDINGS_DIR = "embeddings"     # folder with saved embeddings (.npy)
TEST_IMAGES_DIR = "test_images"   # folder with test images (subfolders per person)
THRESHOLD = 0.75                   # cosine similarity threshold
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Helpers
# -------------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v

# -------------------------------
# Load FaceNet
# -------------------------------
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

def get_embedding(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR image (H,W,3) to a 512-D FaceNet embedding (float32 numpy).
    Expects a reasonably sized face crop; resizes to 160x160.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160), interpolation=cv2.INTER_LINEAR)
    img_rgb = img_rgb.astype(np.float32)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
    img_tensor = (img_tensor - 127.5) / 128.0
    img_tensor = img_tensor.to(DEVICE)
    with torch.no_grad():
        emb = embedder(img_tensor).detach().cpu().numpy().reshape(-1)
    return emb.astype(np.float32)

# -------------------------------
# Main logic
# -------------------------------
def main():
    # load known embeddings
    if not os.path.isdir(EMBEDDINGS_DIR):
        raise SystemExit(f"Embeddings folder not found: {EMBEDDINGS_DIR}")

    known_embeddings = []
    names = []

    for fn in sorted(os.listdir(EMBEDDINGS_DIR)):
        if not fn.lower().endswith(".npy"):
            continue
        name = os.path.splitext(fn)[0]
        emb = np.load(os.path.join(EMBEDDINGS_DIR, fn))
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim > 1:
            emb = emb.mean(axis=0)           # average multiple embeddings if present
        emb = l2_normalize(emb)
        known_embeddings.append(emb)
        names.append(name)

    if len(names) == 0:
        raise SystemExit("No .npy embeddings found in embeddings folder.")

    known_embeddings = np.stack(known_embeddings, axis=0)  # (N, 512)
    print(f"🔹 Loaded {len(names)} known persons: {names}")

    # name mapping for flexible matching of test-folder names
    name_mapping = {}
    for name in names:
        simplified = name.lower().replace(" ", "").replace("_", "").replace("-", "")
        name_mapping[simplified] = name
    print(f"🔹 Name mapping: {name_mapping}")

    # test images loop
    if not os.path.isdir(TEST_IMAGES_DIR):
        raise SystemExit(f"Test images folder not found: {TEST_IMAGES_DIR}")

    # Warm-up GPU (important for stable timing)
    try:
        dummy = np.zeros((160,160,3), dtype=np.uint8)
        for _ in range(5):
            _ = get_embedding(dummy)
    except Exception:
        pass  # safe to ignore on CPU

    all_true = []
    all_pred = []
    correct = 0
    total = 0

    # latency collectors
    latencies = []         # total pipeline latency per image (ms)
    embed_latencies = []   # embedding-only latency per image (ms)
    latency_rows = []      # per-image records to save

    for person_folder in sorted(os.listdir(TEST_IMAGES_DIR)):
        folder_path = os.path.join(TEST_IMAGES_DIR, person_folder)
        if not os.path.isdir(folder_path):
            continue

        folder_simple = person_folder.lower().replace(" ", "").replace("_", "").replace("-", "")
        person_in_database = folder_simple in name_mapping
        expected_name = name_mapping[folder_simple] if person_in_database else None
        expected_str = expected_name if person_in_database else "Unknown"

        print(f"\n--- Testing {person_folder} ---")
        print(f"Folder '{person_folder}' -> simplified: '{folder_simple}' -> in DB: {person_in_database} -> expected: {expected_str}")

        for img_file in sorted(os.listdir(folder_path)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"⚠️ Could not read {img_path}")
                continue

            # ---------- timing starts ----------
            t0 = time.time()

            t_embed_start = time.time()
            test_emb = get_embedding(img)
            t_embed_end = time.time()

            test_emb = l2_normalize(test_emb)

            # fast cosine: known embeddings already normalized
            sims = known_embeddings.dot(test_emb)  # (N,)
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            predicted_name = names[best_idx] if best_score >= THRESHOLD else "Unknown"

            t1 = time.time()
            # ---------- timing ends ----------

            total_ms = (t1 - t0) * 1000.0
            embed_ms = (t_embed_end - t_embed_start) * 1000.0

            latencies.append(total_ms)
            embed_latencies.append(embed_ms)
            latency_rows.append({
                "person_folder": person_folder,
                "img_file": img_file,
                "total_ms": total_ms,
                "embed_ms": embed_ms,
                "score": best_score,
                "predicted": predicted_name,
                "expected": expected_str
            })

            # correctness
            is_correct = (predicted_name == expected_name) if person_in_database else (predicted_name == "Unknown")
            status = "✅ CORRECT" if is_correct else "❌ WRONG"
            if is_correct:
                correct += 1
            total += 1

            print(f"[{person_folder}] {img_file} → Predicted: {predicted_name}, Expected: {expected_str}, Score: {best_score:.2f} {status} — total: {total_ms:.1f} ms, embed: {embed_ms:.1f} ms")

            all_true.append(expected_str)
            all_pred.append(predicted_name)

    # final stats
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"\n📊 FINAL RESULTS:")
    print(f"Total test images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # latency summary
    if len(latencies) > 0:
        lat_np = np.array(latencies)
        emb_np = np.array(embed_latencies)
        def stats(arr):
            return {
                "avg_ms": float(arr.mean()),
                "median_ms": float(np.median(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max())
            }
        total_stats = stats(lat_np)
        embed_stats = stats(emb_np)

        print("\nLatency summary (ms) — TOTAL pipeline:")
        print(f"  avg: {total_stats['avg_ms']:.2f}, median: {total_stats['median_ms']:.2f}, p95: {total_stats['p95_ms']:.2f}, min: {total_stats['min_ms']:.2f}, max: {total_stats['max_ms']:.2f}")
        print("Embedding-only (ms):")
        print(f"  avg: {embed_stats['avg_ms']:.2f}, median: {embed_stats['median_ms']:.2f}, p95: {embed_stats['p95_ms']:.2f}, min: {embed_stats['min_ms']:.2f}, max: {embed_stats['max_ms']:.2f}")

        # save latency CSV
        lat_df = pd.DataFrame(latency_rows)
        lat_df.to_csv("latencies.csv", index=False)
        print("\nSaved latency details to latencies.csv")
    else:
        print("\nNo latency measurements collected (no images processed).")

    # confusion matrix: put Unknown last for readability
    label_set = set(all_true) | set(all_pred)
    labels = sorted(label_set - {"Unknown"})
    if "Unknown" in label_set:
        labels.append("Unknown")

    cm = confusion_matrix(all_true, all_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(df_cm.to_string())

    # per-class precision/recall/f1
    p, r, f1, sup = precision_recall_fscore_support(all_true, all_pred, labels=labels, zero_division=0)
    metrics_df = pd.DataFrame({
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": sup
    }, index=labels)
    print("\nPer-class metrics:")
    print(metrics_df.to_string())

    # save outputs
    df_cm.to_csv("confusion_matrix.csv")
    metrics_df.to_csv("per_class_metrics.csv")
    print("\nSaved: confusion_matrix.csv, per_class_metrics.csv")

if __name__ == "__main__":
    main()
