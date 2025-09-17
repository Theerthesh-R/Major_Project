import numpy as np
import mysql.connector
import os
import sys

# ✅ Get name argument from command line
if len(sys.argv) < 2:
    print("❌ Name argument missing.")
    exit()
name = sys.argv[1]

# ✅ Build correct absolute path to embeddings/<name>.npy (one level up from script)
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "embeddings", f"{name}.npy")
file_path = os.path.abspath(file_path)  # Normalize the full path

# ✅ Check if file exists
if not os.path.exists(file_path):
    print(f"❌ Embedding file not found at: {file_path}")
    exit()

# ✅ Load the .npy embedding
embedding = np.load(file_path)
embedding_str = np.array2string(embedding, separator=',')

# ✅ Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="final_db_for_project"
)
cursor = conn.cursor()

# ✅ Get student_id from students table
cursor.execute("SELECT student_id FROM students WHERE name = %s", (name,))
result = cursor.fetchone()

if result:
    student_id = result[0]

    # ✅ Insert into face_embeddings
    cursor.execute(
        "INSERT INTO face_embeddings (student_id, name, embedding) VALUES (%s, %s, %s)",
        (student_id, name, embedding_str)
    )
    conn.commit()
    print(f"✅ Embedding inserted for {name} (student_id: {student_id})")

else:
    print(f"❌ No student found with name '{name}'. Please register the student first.")

cursor.close()
conn.close()
