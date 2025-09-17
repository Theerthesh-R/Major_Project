const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const path = require('path');
const jwt = require('jsonwebtoken');

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, '../frontend')));

const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'password',
    database: 'final_db_for_project'
});

db.connect(err => {
    if (err) throw err;
    console.log("✅ MySQL Connected!");
});


function authenticateFaculty(req, res, next) {
  const authHeader = req.headers["authorization"];
  if (!authHeader) return res.sendStatus(401);

  const token = authHeader.split(" ")[1];
  if (!token) return res.sendStatus(401);

  jwt.verify(token, "secretkey", (err, user) => {   // 🔥 use same secret as /login
    if (err) return res.sendStatus(403);

    if (user.role !== "faculty") {
      return res.status(403).json({ error: "Access denied: Not a faculty" });
    }

    req.user = user; // req.user.faculty_id is now available
    next();
  });
}

// ---------------- AUTH ----------------
function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) return res.sendStatus(401);

    jwt.verify(token, "secretkey", (err, user) => {
        if (err) return res.sendStatus(403);
        req.user = user;
        next();
    });
}

// ---------------- LOGIN ----------------
app.post("/login", (req, res) => {
    const { username, password } = req.body;
    if (!username || !password)
        return res.status(400).json({ error: "Username and password required" });

    const sql = "SELECT * FROM users WHERE username=? AND password=?";
    db.query(sql, [username, password], (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        if (results.length === 0) return res.status(401).json({ error: "Invalid credentials" });

        const user = results[0];
        const token = jwt.sign({
            id: user.user_id,
            role: user.role,
            student_id: user.student_id,
            faculty_id: user.faculty_id,
            username: user.username
        }, "secretkey", { expiresIn: "2h" });

        res.json({ token, role: user.role, username: user.username });
    });
});


app.post('/attendance', async (req, res) => {
    try {
        const { name } = req.body;

        // 1. Find student_id
        const [studentRows] = await db.promise().query(
            "SELECT student_id FROM students WHERE name = ?",
            [name]
        );

        if (studentRows.length === 0) {
            console.warn("⚠️ Student not found:", name);
            return res.status(404).send("Student not found");
        }

        const studentId = studentRows[0].student_id;

        // 2. Find current subject_id from timetable
        const [rows] = await db.promise().query(
        `SELECT s.subject_id, s.name AS subject_name
        FROM timetable t
        JOIN subjects s ON t.subject_id = s.subject_id
        WHERE t.day = DAYNAME(NOW())
        AND TIME(NOW()) BETWEEN t.start_time AND t.end_time
        LIMIT 1`
        );


        if (rows.length === 0) {
            console.warn("⚠️ No subject found for current time");
            return res.status(404).send("No subject at this time");
        }

        const subjectId = rows[0].subject_id;
        const subjectName = rows[0].subject_name;

        // 3. Prevent duplicate marking for same subject & day
        const [check] = await db.promise().query(
            `SELECT * FROM attendance
             WHERE student_id = ? 
             AND subject_id = ?
             AND DATE(timestamp) = CURDATE()`,
            [studentId, subjectId]
        );

        if (check.length > 0) {
            console.log(`⚠️ Attendance already marked for ${name} in ${subjectName}`);
            return res.status(400).send("Attendance already marked for this subject today");
        }

        // 4. Insert attendance
        await db.promise().query(
                "INSERT INTO attendance (student_id, subject_id, subject_name) VALUES (?, ?, ?)",
                [studentId, subjectId, subjectName]
                );


        console.log(`✅ Attendance marked for ${name} (ID: ${studentId}, Subject: ${subjectName})`);
        res.send({ status: "success", student_id: studentId, subject: subjectName });

    } catch (err) {
        console.error("❌ Error in attendance route:", err);
        res.status(500).send("Internal Server Error");
    }
});


// ✅ GET /attendance — Fetch logs
app.get('/attendance', async (req, res) => {
    try {
        const [rows] = await db.promise().query(
            `SELECT a.attendance_id,
                    st.name AS student_name,
                    s.name AS subject_name,
                    a.timestamp
             FROM attendance a
             JOIN students st ON a.student_id = st.student_id
             JOIN subjects s ON a.subject_id = s.subject_id
             ORDER BY a.timestamp DESC`
        );

        res.json(rows);
    } catch (err) {
        console.error("❌ Error fetching attendance:", err);
        res.status(500).send("Internal Server Error");
    }
});




app.get("/student/attendance", authenticateToken, (req, res) => {
    if (req.user.role !== "student") {
        return res.status(403).json({ error: "Forbidden" });
    }

    const sql = `
        SELECT a.attendance_id,
               st.student_id,
               st.name AS student_name,
               s.name AS subject_name,
               a.timestamp
        FROM attendance a
        JOIN students st ON a.student_id = st.student_id
        JOIN subjects s ON a.subject_id = s.subject_id
        WHERE st.student_id = ?
        ORDER BY a.timestamp DESC
    `;

    db.query(sql, [req.user.student_id], (err, results) => {
        if (err) {
            console.error("❌ Error fetching attendance:", err);
            return res.status(500).json({ error: "Database error" });
        }
        res.json(results);
    });
});


// ---------------- FACULTY ROUTES ----------------
// Get all students
app.get('/faculty/students', authenticateToken, (req, res) => {
    if (req.user.role !== 'faculty') return res.status(403).json({ error: "Forbidden" });

    db.query("SELECT * FROM students", (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(results);
    });
});

// Add student
app.post('/faculty/students', authenticateToken, (req, res) => {
    if (req.user.role !== 'faculty') return res.status(403).json({ error: "Forbidden" });

    const { name, roll_number, class_name, email, phone } = req.body;
    const sql = "INSERT INTO students (name, roll_number, class, email, phone) VALUES (?, ?, ?, ?, ?)";
    db.query(sql, [name, roll_number, class_name, email, phone], (err, result) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ message: "Student added", student_id: result.insertId });
    });
});

// Update student
app.put('/faculty/students/:id', authenticateToken, (req, res) => {
    if (req.user.role !== 'faculty') return res.status(403).json({ error: "Forbidden" });

    const id = req.params.id;
    const { name, roll_number, class_name, email, phone } = req.body;
    const sql = "UPDATE students SET name=?, roll_number=?, class=?, email=?, phone=? WHERE student_id=?";
    db.query(sql, [name, roll_number, class_name, email, phone, id], (err, result) => {
        if (err) return res.status(500).json({ error: err.message });
        if (result.affectedRows === 0) return res.status(404).json({ error: "Student not found" });
        res.json({ message: "Student updated" });
    });
});

// Delete student
app.delete('/faculty/students/:id', authenticateToken, (req, res) => {
    if (req.user.role !== 'faculty') return res.status(403).json({ error: "Forbidden" });

    const id = req.params.id;
    const sql = "DELETE FROM students WHERE student_id=?";
    db.query(sql, [id], (err, result) => {
        if (err) return res.status(500).json({ error: err.message });
        if (result.affectedRows === 0) return res.status(404).json({ error: "Student not found" });
        res.json({ message: "Student deleted" });
    });
});

app.get("/faculty/attendance", authenticateFaculty, (req, res) => {
  const facultyId = req.user.faculty_id;  // 🔥 use faculty_id, not id

  const sql = `
    SELECT a.attendance_id, a.student_id, s.name AS student_name,
           a.subject_id, sub.name AS subject_name, a.timestamp
    FROM attendance a
    JOIN students s ON a.student_id = s.student_id
    JOIN subjects sub ON a.subject_id = sub.subject_id
    WHERE sub.faculty_id = ? 
    ORDER BY a.timestamp DESC
  `;

  db.query(sql, [facultyId], (err, results) => {
    if (err) return res.status(500).json({ error: "Database error" });
    res.json(results);
  });
});


// Add attendance
app.post("/faculty/attendance", authenticateToken, (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });

  const { student_id, subject_id, subject_name } = req.body;
  if (!student_id || !subject_id || !subject_name) {
    return res.status(400).json({ error: "Student ID, Subject ID, and Subject Name are required" });
  }

  const sql = `
    INSERT INTO attendance (student_id, subject_id, subject_name)
    VALUES (?, ?, ?)
  `;
  db.query(sql, [student_id, subject_id, subject_name], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ message: "Attendance added", attendance_id: result.insertId });
  });
});

// Update attendance
app.put("/faculty/attendance/:id", authenticateToken, (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });

  const { id } = req.params;
  const { student_id, subject_id, subject_name, timestamp } = req.body;

  const sql = `
    UPDATE attendance 
    SET student_id=?, subject_id=?, subject_name=?, timestamp=?
    WHERE attendance_id=?
  `;
  db.query(sql, [student_id, subject_id, subject_name, timestamp, id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (result.affectedRows === 0) return res.status(404).json({ error: "Attendance not found" });
    res.json({ message: "Attendance updated" });
  });
});

// Delete attendance
app.delete("/faculty/attendance/:id", authenticateToken, (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });

  const { id } = req.params;
  const sql = "DELETE FROM attendance WHERE attendance_id=?";
  db.query(sql, [id], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (result.affectedRows === 0) return res.status(404).json({ error: "Attendance not found" });
    res.json({ message: "Attendance deleted" });
  });
});


// Student self-registration
app.post("/students", authenticateToken, (req, res) => {
    if (req.user.role !== "student") {
        return res.status(403).json({ error: "Only students can self-register" });
    }

    const { name, roll_number, class_name, email, phone } = req.body;

    const sql = `INSERT INTO students (name, roll_number, class, email, phone) VALUES (?, ?, ?, ?, ?)`;
    db.query(sql, [name, roll_number, class_name, email, phone], (err, result) => {
        if (err) return res.status(500).json({ error: err.message });

        res.json({
            message: "✅ Student registered successfully!",
            student_id: result.insertId
        });
    });
});

// Create user login first (no student_id yet)
app.post("/register-user", (req, res) => {
    const { username, password } = req.body;

    const sql = "INSERT INTO users (username, password, role) VALUES (?, ?, 'student')";
    db.query(sql, [username, password], (err, result) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json({ message: "User created, please login and complete registration", user_id: result.insertId });
    });
});

// Register Student
app.post("/api/students", (req, res) => {
    const { name, roll_number, class_name, email, phone, username } = req.body;

    // Insert into students table
    const studentSql = `
        INSERT INTO students (name, roll_number, class, email, phone, registered_on)
        VALUES (?, ?, ?, ?, ?, NOW())
    `;
    db.query(studentSql, [name, roll_number, class_name, email, phone], (err, result) => {
        if (err) return res.status(500).json({ error: err.message });

        const newStudentId = result.insertId; // <-- newly generated student_id

        // Update users table (where student_id is NULL and username matches)
        const updateUserSql = `
            UPDATE users
            SET student_id = ?
            WHERE username = ? AND student_id IS NULL
        `;
        db.query(updateUserSql, [newStudentId, username], (err2) => {
            if (err2) return res.status(500).json({ error: err2.message });

            res.json({ 
                success: true, 
                message: "Student registered and linked to user successfully!",
                student_id: newStudentId
            });
        });
    });
});

// ✅ Student Timetable Route
app.get('/student/timetable', authenticateToken, (req, res) => {
    if (req.user.role !== 'student') return res.status(403).json({ error: "Forbidden" });

    const sql = `
        SELECT t.timetable_id, t.day, t.start_time, t.end_time, s.name AS subject
        FROM student_timetable st
        JOIN timetable t ON st.timetable_id = t.timetable_id
        JOIN subjects s ON t.subject_id = s.subject_id
        WHERE st.student_id = ?
        ORDER BY FIELD(t.day, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
                 t.start_time;
    `;

    db.query(sql, [req.user.student_id], (err, results) => {
        if (err) return res.status(500).json({ error: err.message });
        res.json(results);
    });
});

// Get timetable for logged-in faculty
// Get timetable for logged-in faculty
app.get("/faculty/timetable", authenticateFaculty, (req, res) => {
  const facultyId = req.user.faculty_id; // comes from JWT

  const sql = `
    SELECT t.timetable_id, t.subject_id, s.name AS subject_name, 
           t.day, t.start_time, t.end_time
    FROM timetable t
    JOIN subjects s ON t.subject_id = s.subject_id
    WHERE s.faculty_id = ?
    ORDER BY FIELD(t.day, 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
             t.start_time
  `;

  db.query(sql, [facultyId], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(results);
  });
});

// Add timetable entry (restricted to logged-in faculty)
app.post("/faculty/timetable", authenticateFaculty, (req, res) => {
  const facultyId = req.user.faculty_id;
  const { subject_id, day, start_time, end_time } = req.body;

  // Validate subject belongs to faculty
  const checkSql = "SELECT * FROM subjects WHERE subject_id = ? AND faculty_id = ?";
  db.query(checkSql, [subject_id, facultyId], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    if (results.length === 0) return res.status(403).json({ error: "Not authorized for this subject" });

    // Now insert timetable row
    const sql = `INSERT INTO timetable (subject_id, day, start_time, end_time) VALUES (?, ?, ?, ?)`;
    db.query(sql, [subject_id, day, start_time, end_time], (err2) => {
      if (err2) return res.status(500).json({ error: err2.message });
      res.json({ message: "Timetable entry added successfully" });
    });
  });
});

// Update timetable entry (only if faculty owns it)
app.put("/faculty/timetable/:id", authenticateFaculty, (req, res) => {
  const facultyId = req.user.faculty_id;
  const timetableId = req.params.id;
  const { day, start_time, end_time } = req.body;

  const sql = `
    UPDATE timetable 
    SET day = ?, start_time = ?, end_time = ?
    WHERE timetable_id = ? AND faculty_id = ?
  `;

  db.query(sql, [day, start_time, end_time, timetableId, facultyId], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (result.affectedRows === 0) return res.status(403).json({ error: "Not authorized" });
    res.json({ message: "Timetable updated successfully" });
  });
});

// Delete timetable entry (only if faculty owns it)
app.delete("/faculty/timetable/:id", authenticateFaculty, (req, res) => {
  const facultyId = req.user.faculty_id;
  const timetableId = req.params.id;

  const sql = `DELETE FROM timetable WHERE timetable_id = ? AND faculty_id = ?`;

  db.query(sql, [timetableId, facultyId], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (result.affectedRows === 0) return res.status(403).json({ error: "Not authorized" });
    res.json({ message: "Timetable deleted successfully" });
  });
});



// ---------------- FULL PIPELINE ----------------
app.get("/pipeline/:name", (req, res) => {
    const name = req.params.name;
    const scripts = [
        "1_capture_images.py",
        "2_crop_faces.py",
        "3_generate_embeddings.py",
        "insert_embedding.py"
    ];

    let outputLog = "";

    function runNext(i) {
        if (i >= scripts.length) {
            return res.json({ message: "✅ Pipeline completed", log: outputLog });
        }

        const scriptPath = path.join(__dirname, "python_scripts", scripts[i]);
        const command = `python3 "${scriptPath}" "${name}"`;

        exec(command, (err, stdout, stderr) => {
            if (err) {
                console.error(`❌ Error running ${scripts[i]}:`, stderr);
                return res.status(500).json({
                    error: `Failed at ${scripts[i]}`,
                    details: stderr.toString()
                });
            }

            console.log(`✅ Ran ${scripts[i]} for ${name}`);
            outputLog += `\n----- ${scripts[i]} -----\n${stdout || stderr}\n`;

            runNext(i + 1);
        });
    }

    runNext(0);
});


// ---------------- FACE RECOGNITION ----------------
app.get("/face-recognition", (req, res) => {
    const scriptPath = path.join(__dirname, "python_scripts", "4_face_recognition.py");
    const command = `python3 "${scriptPath}"`;

    exec(command, (err, stdout, stderr) => {
        if (err) {
            console.error(`❌ Error running face recognition:`, stderr);
            return res.status(500).json({ error: stderr.toString() });
        }

        console.log("✅ Face recognition executed");
        res.json({ output: stdout || stderr });
    });
});


// ---------------- CATCH ALL ----------------
app.use((req, res) => res.status(404).json({ error: "Route not found" }));

// ---------------- START SERVER ----------------
app.listen(port, () => console.log(`🚀 Server running at http://localhost:${port}`));
