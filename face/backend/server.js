const express = require('express');
const mysql = require('mysql2');
const cors = require('cors');
const bodyParser = require('body-parser');
const { exec } = require('child_process');
const path = require('path');
const jwt = require('jsonwebtoken');
const bcrypt = require("bcryptjs");

const app = express();
const port = 3000;

const JWT_SECRET = "super_secret_key";   // ✅ use ONE secret everywhere

// ---------------- MIDDLEWARE ----------------
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, '../frontend')));

// ---------------- DATABASE ----------------
const db = mysql.createConnection({
    host: 'localhost',
    user: 'root',
    password: 'password',
    database: 'majorproject'
});

db.connect(err => {
    if (err) throw err;
    console.log("✅ MySQL Connected!");
});

// ---------------- AUTH MIDDLEWARE ----------------
function authenticateFaculty(req, res, next) {
  const authHeader = req.headers["authorization"];
  if (!authHeader) return res.sendStatus(401);

  const token = authHeader.split(" ")[1];
  if (!token) return res.sendStatus(401);

  jwt.verify(token, JWT_SECRET, (err, user) => {   // ✅ fixed secret usage
    if (err) return res.sendStatus(403);

    if (user.role !== "faculty") {
      return res.status(403).json({ error: "Access denied: Not a faculty" });
    }

    req.user = user; 
    next();
  });
}

function authenticateToken(req, res, next) {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];
    if (!token) return res.sendStatus(401);

    jwt.verify(token, JWT_SECRET, (err, user) => {   // ✅ fixed secret usage
        if (err) return res.sendStatus(403);
        req.user = user;
        next();
    });
}

function isAdmin(req, res, next) {
  if (req.user && req.user.role === "admin") {
    return next();
  }
  return res.status(403).json({ error: "Access denied. Admins only." });
}

function authenticateStudent(req, res, next) {
  const token = req.headers["authorization"];
  if (!token) return res.status(401).json({ message: "No token provided" });

  try {
    const decoded = jwt.verify(token.split(" ")[1], JWT_SECRET); // ✅ fixed
    if (decoded.role !== "student") {
      return res.status(403).json({ message: "Access denied" });
    }
    req.user = decoded; 
    next();
  } catch (err) {
    console.error(err);
    res.status(401).json({ message: "Invalid token" });
  }
}

// ---------------- LOGIN ----------------
app.post("/login", (req, res) => {
  const { username, password } = req.body;

  db.query("SELECT * FROM users WHERE username = ?", [username], (err, results) => {
    if (err) return res.status(500).json({ message: "DB error" });
    if (results.length === 0) return res.status(401).json({ message: "Invalid credentials" });

    const user = results[0];
    const storedPassword = user.password;

    function sendLoginResponse() {
      const token = jwt.sign(
        { 
          user_id: user.user_id, 
          role: user.role, 
          student_id: user.student_id, 
          faculty_id: user.faculty_id 
        },
        JWT_SECRET,
        { expiresIn: "1h" }
      );
      
      // ✅ CORRECTED RESPONSE - Include all fields
      res.json({ 
        message: "Login successful", 
        token,
        role: user.role, 
        userId: user.user_id,
        student_id: user.student_id,    // ✅ Added
        faculty_id: user.faculty_id     // ✅ Added
      });
    }

    if (storedPassword.startsWith("$2b$")) {
      bcrypt.compare(password, storedPassword, (err, isMatch) => {
        if (err) return res.status(500).json({ message: "Error comparing passwords" });
        if (!isMatch) return res.status(401).json({ message: "Invalid credentials" });
        sendLoginResponse();
      });
    } else {
      if (password !== storedPassword) {
        return res.status(401).json({ message: "Invalid credentials" });
      }
      sendLoginResponse();
    }
  });
});

// ================== Attendance Routes ==================

// Student marks their attendance (with subject check)
require('dotenv').config();
const sendMail = require("./mailer");

app.post("/attendance", async (req, res) => {
  try {
    const { name } = req.body;

    // 1. Find student info
    const [studentRows] = await db.promise().query(
      "SELECT student_id, name, email FROM students WHERE name = ?",
      [name]
    );

    if (studentRows.length === 0) {
      console.warn("⚠️ Student not found:", name);
      return res.status(404).send("Student not found");
    }

    const studentId = studentRows[0].student_id;
    const studentName = studentRows[0].name;
    const studentEmail = studentRows[0].email;

    // 2. Find subject from timetable
    const [rows] = await db.promise().query(
      `SELECT s.subject_id, s.name AS subject_name
       FROM timetable t
       JOIN subjects s ON t.subject_id = s.subject_id
       JOIN student_timetable st ON st.timetable_id = t.timetable_id
       WHERE st.student_id = ?
       AND LOWER(t.day) IN (LOWER(DAYNAME(NOW())), LOWER(DATE_FORMAT(NOW(), '%a')))
       AND TIME(NOW()) BETWEEN t.start_time AND ADDTIME(t.end_time, '00:05:00')
       LIMIT 1`,
      [studentId]
    );

    if (rows.length === 0) {
      console.warn("⚠️ No subject found for current time or student not enrolled");
      return res.status(404).send("No subject at this time or not enrolled");
    }

    const subjectId = rows[0].subject_id;
    const subjectName = rows[0].subject_name;

    // 3. Prevent duplicate marking
    const [check] = await db.promise().query(
      `SELECT * FROM attendance
       WHERE student_id = ? 
       AND subject_id = ?
       AND DATE(timestamp) = CURDATE()`,
      [studentId, subjectId]
    );

    if (check.length > 0) {
      console.log(`⚠️ Attendance already marked for ${studentName} in ${subjectName}`);
      return res.status(400).send("Attendance already marked for this subject today");
    }

    // 4. Insert attendance
    await db.promise().query(
      "INSERT INTO attendance (student_id, subject_id, subject_name) VALUES (?, ?, ?)",
      [studentId, subjectId, subjectName]
    );

    const timestamp = new Date().toLocaleString();
    console.log(`✅ Attendance marked for ${studentName} (ID: ${studentId}, Subject: ${subjectName})`);

    // 5. Send email only if student has email
    if (!studentEmail) {
      console.warn(`⚠️ No email for student ${studentName} (ID ${studentId}) — skipping mail.`);
    } else {
      await sendMail(
        studentEmail,
        "📅 Attendance Marked",
        `Your attendance for ${subjectName} has been marked.`,
        `<h3>Hello ${studentName},</h3>
         <p>Your attendance has been marked successfully.</p>
         <p><b>Subject:</b> ${subjectName}<br>
            <b>Time:</b> ${timestamp}</p>`
      );
    }

    res.send({
      status: "success",
      student_id: studentId,
      subject: subjectName,
      timestamp,
    });

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

// ================== AUTO ATTENDANCE (For Face Recognition Service) ==================
// This endpoint doesn't require authentication for the face recognition service
app.post("/faculty/attendance-auto", async (req, res) => {
  try {
    const { student_id, subject_id, subject_name } = req.body;
    
    console.log("📥 Auto-attendance request received:", { student_id, subject_id, subject_name });
    
    if (!student_id || !subject_id || !subject_name) {
      return res.status(400).json({ 
        error: "All fields required: student_id, subject_id, subject_name" 
      });
    }

    // Check if student exists
    const [studentRows] = await db.promise().query(
      "SELECT student_id, name FROM students WHERE student_id = ?",
      [student_id]
    );

    if (studentRows.length === 0) {
      return res.status(404).json({ error: "Student not found" });
    }

    const studentName = studentRows[0].name;

    // Check if attendance already exists for today
    const today = new Date().toISOString().split('T')[0];
    const [existing] = await db.promise().query(
      'SELECT * FROM attendance WHERE student_id = ? AND DATE(timestamp) = ? AND subject_id = ?',
      [student_id, today, subject_id]
    );
    
    if (existing.length > 0) {
      console.log(`⚠️ Attendance already marked for ${studentName} in ${subject_name} today`);
      return res.status(400).json({ 
        error: 'Attendance already marked for this subject today',
        student_name: studentName
      });
    }
    
    // Insert new attendance record
    await db.promise().query(
      'INSERT INTO attendance (student_id, subject_id, subject_name, timestamp) VALUES (?, ?, ?, NOW())',
      [student_id, subject_id, subject_name]
    );
    
    console.log(`✅ Auto-attendance marked: ${studentName} (ID: ${student_id}) for ${subject_name}`);
    
    res.json({ 
      message: `Attendance marked successfully for ${studentName} in ${subject_name}`,
      student_name: studentName,
      timestamp: new Date().toLocaleString()
    });
    
  } catch (error) {
    console.error('❌ Auto-attendance error:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


// ---------------- STUDENT ROUTES ----------------
// Get student attendance (student)
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
app.post("/faculty/students", authenticateToken, (req, res) => {
  if (req.user.role !== "faculty")
    return res.status(403).json({ error: "Forbidden" });

  const { name, roll_number, class_name, email, phone } = req.body;

  const sql =
    "INSERT INTO students (name, roll_number, class, email, phone) VALUES (?, ?, ?, ?, ?)";
  db.query(sql, [name, roll_number, class_name, email, phone], async (err, result) => {
    if (err) return res.status(500).json({ error: err.message });

    const newStudentId = result.insertId;

    // ✅ Send welcome email
    try {
      await sendMail(
        email,
        "🎓 Student Registration - Smart Attendance System",
        `Hello ${name},\n\nYou have been successfully registered by your faculty in the Smart Attendance System.\n\nDetails:\n- Roll Number: ${roll_number}\n- Class: ${class_name}\n- Student ID: ${newStudentId}\n\nYou can now use your account to mark attendance.\n\nBest Regards,\nFaculty Team`,
        `<h2>Welcome, ${name}! 🎓</h2>
         <p>Your faculty has registered you in the <b>Smart Attendance System</b>.</p>
         <p>
           <b>Roll Number:</b> ${roll_number}<br>
           <b>Class:</b> ${class_name}<br>
           <b>Student ID:</b> ${newStudentId}
         </p>
         <p>You can now use your account to mark attendance.</p>
         <br>
         <p>Best Regards,<br><b>Faculty Team</b></p>`
      );

      console.log(`📧 Registration email sent to ${email}`);
    } catch (mailErr) {
      console.error("❌ Failed to send registration email:", mailErr.message);
    }

    res.json({
      message: "Student added & email sent",
      student_id: newStudentId,
    });
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


app.get("/faculty/assigned-students", authenticateToken, (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });

  const facultyId = req.user.faculty_id;

  const sql = `
    SELECT fs.id, s.student_id, s.name, s.roll_number, s.class, s.email, s.phone, fs.assigned_on
    FROM faculty_students fs
    JOIN students s ON fs.student_id = s.student_id
    WHERE fs.faculty_id = ?
  `;
  db.query(sql, [facultyId], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(results);
  });
});

app.post("/faculty/assigned-students", authenticateToken, (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });

  const facultyId = req.user.faculty_id;
  const { student_id } = req.body;

  const sql = "INSERT INTO faculty_students (faculty_id, student_id) VALUES (?, ?)";
  db.query(sql, [facultyId, student_id], (err, result) => {
    if (err) {
      if (err.code === "ER_DUP_ENTRY") {
        return res.status(400).json({ error: "Student already assigned" });
      }
      return res.status(500).json({ error: err.message });
    }

    // ✅ Assign all subject timetables of this faculty
    const assignSql = `
      INSERT INTO student_timetable (student_id, timetable_id)
      SELECT ?, t.timetable_id
      FROM timetable t
      JOIN subjects sub ON t.subject_id = sub.subject_id
      WHERE sub.faculty_id = ?
      ON DUPLICATE KEY UPDATE student_id = student_id
    `;
    db.query(assignSql, [student_id, facultyId], (err2) => {
      if (err2) return res.status(500).json({ error: err2.message });
      res.json({ message: "✅ Student assigned and timetable updated", id: result.insertId });
    });
  });
});

app.delete("/faculty/assigned-students/:studentId", authenticateToken, (req, res) => {
  if (req.user.role !== "faculty") return res.status(403).json({ error: "Forbidden" });

  const facultyId = req.user.faculty_id;
  const { studentId } = req.params;

  const sql = "DELETE FROM faculty_students WHERE faculty_id = ? AND student_id = ?";
  db.query(sql, [facultyId, studentId], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });

    if (result.affectedRows === 0) {
      return res.status(404).json({ error: "Student not assigned to this faculty" });
    }

    // ✅ Remove student from this faculty’s subject timetables
    const deleteSql = `
      DELETE st
      FROM student_timetable st
      JOIN timetable t ON st.timetable_id = t.timetable_id
      JOIN subjects sub ON t.subject_id = sub.subject_id
      WHERE st.student_id = ? AND sub.faculty_id = ?
    `;
    db.query(deleteSql, [studentId, facultyId], (err2) => {
      if (err2) return res.status(500).json({ error: err2.message });
      res.json({ message: "✅ Student unassigned and timetable updated" });
    });
  });
});


// 📌 Faculty: Fetch all attendance records with student + subject info
app.get("/faculty/attendance", authenticateToken, async (req, res) => {
  if (req.user.role !== "faculty") 
    return res.status(403).json({ error: "Access denied: Not a faculty" });

  const facultyId = req.user.faculty_id;

  try {
    const [rows] = await db.promise().query(
      `SELECT a.attendance_id, a.timestamp, 
              s.student_id, s.name AS student_name,
              sub.subject_id, sub.name AS subject_name
       FROM attendance a
       JOIN students s ON a.student_id = s.student_id
       JOIN subjects sub ON a.subject_id = sub.subject_id
       WHERE sub.faculty_id = ?
       ORDER BY a.timestamp DESC`,
      [facultyId]
    );

    res.json(rows);
  } catch (err) {
    console.error("❌ Error fetching attendance:", err);
    res.status(500).json({ error: "Failed to fetch attendance" });
  }
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


// Student self-registration (transaction-safe)
// Student self-registration (FIXED)
app.post("/students", authenticateToken, async (req, res) => {
  if (req.user.role !== "student") {
    return res.status(403).json({ error: "Only students can self-register" });
  }

  const { name, roll_number, class_name, email, phone } = req.body;
  const userId = req.user.user_id; // Get user ID from token

  try {
    // Start transaction
    await db.promise().query("START TRANSACTION");

    // 1. Check if student already exists for this user
    const [existingStudent] = await db.promise().query(
      "SELECT student_id FROM students WHERE email = ? OR roll_number = ?",
      [email, roll_number]
    );

    if (existingStudent.length > 0) {
      await db.promise().query("ROLLBACK");
      return res.status(400).json({ error: "Student with this email or roll number already exists" });
    }

    // 2. Insert into students table
    const [studentResult] = await db.promise().query(
      `INSERT INTO students (name, roll_number, class, email, phone, registered_on) 
       VALUES (?, ?, ?, ?, ?, NOW())`,
      [name, roll_number, class_name, email, phone]
    );

    const newStudentId = studentResult.insertId;

    // 3. Update users table to link student_id using user_id from token
    const [updateResult] = await db.promise().query(
      `UPDATE users SET student_id = ? WHERE user_id = ? AND role = 'student'`,
      [newStudentId, userId]
    );

    if (updateResult.affectedRows === 0) {
      await db.promise().query("ROLLBACK");
      return res.status(400).json({ 
        error: "User not found or not a student role" 
      });
    }

    // Commit transaction
    await db.promise().query("COMMIT");

    // Send welcome email
    try {
      await sendMail(
        email,
        "🎓 Welcome to Smart Attendance System",
        `Hello ${name},\n\nYou have successfully registered.\n- Roll Number: ${roll_number}\n- Class: ${class_name}\n- Student ID: ${newStudentId}\n\nBest Regards,\nAdmin Team`,
        `<h2>Welcome, ${name}! 🎓</h2>
         <p>You have successfully registered.</p>
         <p><b>Roll Number:</b> ${roll_number}<br>
            <b>Class:</b> ${class_name}<br>
            <b>Student ID:</b> ${newStudentId}</p>
         <p>Best Regards,<br><b>Admin Team</b></p>`
      );
      console.log(`📧 Registration email sent to ${email}`);
    } catch (mailErr) {
      console.error("❌ Failed to send registration email:", mailErr.message);
    }

    res.json({
      success: true,
      message: "Student registered & linked successfully!",
      student_id: newStudentId,
    });

  } catch (err) {
    await db.promise().query("ROLLBACK");
    console.error("❌ Registration error:", err);
    res.status(500).json({ error: "Registration failed" });
  }
});


// Create user login first (no student_id yet)
// Create user login first (IMPROVED)
app.post("/register-user", async (req, res) => {
  const { username, password, role } = req.body;

  if (!username || !password || !role) {
    return res.status(400).json({ error: "Username, password, and role are required" });
  }

  try {
    // Hash password
    const hashedPassword = await bcrypt.hash(password, 10);
    
    const [result] = await db.promise().query(
      "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
      [username, hashedPassword, role]
    );

    res.json({ 
      message: "User created successfully. Please login and complete your profile.", 
      user_id: result.insertId,
      role: role 
    });
  } catch (err) {
    if (err.code === 'ER_DUP_ENTRY') {
      return res.status(400).json({ error: "Username already exists" });
    }
    console.error("❌ User creation error:", err);
    res.status(500).json({ error: "User creation failed" });
  }
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
    db.query(updateUserSql, [newStudentId, username], async (err2) => {
      if (err2) return res.status(500).json({ error: err2.message });

      // ✅ Send registration email
      try {
        await sendMail({
          to: email,
          subject: "🎉 Registration Successful - Smart Attendance System",
          text: `Hello ${name},\n\nYou have been successfully registered in the Smart Attendance System.\n\nDetails:\n- Roll Number: ${roll_number}\n- Class: ${class_name}\n- Student ID: ${newStudentId}\n\nYou can now log in using your username: ${username}.\n\nBest Regards,\nAdmin Team`,
        });
        console.log(`📧 Registration email sent to ${email}`);
      } catch (mailErr) {
        console.error("❌ Failed to send registration email:", mailErr.message);
      }

      res.json({
        success: true,
        message: "Student registered, linked to user, and email sent successfully!",
        student_id: newStudentId,
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
// Add timetable entry (restricted to logged-in faculty)
// Add timetable entry (restricted to logged-in faculty)
app.post("/faculty/timetable", authenticateFaculty, (req, res) => {
  const facultyId = req.user.faculty_id;
  const { subject_id, day, start_time, end_time } = req.body;

  // Validate subject belongs to this faculty
  const checkSql = "SELECT * FROM subjects WHERE subject_id = ? AND faculty_id = ?";
  db.query(checkSql, [subject_id, facultyId], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    if (results.length === 0) return res.status(403).json({ error: "Not authorized for this subject" });

    // Insert timetable row
    const sql = `
      INSERT INTO timetable (faculty_id, subject_id, day, start_time, end_time)
      VALUES (?, ?, ?, ?, ?)
    `;
    db.query(sql, [facultyId, subject_id, day, start_time, end_time], (err2, result) => {
      if (err2) return res.status(500).json({ error: err2.message });

      const timetableId = result.insertId;

      // Insert into student_timetable for assigned students (ignore duplicates)
      const studentSql = `
        INSERT IGNORE INTO student_timetable (student_id, timetable_id)
        SELECT fs.student_id, ? 
        FROM faculty_students fs
        WHERE fs.faculty_id = ?
      `;
      db.query(studentSql, [timetableId, facultyId], (err3) => {
        if (err3) return res.status(500).json({ error: err3.message });
        res.json({ message: "Timetable entry added successfully for faculty and assigned students" });
      });
    });
  });
});


// Update timetable entry (only if faculty owns it)
app.put("/faculty/timetable/:id", authenticateFaculty, (req, res) => {
  const facultyId = req.user.faculty_id;
  const timetableId = req.params.id;
  const { day, start_time, end_time } = req.body;

  const sql = `
    UPDATE timetable t
    JOIN subjects s ON t.subject_id = s.subject_id
    SET t.day = ?, t.start_time = ?, t.end_time = ?
    WHERE t.timetable_id = ? AND s.faculty_id = ?
  `;

  db.query(sql, [day, start_time, end_time, timetableId, facultyId], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (result.affectedRows === 0) return res.status(403).json({ error: "Not authorized" });
    res.json({ message: "Timetable updated successfully" });
  });
});


// Delete timetable entry (only if faculty owns it)
// Delete timetable entry (faculty only)
app.delete("/faculty/timetable/:id", authenticateFaculty, (req, res) => {
  const facultyId = req.user.faculty_id;
  const timetableId = req.params.id;

  // 1️⃣ Delete related student timetable entries first
  const deleteStudentSql = `DELETE FROM student_timetable WHERE timetable_id = ?`;

  db.query(deleteStudentSql, [timetableId], (err1) => {
    if (err1) return res.status(500).json({ error: err1.message });

    // 2️⃣ Then delete timetable entry (only if owned by faculty)
    const deleteTimetableSql = `
      DELETE t FROM timetable t
      JOIN subjects s ON t.subject_id = s.subject_id
      WHERE t.timetable_id = ? AND s.faculty_id = ?
    `;

    db.query(deleteTimetableSql, [timetableId, facultyId], (err2, result) => {
      if (err2) return res.status(500).json({ error: err2.message });
      if (result.affectedRows === 0) return res.status(403).json({ error: "Not authorized" });

      res.json({ message: "Timetable deleted successfully (faculty + students)" });
    });
  });
});


// ================= ADMIN ROUTES =================

// Get all users
app.get("/admin/users", authenticateToken, isAdmin, (req, res) => {
  db.query("SELECT user_id, username, role FROM users", (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(result);
  });
});

// Add new user
app.post("/admin/addUser", authenticateToken, isAdmin, async (req, res) => {
  const { username, password, role } = req.body;
  if (!username || !password || !role) {
    return res.status(400).json({ error: "All fields required" });
  }

  const hashedPassword = await bcrypt.hash(password, 10);
  db.query(
    "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
    [username, hashedPassword, role],
    (err) => {
      if (err) return res.status(500).json({ error: err.message });
      res.json({ message: "✅ User added successfully" });
    }
  );
});

// Delete user
app.delete("/admin/users/:id", authenticateToken, isAdmin, (req, res) => {
  const { id } = req.params;
  db.query("DELETE FROM users WHERE user_id = ?", [id], (err) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json({ message: "🗑 User deleted" });
  });
});

// ---------------- STUDENTS ----------------
app.get("/admin/students", authenticateToken, (req, res) => {
  db.query("SELECT * FROM students", (err, results) => {
    if (err) return res.status(500).json(err);
    res.json(results);
  });
});

app.post("/admin/students", authenticateToken, (req, res) => {
  const { name, roll_number, class: studentClass, email, phone } = req.body;

  db.query(
    "INSERT INTO students (name, roll_number, class, email, phone) VALUES (?,?,?,?,?)",
    [name, roll_number, studentClass, email, phone],
    async (err, result) => {
      if (err) return res.status(500).json(err);

      const newStudentId = result.insertId;

      // ✅ Send welcome email
      try {
        await sendMail(
          email,
          "🎉 Registration Successful - Smart Attendance System",
          `Hello ${name},\n\nYou have been successfully registered in the Smart Attendance System.\n\nDetails:\n- Roll Number: ${roll_number}\n- Class: ${studentClass}\n- Student ID: ${newStudentId}\n\nYou can now use your account to mark attendance.\n\nBest Regards,\nAdmin Team`,
          `<h2>Welcome, ${name}! 🎉</h2>
           <p>You have been successfully registered in the <b>Smart Attendance System</b>.</p>
           <p>
             <b>Roll Number:</b> ${roll_number}<br>
             <b>Class:</b> ${studentClass}<br>
             <b>Student ID:</b> ${newStudentId}
           </p>
           <p>You can now use your account to mark attendance.</p>
           <br>
           <p>Best Regards,<br><b>Admin Team</b></p>`
        );

        console.log(`📧 Registration email sent to ${email}`);
      } catch (mailErr) {
        console.error("❌ Failed to send registration email:", mailErr.message);
      }

      res.json({ message: "Student added & email sent", student_id: newStudentId });
    }
  );
});

app.delete("/admin/students/:id", authenticateToken, (req, res) => {
  db.query(
    "DELETE FROM students WHERE student_id = ?",
    [req.params.id],
    (err) => {
      if (err) return res.status(500).json(err);
      res.json({ message: "Student deleted" });
    }
  );
});

// ---------------- FACULTY ----------------
// Get all faculty with user_id
app.get("/admin/faculty", authenticateToken, (req, res) => {
  const sql = `
    SELECT f.faculty_id, f.name, f.department, u.user_id
    FROM faculty f
    LEFT JOIN users u ON u.faculty_id = f.faculty_id
  `;
  db.query(sql, (err, results) => {
    if (err) return res.status(500).json(err);
    res.json(results);
  });
});



// Faculty registration (FIXED)
app.post("/admin/faculty", authenticateToken, async (req, res) => {
  const { user_id, name, department } = req.body;

  if (!user_id) {
    return res.status(400).json({ error: "user_id is required" });
  }

  try {
    await db.promise().query("START TRANSACTION");

    // 1. Insert into faculty table
    const [facultyResult] = await db.promise().query(
      "INSERT INTO faculty (name, department) VALUES (?, ?)",
      [name, department]
    );

    const facultyId = facultyResult.insertId;

    // 2. Update user table with faculty_id
    const [updateResult] = await db.promise().query(
      "UPDATE users SET faculty_id = ? WHERE user_id = ? AND role = 'faculty'",
      [facultyId, user_id]
    );

    if (updateResult.affectedRows === 0) {
      await db.promise().query("ROLLBACK");
      return res.status(400).json({ 
        error: "User not found or not a faculty role" 
      });
    }

    await db.promise().query("COMMIT");

    res.json({
      message: "Faculty added and linked to user",
      faculty_id: facultyId,
      user_id: user_id,
    });

  } catch (err) {
    await db.promise().query("ROLLBACK");
    console.error("❌ Faculty creation error:", err);
    res.status(500).json({ error: "Faculty creation failed" });
  }
});


app.delete("/admin/faculty/:id", authenticateToken, (req, res) => {
  db.query(
    "DELETE FROM faculty WHERE faculty_id = ?",
    [req.params.id],
    (err) => {
      if (err) return res.status(500).json(err);
      res.json({ message: "Faculty deleted" });
    }
  );
});



// --- SUBJECTS CRUD ---

// ---------------- SUBJECT ASSIGNMENT ----------------

// Get all subjects
app.get("/admin/subjects", authenticateToken, (req, res) => {
  db.query("SELECT * FROM subjects", (err, results) => {
    if (err) return res.status(500).json(err);
    res.json(results);
  });
});

// Unassign subject from faculty
// Must come BEFORE /assign
app.put("/admin/subjects/:id/unassign", authenticateToken, (req, res) => {
  const subjectId = req.params.id;

  db.query(
    "UPDATE subjects SET faculty_id = NULL WHERE subject_id = ?",
    [subjectId],
    (err, result) => {
      if (err) return res.status(500).json(err);
      console.log("Unassign result:", result);   // 👈 Debug
      res.json({ message: "Subject unassigned successfully", result });
    }
  );
});

app.put("/admin/subjects/:id/assign", authenticateToken, (req, res) => {
  const subjectId = req.params.id;
  const { faculty_id } = req.body;

  db.query(
    "UPDATE subjects SET faculty_id = ? WHERE subject_id = ?",
    [faculty_id, subjectId],
    (err, result) => {
      if (err) return res.status(500).json(err);
      res.json({ message: "Subject assigned successfully", result });
    }
  );
});


app.post("/admin/subjects", authenticateToken, (req, res) => {
  const { name } = req.body;

  if (!name) return res.status(400).json({ error: "Subject name is required" });

  db.query(
    "INSERT INTO subjects (name) VALUES (?)",
    [name],
    (err, result) => {
      if (err) return res.status(500).json(err);

      res.json({
        message: "Subject added successfully",
        subject_id: result.insertId,
        name
      });
    }
  );
});

// Update subject (only name)
app.put("/admin/subjects/:id", authenticateToken, (req, res) => {
  const subjectId = req.params.id;
  const { name } = req.body;

  if (!name) return res.status(400).json({ error: "New subject name is required" });

  db.query(
    "UPDATE subjects SET name = ? WHERE subject_id = ?",
    [name, subjectId],
    (err, result) => {
      if (err) return res.status(500).json(err);

      res.json({ message: "Subject updated successfully" });
    }
  );
});

// Delete subject
app.delete("/admin/subjects/:id", authenticateToken, (req, res) => {
  const subjectId = req.params.id;

  db.query(
    "DELETE FROM subjects WHERE subject_id = ?",
    [subjectId],
    (err, result) => {
      if (err) return res.status(500).json(err);

      res.json({ message: "Subject deleted successfully" });
    }
  );
});


// Delete faculty
app.delete("/admin/faculty/:id", authenticateToken, (req, res) => {
  const facultyId = req.params.id;

  // First, unassign all subjects linked to this faculty
  db.query("UPDATE subjects SET faculty_id = NULL WHERE faculty_id = ?", [facultyId], (err) => {
    if (err) return res.status(500).json(err);

    // Then delete the faculty
    db.query("DELETE FROM faculty WHERE faculty_id = ?", [facultyId], (err, result) => {
      if (err) return res.status(500).json(err);
      res.json({ message: "Faculty deleted successfully" });
    });
  });
});



// ---------------- ASSIGNMENTS ----------------
// ---------------- ADMIN: ASSIGN STUDENTS TO SUBJECTS ----------------
// ---------------- ADMIN: FACULTY-STUDENT ASSIGNMENTS ----------------
// ---------------- ADMIN: MANAGE FACULTY-STUDENT ASSIGNMENTS ----------------

// Get assigned students of a faculty
// Get assigned students of a faculty
app.get("/admin/faculty/:facultyId/assigned-students", authenticateToken, (req, res) => {
  if (req.user.role !== "admin") return res.status(403).json({ error: "Forbidden" });
  const { facultyId } = req.params;
  const sql = `
    SELECT fs.id, s.student_id, s.name, s.roll_number, s.class, s.email, s.phone, fs.assigned_on
    FROM faculty_students fs
    JOIN students s ON fs.student_id = s.student_id
    WHERE fs.faculty_id = ?
  `;
  db.query(sql, [facultyId], (err, results) => {
    if (err) return res.status(500).json({ error: err.message });
    res.json(results);
  });
});

// Assign student
app.post("/admin/faculty/:facultyId/assigned-students", authenticateToken, (req, res) => {
  if (req.user.role !== "admin") return res.status(403).json({ error: "Forbidden" });
  const { facultyId } = req.params;
  const { student_id } = req.body;
  const sql = "INSERT INTO faculty_students (faculty_id, student_id) VALUES (?, ?)";
  db.query(sql, [facultyId, student_id], (err, result) => {
    if (err) {
      if (err.code === "ER_DUP_ENTRY") return res.status(400).json({ error: "Student already assigned" });
      return res.status(500).json({ error: err.message });
    }
    const assignSql = `
      INSERT INTO student_timetable (student_id, timetable_id)
      SELECT ?, t.timetable_id
      FROM timetable t
      JOIN subjects sub ON t.subject_id = sub.subject_id
      WHERE sub.faculty_id = ?
      ON DUPLICATE KEY UPDATE student_id = student_id
    `;
    db.query(assignSql, [student_id, facultyId], (err2) => {
      if (err2) return res.status(500).json({ error: err2.message });
      res.json({ message: "✅ Student assigned and timetable updated", id: result.insertId });
    });
  });
});

// Unassign student
app.delete("/admin/faculty/:facultyId/assigned-students/:studentId", authenticateToken, (req, res) => {
  if (req.user.role !== "admin") return res.status(403).json({ error: "Forbidden" });
  const { facultyId, studentId } = req.params;
  const sql = "DELETE FROM faculty_students WHERE faculty_id = ? AND student_id = ?";
  db.query(sql, [facultyId, studentId], (err, result) => {
    if (err) return res.status(500).json({ error: err.message });
    if (result.affectedRows === 0) return res.status(404).json({ error: "Not assigned" });
    const deleteSql = `
      DELETE st
      FROM student_timetable st
      JOIN timetable t ON st.timetable_id = t.timetable_id
      JOIN subjects sub ON t.subject_id = sub.subject_id
      WHERE st.student_id = ? AND sub.faculty_id = ?
    `;
    db.query(deleteSql, [studentId, facultyId], (err2) => {
      if (err2) return res.status(500).json({ error: err2.message });
      res.json({ message: "✅ Student unassigned and timetable updated" });
    });
  });
});



// ---------------- TIMETABLE ----------------
// ---------------- GET TIMETABLE BY FACULTY ----------------
// ---------------- GET TIMETABLE BY FACULTY ----------------
// ---------------- GET FACULTY TIMETABLE ----------------
app.get("/admin/timetable/faculty/:facultyId", authenticateToken, (req, res) => {
  const { facultyId } = req.params;

  const sql = `
    SELECT 
      t.timetable_id, 
      s.subject_id, 
      s.name AS subject, 
      t.day, 
      t.start_time, 
      t.end_time
    FROM timetable t
    JOIN subjects s ON t.subject_id = s.subject_id
    WHERE s.faculty_id = ?`;

  db.query(sql, [facultyId], (err, results) => {
    if (err) return res.status(500).json(err);
    res.json(results);
  });
});


// ---------------- ADD TIMETABLE (Admin) ----------------
app.post("/admin/timetable", authenticateToken, (req, res) => {
  const { faculty_id, subject_id, day, start_time, end_time } = req.body;

  const sql = `
    INSERT INTO timetable (faculty_id, subject_id, day, start_time, end_time)
    VALUES (?, ?, ?, ?, ?)`;

  db.query(sql, [faculty_id, subject_id, day, start_time, end_time], (err, result) => {
    if (err) return res.status(500).json(err);

    const timetableId = result.insertId;

    // 🔑 Insert timetable for all assigned students
    const studentSql = `
      INSERT INTO student_timetable (student_id, timetable_id)
      SELECT fs.student_id, ?
      FROM faculty_students fs
      WHERE fs.faculty_id = ?`;

    db.query(studentSql, [timetableId, faculty_id], (err2) => {
      if (err2) return res.status(500).json(err2);
      res.json({ message: "Timetable added (faculty + students)", id: timetableId });
    });
  });
});


// ---------------- UPDATE TIMETABLE (Admin) ----------------
app.put("/admin/timetable/:id", authenticateToken, (req, res) => {
  const { id } = req.params;
  const { subject_id, day, start_time, end_time } = req.body;

  const sql = `
    UPDATE timetable
    SET subject_id = ?, day = ?, start_time = ?, end_time = ?
    WHERE timetable_id = ?`;

  db.query(sql, [subject_id, day, start_time, end_time, id], (err, result) => {
    if (err) return res.status(500).json(err);
    if (result.affectedRows === 0) return res.status(404).json({ error: "Timetable not found" });

    res.json({ message: "Timetable updated (faculty + students)" });
  });
});


// ---------------- DELETE TIMETABLE (Admin) ----------------
app.delete("/admin/timetable/:id", authenticateToken, (req, res) => {
  const { id } = req.params;

  // 1️⃣ Delete from student_timetable first
  const deleteStudentSql = `DELETE FROM student_timetable WHERE timetable_id = ?`;

  db.query(deleteStudentSql, [id], (err1) => {
    if (err1) return res.status(500).json(err1);

    // 2️⃣ Delete timetable entry
    const deleteSql = `DELETE FROM timetable WHERE timetable_id = ?`;

    db.query(deleteSql, [id], (err2, result) => {
      if (err2) return res.status(500).json(err2);
      if (result.affectedRows === 0) return res.status(404).json({ error: "Timetable not found" });

      res.json({ message: "Timetable deleted (faculty + students)" });
    });
  });
});



// ---------------- GET SUBJECTS BY FACULTY ----------------
// get subjects for faculty
app.get("/admin/subjects/faculty/:facultyId", authenticateToken, (req, res) => {
  const { facultyId } = req.params;
  db.query("SELECT * FROM subjects WHERE faculty_id = ?", [facultyId], (err, result) => {
    if (err) return res.status(500).json(err);
    res.json(result);
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