import os
import cv2
import numpy as np
import time
import threading
from queue import Queue
import shutil
from deepface import DeepFace
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from config import Config
from models.database import db, Admin, Student, Attendance, SystemLog
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date
from werkzeug.utils import secure_filename
from utils.logger import log_action
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

class FaceRecognition:
    def __init__(self, app):
        self.app = app
        self.known_faces = {}
        self.model_name = Config.FACE_RECOGNITION_BACKEND
        self.detector_backend = Config.FACE_DETECTION_BACKEND
        self.lock = threading.Lock()
        self.task_queue = Queue(maxsize=100)
        self._initialize_workers()
        self._preload_models()
        self.refresh_known_faces()

    def _initialize_workers(self):
        """Start worker threads for background tasks"""
        for _ in range(Config.THREAD_POOL_SIZE):
            worker = threading.Thread(target=self._process_tasks, daemon=True)
            worker.start()

    def _preload_models(self):
        """Preload models in background thread"""
        def load_models():
            try:
                dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
                DeepFace.extract_faces(
                    img_path=dummy_img,
                    detector_backend='retinaface',
                    enforce_detection=False
                )
            except Exception as e:
                app.logger.error(f"Model preloading failed: {str(e)}")

        threading.Thread(target=load_models, daemon=True).start()

    def _process_tasks(self):
        """Process background tasks with error handling"""
        while True:
            try:
                task, args, kwargs = self.task_queue.get()
                with self.app.app_context():
                    task(*args, **kwargs)
            except Exception as e:
                app.logger.error(f"Background task failed: {str(e)}")
            finally:
                self.task_queue.task_done()

    def refresh_known_faces(self):
        """Thread-safe refresh of face database"""
        with self.lock:
            with self.app.app_context():
                self.known_faces = {}
                # Corrected load_only usage - using actual model attributes
                students = Student.query.options(
                    db.load_only(
                        Student.student_id,
                        Student.face_embedding,
                        Student.photo_path,
                        Student.name
                    )
                ).filter(
                    Student.face_embedding.isnot(None),
                    Student.photo_path.isnot(None)
                ).all()

                for student in students:
                    try:
                        embedding = np.frombuffer(student.face_embedding, dtype=np.float32)
                        if os.path.exists(student.photo_path):
                            self.known_faces[student.student_id] = {
                                'embedding': embedding,
                                'photo_path': student.photo_path,
                                'name': student.name
                            }
                    except Exception as e:
                        app.logger.error(f"Failed to load face data for {student.student_id}: {str(e)}")
    def detect_faces(self, frame):
        """Optimized face detection with multiple fallback methods"""
        if frame is None or frame.size == 0:
            return [], []

        try:
            # Convert and resize frame for better performance
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Try multiple backends with timeout
            face_objs = []
            for backend in ['retinaface', 'mtcnn', 'opencv']:
                try:
                    face_objs = DeepFace.extract_faces(
                        img_path=frame,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True
                    )
                    if face_objs:
                        break
                except:
                    continue

            locations = []
            encodings = []

            for face_obj in face_objs:
                if face_obj['confidence'] < Config.MIN_CONFIDENCE:
                    continue

                area = face_obj['facial_area']
                locations.append((
                    max(0, area['y']),
                    min(frame.shape[0], area['y'] + area['h']),
                    max(0, area['x']),
                    min(frame.shape[1], area['x'] + area['w'])
                ))

                try:
                    embedding = DeepFace.represent(
                        img_path=face_obj['face'],
                        model_name=self.model_name,
                        enforce_detection=False
                    )[0]['embedding']
                    encodings.append(np.array(embedding, dtype=np.float32))
                except:
                    continue

            return locations, encodings

        except Exception as e:
            app.logger.error(f"Face detection error: {str(e)}")
            return [], []

    def recognize_faces(self, face_encodings):
        """Optimized face recognition with thread safety"""
        if not face_encodings:
            return ["Unknown"] * len(face_encodings)

        with self.lock:
            if not self.known_faces:
                return ["Unknown"] * len(face_encodings)

            known_embeddings = []
            known_ids = []
            for student_id, data in self.known_faces.items():
                if data['embedding'] is not None:
                    known_embeddings.append(data['embedding'])
                    known_ids.append(student_id)

            if not known_embeddings:
                return ["Unknown"] * len(face_encodings)

            known_embeddings = np.array(known_embeddings)
            recognized_names = []

            for encoding in face_encodings:
                try:
                    similarities = cosine_similarity([encoding], known_embeddings)[0]
                    max_idx = np.argmax(similarities)
                    if similarities[max_idx] >= Config.MIN_CONFIDENCE:
                        recognized_names.append(known_ids[max_idx])
                    else:
                        recognized_names.append("Unknown")
                except:
                    recognized_names.append("Unknown")

            return recognized_names

    def enroll_face(self, image_path, student_id, callback=None):
        """Optimized face enrollment with validation"""
        def _enroll_task():
            try:
                # Validate image
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Invalid image file")

                # Check image quality
                if img.shape[0] < 300 or img.shape[1] < 300:
                    raise ValueError("Image too small (minimum 300x300)")

                # Verify student
                with self.app.app_context():
                    student = Student.query.filter_by(student_id=student_id).first()
                    if not student:
                        raise ValueError("Student not found")
                    if student.face_embedding is not None:
                        raise ValueError("Student already enrolled")

                # Face detection
                face_objs = []
                for backend in ['retinaface', 'mtcnn', 'opencv']:
                    try:
                        face_objs = DeepFace.extract_faces(
                            img_path=image_path,
                            detector_backend=backend,
                            enforce_detection=True,
                            align=True
                        )
                        if face_objs:
                            break
                    except:
                        continue

                if not face_objs or len(face_objs) > 1:
                    raise ValueError("Invalid face detection")
                if face_objs[0]['confidence'] < 0.9:
                    raise ValueError("Low detection confidence")

                # Generate embedding
                embedding = DeepFace.represent(
                    img_path=image_path,
                    model_name=self.model_name,
                    enforce_detection=False,
                    align=True
                )
                if not embedding:
                    raise ValueError("Failed to generate embedding")

                # Save to database
                backup_dir = os.path.join(Config.BACKUP_FOLDER, 'enrollments')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, f"{student_id}_{int(time.time())}.jpg")
                shutil.copy2(image_path, backup_path)

                with self.app.app_context():
                    student.face_embedding = np.array(embedding[0]['embedding']).tobytes()
                    student.photo_path = backup_path
                    student.face_enrollment_date = datetime.utcnow()
                    db.session.commit()

                # Refresh cache
                self.refresh_known_faces()
                result = (True, "Enrollment successful")

            except Exception as e:
                result = (False, str(e))
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                    except:
                        pass
            finally:
                if callback:
                    callback(*result)

        try:
            self.task_queue.put((_enroll_task, (), {}), timeout=5)
            return True, "Enrollment processing started"
        except:
            return False, "System busy, please try again"

# Initialize system
with app.app_context():
    db.create_all()
    if not Admin.query.filter_by(username='admin').first():
        admin = Admin(
            username='admin',
            password=generate_password_hash('admin123'),
            email='admin@example.com'
        )
        db.session.add(admin)
        db.session.commit()

face_recognition = FaceRecognition(app)

# Utility functions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def capture_frame(camera_index=0):
    """Optimized frame capture with multiple backend support"""
    for backend in [Config.get_camera_backend(), cv2.CAP_ANY]:
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    return frame
        except:
            continue
        finally:
            if cap:
                cap.release()
    return None

# Routes

@app.route('/')
def home():
    return redirect(url_for('dashboard')) if 'admin_id' in session else redirect(url_for('login'))


@app.route('/verify-enrollment/<student_id>')
def verify_enrollment(student_id):
    """Endpoint to verify face enrollment status"""
    if 'admin_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    with app.app_context():
        student = Student.query.filter_by(student_id=student_id).first()
        if not student:
            return jsonify({'error': 'Student not found'}), 404

        return jsonify({
            'has_face_data': student.face_embedding is not None,
            'photo_exists': os.path.exists(student.photo_path) if student.photo_path else False,
            'enrollment_date': student.face_enrollment_date.isoformat() if student.face_enrollment_date else None
        })

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        admin = Admin.query.filter_by(email=email).first()
        if admin:
            # In a real app, you would send a password reset email here
            flash('If this email exists, a reset link has been sent', 'info')
            return redirect(url_for('login'))
        flash('Email not found', 'error')
    return render_template('forgot_password.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        admin = Admin.query.filter_by(username=username).first()
        if admin and check_password_hash(admin.password, password):
            session['admin_id'] = admin.id
            admin.last_login = datetime.utcnow()
            db.session.commit()
            log_action(admin.id, "Admin logged in", request.remote_addr)
            return redirect(url_for('dashboard'))
        flash('Invalid username or password', 'error')
    return render_template('login.html')


@app.route('/logout')
def logout():
    if 'admin_id' in session:
        log_action(session['admin_id'], "Admin logged out", request.remote_addr)
        session.pop('admin_id')
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    total_students = Student.query.count()
    today_attendance = Attendance.query.filter_by(date=date.today()).count()
    return render_template('dashboard.html', total_students=total_students, today_attendance=today_attendance)


@app.route('/student-management', methods=['GET', 'POST'])
def student_management():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        data = request.form
        student = Student.query.filter_by(student_id=data['student_id']).first()
        action = "Updated student" if student else "Added new student"

        if student:
            student.name = data['name']
            student.class_name = data['class']
            student.section = data['section']
            student.roll_number = data['roll_number']
        else:
            db.session.add(Student(
                student_id=data['student_id'],
                name=data['name'],
                class_name=data['class'],
                section=data['section'],
                roll_number=data['roll_number']
            ))
        db.session.commit()
        flash(f'Student {action.lower()} successfully', 'success')
        log_action(session['admin_id'], f"{action} {data['student_id']}", request.remote_addr)
        return redirect(url_for('student_management'))

    return render_template('student_management.html', students=Student.query.all())


@app.route('/face-enrollment', methods=['GET', 'POST'])
def face_enrollment():
    if 'admin_id' not in session:
        if request.method == 'POST':
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            student_id = request.form.get('student_id')
            capture_method = request.form.get('capture_method')

            if not student_id:
                return jsonify({'success': False, 'message': 'Please select a student'}), 400

            # Verify student exists before processing
            with app.app_context():
                student = Student.query.filter_by(student_id=student_id).first()
                if not student:
                    return jsonify({'success': False, 'message': 'Student not found'}), 404

            # Handle file upload
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            upload_path = None

            if capture_method == 'webcam':
                if 'webcamImage' not in request.files:
                    return jsonify({'success': False, 'message': 'No image captured'}), 400
                file = request.files['webcamImage']
                if file.filename == '':
                    return jsonify({'success': False, 'message': 'No image captured'}), 400
                filename = secure_filename(f"{student_id}_{int(time.time())}.jpg")
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)

            elif capture_method == 'upload':
                if 'photo' not in request.files:
                    return jsonify({'success': False, 'message': 'No file selected'}), 400
                file = request.files['photo']
                if not allowed_file(file.filename):
                    return jsonify({'success': False, 'message': 'Invalid file type'}), 400
                filename = secure_filename(f"{student_id}_{int(time.time())}.jpg")
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(upload_path)

            else:
                return jsonify({'success': False, 'message': 'Invalid capture method'}), 400

            # Verify image
            if cv2.imread(upload_path) is None:
                os.remove(upload_path)
                return jsonify({'success': False, 'message': 'Invalid image file'}), 400

            # Enrollment callback
            def enrollment_callback(success, message):
                with app.app_context():
                    if success:
                        log_action(session['admin_id'], f"Enrolled face for {student_id}", request.remote_addr)
                        # Verify the data was actually saved
                        student = Student.query.filter_by(student_id=student_id).first()
                        if student and student.face_embedding:
                            print(f"[VERIFICATION] Successfully saved face data for {student_id}")
                        else:
                            print(f"[VERIFICATION ERROR] Failed to save face data for {student_id}")
                    else:
                        print(f"[ENROLLMENT FAILED] {message}")

            # Start enrollment
            success, msg = face_recognition.enroll_face(upload_path, student_id, enrollment_callback)
            return jsonify({'success': success, 'message': msg})

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500

    # GET request
    students = Student.query.all()
    return render_template('face_enrollment.html', students=students)


@app.route('/attendance-capture', methods=['GET', 'POST'])
def attendance_capture():
    if 'admin_id' not in session:
        if request.method == 'POST':
            return jsonify({
                'success': False,
                'message': 'Unauthorized',
                'error_type': 'authentication_error'
            }), 401
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # 1. Capture frame from webcam
            frame = capture_frame()
            if frame is None:
                return jsonify({
                    'success': False,
                    'message': 'Camera Error',
                    'details': 'Webcam unavailable or inaccessible',
                    'error_type': 'camera_error'
                }), 400

            # 2. Convert frame to RGB
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'message': 'Image Processing Error',
                    'details': str(e),
                    'error_type': 'image_processing_error'
                }), 400

            # 3. Detect faces with multiple fallback methods
            face_objs = []
            backends = ['retinaface', 'mtcnn', 'opencv']  # Ordered by preference

            for backend in backends:
                try:
                    face_objs = DeepFace.extract_faces(
                        img_path=frame_rgb,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True
                    )
                    if face_objs:
                        break
                except Exception as e:
                    app.logger.warning(f"Face detection with {backend} failed: {str(e)}")
                    continue

            if not face_objs:
                return jsonify({
                    'success': False,
                    'message': 'No Faces Detected',
                    'details': 'Please ensure: 1) Good lighting 2) Face the camera directly',
                    'error_type': 'no_faces_detected'
                }), 400

            # 4. Process detected faces
            face_locations = []
            face_encodings = []
            min_confidence = app.config.get('MIN_CONFIDENCE', 0.6)

            for face_obj in face_objs:
                if face_obj['confidence'] < min_confidence:
                    continue

                area = face_obj['facial_area']
                face_locations.append([
                    area['y'],
                    area['y'] + area['h'],
                    area['x'],
                    area['x'] + area['w']
                ])

                try:
                    embedding = DeepFace.represent(
                        img_path=face_obj['face'],
                        model_name="VGG-Face",
                        enforce_detection=False,
                        align=True
                    )[0]['embedding']
                    face_encodings.append(np.array(embedding, dtype=np.float32))
                except Exception as e:
                    app.logger.warning(f"Embedding generation failed: {str(e)}")
                    continue

            if not face_encodings:
                return jsonify({
                    'success': False,
                    'message': 'Feature Extraction Failed',
                    'details': 'Could not extract facial features',
                    'error_type': 'feature_extraction_error'
                }), 400

            # 5. Recognize faces
            recognized_names = face_recognition.recognize_faces(face_encodings)
            attendance_data = []

            # 6. Record attendance
            try:
                with app.app_context():
                    for student_id in recognized_names:
                        if student_id == "Unknown":
                            continue

                        student = Student.query.filter_by(student_id=student_id).first()
                        if not student:
                            continue

                        attendance = Attendance.query.filter_by(
                            student_id=student.id,
                            date=date.today()
                        ).first()

                        if attendance:
                            attendance.exit_time = datetime.now()
                            status = 'Updated'
                        else:
                            attendance = Attendance(
                                student_id=student.id,
                                date=date.today(),
                                entry_time=datetime.now(),
                                status='Present'
                            )
                            db.session.add(attendance)
                            status = 'Recorded'

                        db.session.commit()
                        log_action(session['admin_id'], f"{status} attendance for {student_id}", request.remote_addr)

                        attendance_data.append({
                            'student_id': student_id,
                            'name': student.name,
                            'status': status
                        })

                return jsonify({
                    'success': True,
                    'message': 'Attendance Recorded',
                    'recognized_names': recognized_names,
                    'face_locations': face_locations,
                    'attendance_data': attendance_data
                })

            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Database error during attendance recording: {str(e)}")
                return jsonify({
                    'success': False,
                    'message': 'Database Error',
                    'details': str(e),
                    'error_type': 'database_error'
                }), 500

        except Exception as e:
            app.logger.error(f"Unexpected error in attendance capture: {str(e)}")
            return jsonify({
                'success': False,
                'message': 'System Error',
                'details': str(e),
                'error_type': 'system_error'
            }), 500

    # GET request - show recent attendance
    try:
        with app.app_context():
            recent_attendance = db.session.query(Attendance, Student) \
                .join(Student) \
                .filter(Attendance.date == date.today()) \
                .order_by(Attendance.entry_time.desc()) \
                .limit(10) \
                .all()

        return render_template('attendance_capture.html', recent_attendance=recent_attendance)

    except Exception as e:
        app.logger.error(f"Error fetching recent attendance: {str(e)}")
        flash('Error loading attendance records', 'error')
        return render_template('attendance_capture.html', recent_attendance=[])
@app.route('/attendance-report')
def attendance_report():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    query = db.session.query(Attendance, Student).join(Student)
    student_id = request.args.get('student_id')
    class_name = request.args.get('class')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if student_id:
        query = query.filter(Student.student_id == student_id)
    if class_name:
        query = query.filter(Student.class_name == class_name)
    if start_date:
        query = query.filter(Attendance.date >= start_date)
    if end_date:
        query = query.filter(Attendance.date <= end_date)

    records = query.order_by(Attendance.date.desc()).all()
    return render_template('attendance_report.html', records=records)


@app.route('/logs')
def logs():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    return render_template('logs.html', logs=SystemLog.query.order_by(SystemLog.timestamp.desc()).all())


@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        app.config['MIN_CONFIDENCE'] = float(request.form['min_confidence'])
        flash('Settings updated successfully', 'success')
        log_action(session['admin_id'], "Updated system settings", request.remote_addr)
        return redirect(url_for('settings'))

    return render_template('settings.html', min_confidence=app.config['MIN_CONFIDENCE'])


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'admin_id' not in session:
        return redirect(url_for('login'))

    admin = Admin.query.get(session['admin_id'])

    if request.method == 'POST':
        admin.email = request.form['email']
        new_password = request.form.get('new_password')
        if new_password:
            admin.password = generate_password_hash(new_password)
            flash('Password changed successfully', 'success')
        db.session.commit()
        flash('Profile updated successfully', 'success')
        log_action(admin.id, "Updated admin profile", request.remote_addr)
        return redirect(url_for('profile'))

    return render_template('profile.html', admin=admin)


@app.route('/backup', methods=['GET', 'POST'])
def backup():
    if 'admin_id' not in session:
        return redirect(url_for('login'))
    # Implement backup logic here
    return render_template('backup.html')


if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.BACKUP_FOLDER, exist_ok=True)

    # Start application
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=True,
        debug=os.getenv('FLASK_DEBUG', False)
    )