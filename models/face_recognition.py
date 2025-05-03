import os
import cv2
import datetime
import numpy as np
import logging
from functools import lru_cache
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from flask import current_app
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

from models.database import Student, db

logger = logging.getLogger(__name__)


class FaceRecognition:
    _instance = None
    _lock = Lock()

    def __new__(cls, app=None):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(FaceRecognition, cls).__new__(cls)
                    cls._instance._initialized = False
                    if app is not None:
                        cls._instance.init_app(app)
        return cls._instance

    def init_app(self, app):
        """Initialize with application context"""
        self.app = app
        self.model_name = app.config.get("FACE_MODEL", "ArcFace")
        self.detector_backend = app.config.get("DETECTOR_BACKEND", "retinaface")
        self.min_confidence = float(app.config.get("MIN_CONFIDENCE", 0.55))
        self.known_faces = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._face_cache = {}
        self._initialized = True
        self.refresh_known_faces()
        logger.info("Face Recognition system initialized")

    def refresh_known_faces(self):
        """Reload known faces from database with thread safety"""
        with self.app.app_context(), self._lock:
            try:
                start_time = time.time()
                students = Student.query.filter(
                    Student.face_embedding.isnot(None),
                    Student.photo_path.isnot(None)
                ).all()

                temp_faces = {}
                for student in students:
                    try:
                        if isinstance(student.face_embedding, bytes):
                            embedding = np.frombuffer(student.face_embedding, dtype=np.float32)
                        else:
                            embedding = student.face_embedding

                        if os.path.exists(student.photo_path):
                            temp_faces[str(student.student_id)] = {
                                'embedding': embedding,
                                'photo_path': student.photo_path,
                                'name': student.name,
                                'last_updated': student.face_enrollment_date
                            }
                        else:
                            logger.warning(f"Photo missing for student {student.student_id}")
                    except Exception as e:
                        logger.error(f"Failed to load face data for {student.student_id}: {str(e)}")

                self.known_faces = temp_faces
                logger.info(f"Refreshed {len(self.known_faces)} known faces in {time.time() - start_time:.2f}s")
                return True
            except Exception as e:
                logger.error(f"Failed to refresh known faces: {str(e)}", exc_info=True)
                return False

    def _preprocess_image(self, img):
        """Enhanced image preprocessing pipeline"""
        try:
            # Convert to RGB if needed
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Quality checks
            if img.mean() < 30:  # Too dark
                logger.warning("Image too dark")
                return None

            # Enhance contrast using CLAHE in LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            limg = clahe.apply(l)
            enhanced = cv2.merge((limg, a, b))
            img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

            # Resize to optimal dimensions
            img = cv2.resize(img, (160, 160))

            # Normalize pixel values
            img = img.astype('float32') / 255.0
            return img

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            return None

    def _get_embedding(self, img_path, use_cache=True):
        """Robust face embedding extraction with caching"""
        cache_key = img_path if isinstance(img_path, str) else str(img_path.tobytes())

        if use_cache and cache_key in self._face_cache:
            return self._face_cache[cache_key]

        try:
            # Load and validate image
            img = cv2.imread(img_path) if isinstance(img_path, str) else img_path
            if img is None or img.size == 0:
                return None

            # Preprocess image
            img = self._preprocess_image(img)
            if img is None:
                return None

            # Try multiple models if primary fails
            models = [self.model_name, 'Facenet', 'ArcFace', 'OpenFace']
            embedding = None

            for model in models:
                try:
                    result = DeepFace.represent(
                        img_path=img,
                        model_name=model,
                        detector_backend='skip',
                        enforce_detection=False,
                        align=True
                    )
                    if result and len(result) > 0:
                        embedding = np.array(result[0]['embedding'], dtype=np.float32)
                        break
                except Exception as e:
                    logger.debug(f"Model {model} failed: {str(e)}")
                    continue

            if use_cache and embedding is not None:
                self._face_cache[cache_key] = embedding

            return embedding

        except Exception as e:
            logger.error(f"Embedding extraction failed: {str(e)}")
            return None

    def detect_faces(self, frame):
        """Enhanced face detection with multiple fallbacks"""
        if frame is None or frame.size == 0:
            return [], []

        try:
            # Try multiple backends if primary fails
            backends = [self.detector_backend, 'mtcnn', 'opencv']
            face_objs = []

            for backend in backends:
                try:
                    face_objs = DeepFace.extract_faces(
                        img_path=frame,
                        detector_backend=backend,
                        enforce_detection=False,
                        align=True
                    )
                    if face_objs:
                        break
                except Exception as e:
                    logger.debug(f"Detection with {backend} failed: {str(e)}")
                    continue

            locations = []
            encodings = []

            for face_obj in face_objs:
                if face_obj['confidence'] < 0.7:  # Confidence threshold
                    continue

                area = face_obj['facial_area']
                locations.append((
                    max(0, area['y']),
                    min(frame.shape[0], area['y'] + area['h']),
                    max(0, area['x']),
                    min(frame.shape[1], area['x'] + area['w'])
                ))

                # Get embedding from face region
                face_region = frame[area['y']:area['y'] + area['h'], area['x']:area['x'] + area['w']]
                embedding = self._get_embedding(face_region)
                if embedding is not None:
                    encodings.append(embedding)

            return locations, encodings

        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return [], []

    def recognize_faces(self, face_encodings):
        """Face recognition with adaptive thresholding"""
        if not self.known_faces or not face_encodings:
            return ["Unknown"] * len(face_encodings) if face_encodings else []

        # Prepare known embeddings
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
            if encoding is None:
                recognized_names.append("Unknown")
                continue

            # Calculate similarities
            similarities = cosine_similarity([encoding], known_embeddings)[0]
            max_idx = np.argmax(similarities)
            max_similarity = similarities[max_idx]

            # Adaptive threshold based on image quality
            effective_threshold = self.min_confidence
            if np.mean(encoding) < 100:  # Dark image
                effective_threshold = max(self.min_confidence - 0.1, 0.4)
            elif np.mean(encoding) > 200:  # Bright image
                effective_threshold = max(self.min_confidence - 0.05, 0.45)

            if max_similarity >= effective_threshold:
                recognized_names.append(known_ids[max_idx])
            else:
                recognized_names.append("Unknown")

        return recognized_names

    def enroll_face(self, image_path, student_id):
        """Secure face enrollment with quality checks"""
        try:
            # Validate inputs
            if not os.path.exists(image_path):
                return False, "Image file not found"

            with self.app.app_context():
                student = Student.query.filter_by(student_id=str(student_id)).first()
                if not student:
                    return False, "Student not found"

            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return False, "Invalid image file"

            # Quality checks
            if img.shape[0] < 300 or img.shape[1] < 300:
                return False, "Image too small (min 300x300 pixels)"

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if cv2.mean(gray)[0] < 50:
                return False, "Image too dark"

            # Face detection
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=img,
                    detector_backend=self.detector_backend,
                    enforce_detection=True,
                    align=True
                )

                if len(face_objs) != 1:
                    return False, "Image must contain exactly one face"

                if face_objs[0]['confidence'] < 0.9:
                    return False, "Face detection confidence too low"
            except Exception as e:
                return False, f"Face detection failed: {str(e)}"

            # Generate embedding
            embedding = self._get_embedding(img, use_cache=False)
            if embedding is None:
                return False, "Failed to extract facial features"

            # Save to database
            with self.app.app_context():
                student.face_embedding = embedding.tobytes()
                student.photo_path = os.path.abspath(image_path)
                student.face_enrollment_date = datetime.utcnow()
                db.session.commit()

            # Update cache
            self.refresh_known_faces()
            return True, "Face enrolled successfully"

        except Exception as e:
            logger.error(f"Enrollment error: {str(e)}")
            return False, f"Enrollment error: {str(e)}"

    def detect_and_recognize(self, frame):
        """Combined detection and recognition"""
        locations, encodings = self.detect_faces(frame)
        names = self.recognize_faces(encodings)
        return locations, names

    def batch_process(self, frame_list):
        """Process multiple frames in parallel"""
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.detect_and_recognize, frame_list))
        return results

    def __del__(self):
        """Cleanup resources"""
        self._executor.shutdown(wait=True)
        cv2.destroyAllWindows()
        logger.info("Face Recognition system shutdown")