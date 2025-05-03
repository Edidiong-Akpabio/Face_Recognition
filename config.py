import os
from dotenv import load_dotenv
from pathlib import Path
import cv2
from enum import Enum

# Load environment variables
load_dotenv()

class FaceRecognitionBackend(Enum):
    """Enumeration of supported face recognition backends"""
    ARCFACE = "ArcFace"
    FACENET = "Facenet"
    VGG_FACE = "VGG-Face"
    OPENFACE = "OpenFace"
    DEEPFACE = "DeepFace"

class FaceDetectionBackend(Enum):
    """Enumeration of supported face detection backends"""
    RETINAFACE = "retinaface"
    MTCNN = "mtcnn"
    OPENCV = "opencv"
    SSD = "ssd"
    DLIB = "dlib"
    MEDIAPIPE = "mediapipe"

class Config:
    # Security and Database Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///attendance.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # File Storage Configuration
    BASE_DIR = Path(__file__).parent.parent  # Project root directory
    DATA_DIR = BASE_DIR / 'data'

    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)

    UPLOAD_FOLDER = str(DATA_DIR / 'known_faces')
    BACKUP_FOLDER = str(DATA_DIR / 'backups')
    TEMP_FOLDER = str(DATA_DIR / 'temp')
    LOGS_FOLDER = str(DATA_DIR / 'logs')

    # Ensure subdirectories exist
    Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(BACKUP_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(TEMP_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(LOGS_FOLDER).mkdir(parents=True, exist_ok=True)

    # File Handling Configuration
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB file size limit

    # Face Recognition Settings
    FACE_RECOGNITION_BACKEND = FaceRecognitionBackend(os.getenv('FACE_RECOGNITION_BACKEND', 'ArcFace')).value
    FACE_DETECTION_BACKEND = FaceDetectionBackend(os.getenv('FACE_DETECTION_BACKEND', 'retinaface')).value
    MIN_CONFIDENCE = float(os.getenv('MIN_CONFIDENCE', 0.6))
    ENROLLMENT_QUALITY_THRESHOLD = float(os.getenv('ENROLLMENT_QUALITY_THRESHOLD', 0.7))
    MAX_FACE_WIDTH = int(os.getenv('MAX_FACE_WIDTH', 1000))
    MAX_FACE_HEIGHT = int(os.getenv('MAX_FACE_HEIGHT', 1000))
    FACE_DB_PATH = str(DATA_DIR / 'face_encodings')

    # Performance Settings
    THREAD_POOL_SIZE = int(os.getenv('THREAD_POOL_SIZE', 4))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))

    # Webcam Settings
    CAMERA_BACKEND = os.getenv('CAMERA_BACKEND', 'CAP_DSHOW')  # Default to DShow for Windows
    CAMERA_RESOLUTION = tuple(map(int, os.getenv('CAMERA_RESOLUTION', '1280,720').split(',')))
    CAMERA_FPS = int(os.getenv('CAMERA_FPS', 30))

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = str(Path(LOGS_FOLDER) / 'app.log')

    @staticmethod
    def get_camera_backend():
        """Returns the OpenCV camera backend constant"""
        backends = {
            'CAP_DSHOW': cv2.CAP_DSHOW,
            'CAP_MSMF': cv2.CAP_MSMF,
            'CAP_V4L': cv2.CAP_V4L,
            'CAP_ANY': cv2.CAP_ANY
        }
        return backends.get(Config.CAMERA_BACKEND, cv2.CAP_DSHOW)

    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        if not 0 <= cls.MIN_CONFIDENCE <= 1:
            raise ValueError("MIN_CONFIDENCE must be between 0 and 1")
        if not 0 <= cls.ENROLLMENT_QUALITY_THRESHOLD <= 1:
            raise ValueError("ENROLLMENT_QUALITY_THRESHOLD must be between 0 and 1")
        if cls.THREAD_POOL_SIZE <= 0:
            raise ValueError("THREAD_POOL_SIZE must be positive")
        if cls.BATCH_SIZE <= 0:
            raise ValueError("BATCH_SIZE must be positive")

# Validate configuration on import
Config.validate_config()