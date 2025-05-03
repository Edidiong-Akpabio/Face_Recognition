from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pickle
import numpy as np
from sqlalchemy import TypeDecorator, Text

class NumpyArray(TypeDecorator):
    """Converts numpy arrays to TEXT for database storage"""
    impl = Text

    def process_bind_param(self, value, dialect):
        if value is not None:
            # Convert numpy array to bytes then to base64 string
            return pickle.dumps(value)
        return None

    def process_result_value(self, value, dialect):
        if value is not None:
            # Convert base64 string back to numpy array
            return pickle.loads(value)
        return None

db = SQLAlchemy()

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    class_name = db.Column(db.String(50), nullable=False)
    section = db.Column(db.String(10), nullable=False)
    roll_number = db.Column(db.Integer, nullable=False)
    face_embedding = db.Column(NumpyArray, nullable=True)
    photo_path = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    entry_time = db.Column(db.DateTime, nullable=False)
    exit_time = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='Present')

class SystemLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    admin_id = db.Column(db.Integer, db.ForeignKey('admin.id'), nullable=True)
    action = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(50))