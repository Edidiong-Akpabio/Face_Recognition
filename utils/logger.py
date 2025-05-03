from datetime import datetime
from models.database import SystemLog, db

def log_action(admin_id, action, ip_address):
    """Log an admin action to the database"""
    log_entry = SystemLog(
        admin_id=admin_id,
        action=action,
        ip_address=ip_address,
        timestamp=datetime.utcnow()
    )
    db.session.add(log_entry)
    db.session.commit()