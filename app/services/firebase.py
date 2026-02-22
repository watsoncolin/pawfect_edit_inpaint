import base64
import datetime
import json
import logging
import os

import firebase_admin
from firebase_admin import credentials, firestore, storage

logger = logging.getLogger(__name__)

_initialized = False


def _init():
    global _initialized
    if _initialized:
        return
    creds_b64 = os.environ["FIREBASE_CREDENTIALS"]
    creds_json = json.loads(base64.b64decode(creds_b64))
    cred = credentials.Certificate(creds_json)
    firebase_admin.initialize_app(cred, {"storageBucket": f"{creds_json['project_id']}.firebasestorage.app"})
    _initialized = True
    logger.info("Firebase Admin SDK initialized")


def get_firestore_client():
    _init()
    return firestore.client()


def get_storage_bucket():
    _init()
    return storage.bucket()


def download_blob(path: str) -> bytes:
    """Download a file from Firebase Storage by path."""
    bucket = get_storage_bucket()
    blob = bucket.blob(path)
    data = blob.download_as_bytes()
    logger.info(f"Downloaded {path}: {len(data)} bytes")
    return data


def upload_blob(path: str, data: bytes, content_type: str):
    """Upload a file to Firebase Storage."""
    bucket = get_storage_bucket()
    blob = bucket.blob(path)
    blob.upload_from_string(data, content_type=content_type)
    logger.info(f"Uploaded {path}: {len(data)} bytes")
    return blob.name


def session_exists(user_id: str, session_id: str) -> bool:
    """Check if a session document exists in Firestore."""
    db = get_firestore_client()
    doc_ref = db.collection("users").document(user_id).collection("sessions").document(session_id)
    return doc_ref.get().exists


def update_session(user_id: str, session_id: str, updates: dict):
    """Update Firestore session document."""
    db = get_firestore_client()
    doc_ref = db.collection("users").document(user_id).collection("sessions").document(session_id)
    doc_ref.update(updates)
    logger.info(f"Updated Firestore session {session_id}: {list(updates.keys())}")


def generate_signed_url(path: str, expiry_hours: int = 72) -> str:
    """Generate a V4 signed URL for a Firebase Storage blob."""
    bucket = get_storage_bucket()
    blob = bucket.blob(path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(hours=expiry_hours),
        method="GET",
    )
    return url


def upload_and_sign(path: str, data: bytes, content_type: str, expiry_hours: int = 72) -> str:
    """Upload bytes to Firebase Storage and return a signed URL."""
    upload_blob(path, data, content_type)
    return generate_signed_url(path, expiry_hours)
