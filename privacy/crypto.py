import base64
import hashlib

import numpy as np
from cryptography.fernet import Fernet


def _fernet(secret: str) -> Fernet:
    # Derive a 32-byte key from the secret so the key length is always valid for Fernet
    raw = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(raw))


def encrypt_embedding(embedding: np.ndarray, secret: str) -> bytes:
    return _fernet(secret).encrypt(embedding.astype(np.float32).tobytes())


def decrypt_embedding(ciphertext: bytes, secret: str) -> np.ndarray:
    raw = _fernet(secret).decrypt(ciphertext)
    return np.frombuffer(raw, dtype=np.float32).copy()
