import base64
import hashlib
from functools import lru_cache

import numpy as np
from cryptography.fernet import Fernet


_PLACEHOLDER_KEY = "change_me_generate_a_random_hex_key"


@lru_cache(maxsize=1)
def _fernet(secret: str) -> Fernet:
    if not secret or secret == _PLACEHOLDER_KEY:
        raise RuntimeError(
            "BIOMETRIC_SECRET_KEY non configurata: genera una chiave casuale con "
            "`python -c \"import secrets; print(secrets.token_hex(32))\"` e impostala nel file .env"
        )
    raw = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(raw))


def encrypt_embedding(embedding: np.ndarray, secret: str) -> bytes:
    return _fernet(secret).encrypt(embedding.astype(np.float32).tobytes())


def decrypt_embedding(ciphertext: bytes, secret: str) -> np.ndarray:
    raw = _fernet(secret).decrypt(ciphertext)
    return np.frombuffer(raw, dtype=np.float32).copy()
