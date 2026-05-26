#!/usr/bin/env python
"""Enroll a new person: capture face samples via webcam and store encrypted templates."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from loguru import logger

from config.settings import get_settings
from core.camera import CameraStream
from core.detector import detect_and_encode
from database.models import Base
from database.repository import PersonRepository
from database.session import engine, get_session

_settings = get_settings()

_CONSENT_NOTICE = f"""
╔══════════════════════════════════════════════════════════════════╗
║         INFORMATIVA SUL TRATTAMENTO DEI DATI BIOMETRICI          ║
╠══════════════════════════════════════════════════════════════════╣
║  Ai sensi del GDPR (Reg. UE 2016/679), Art. 9:                  ║
║                                                                  ║
║  • Vengono acquisiti dati biometrici (embedding ArcFace 512-d)  ║
║  • I template sono cifrati con AES-128 + HMAC (Fernet)           ║
║  • Nessuna immagine originale viene conservata                   ║
║  • Conservazione massima: {_settings.data_retention_days} giorni                       ║
║  • Diritto di cancellazione: scripts/delete_person.py --name ... ║
╚══════════════════════════════════════════════════════════════════╝
"""


def enroll(name: str, num_samples: int = 5, camera_source: str = "0") -> None:
    Base.metadata.create_all(engine)

    print(_CONSENT_NOTICE)
    answer = input("Acconsento al trattamento dei miei dati biometrici? [s/N]: ").strip().lower()
    if answer not in ("s", "si", "sì", "y", "yes"):
        print("Consenso negato. Iscrizione annullata.")
        sys.exit(0)

    cam = CameraStream(camera_source).start()
    embeddings: list[np.ndarray] = []

    print(f"\nAcquisizione volto per '{name}'.")
    print(f"Premi SPAZIO per catturare un campione ({num_samples} necessari). Q per annullare.\n")
    cv2.namedWindow("Enrollment — Face ID")

    while len(embeddings) < num_samples:
        frame = cam.read(timeout=0.5)
        if frame is None:
            continue

        face_data = detect_and_encode(frame)
        display = frame.copy()

        for (top, right, bottom, left), _ in face_data:
            cv2.rectangle(display, (left, top), (right, bottom), (34, 197, 94), 2)

        progress = f"Campioni: {len(embeddings)}/{num_samples}"
        cv2.putText(display, progress, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 197, 94), 2)
        cv2.imshow("Enrollment — Face ID", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Iscrizione annullata dall'utente.")
            cam.stop()
            cv2.destroyAllWindows()
            sys.exit(0)

        if key == ord(" "):
            if not face_data:
                print("  Nessun volto rilevato. Avvicinati e riprova.")
                continue
            if len(face_data) > 1:
                print("  Più volti nell'inquadratura. Assicurati di essere solo.")
                continue
            _, embedding = face_data[0]
            embeddings.append(embedding)
            print(f"  Campione {len(embeddings)}/{num_samples} acquisito.")

    cam.stop()
    cv2.destroyAllWindows()

    mean_embedding = np.mean(embeddings, axis=0).astype(np.float32)

    with get_session() as session:
        repo = PersonRepository(session)
        person = repo.get_by_name(name)
        if person:
            print(f"Persona '{name}' già presente — aggiornamento template.")
        else:
            person = repo.create(name)
        repo.give_consent(person)
        repo.add_template(person, mean_embedding)

    logger.success(f"Iscrizione completata per '{name}'.")
    print(f"\n✓ Persona '{name}' iscritta con successo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iscrivi una nuova persona nel sistema Face ID")
    parser.add_argument("--name", "-n", required=True, help="Nome e cognome della persona")
    parser.add_argument("--samples", "-s", type=int, default=5, help="Numero di campioni (default: 5)")
    parser.add_argument("--camera", "-c", default="0", help="Sorgente camera (default: 0)")
    args = parser.parse_args()
    enroll(args.name, args.samples, args.camera)
