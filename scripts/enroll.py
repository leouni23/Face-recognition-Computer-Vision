#!/usr/bin/env python
"""Enroll a new person: capture face samples via webcam and store encrypted templates."""
import argparse
import sys

import cv2
import face_recognition
import numpy as np
from loguru import logger

from config.settings import get_settings
from core.camera import CameraStream
from database.repository import PersonRepository
from database.session import get_session
from database.models import Base
from database.session import engine

_settings = get_settings()

_CONSENT_NOTICE = f"""
╔══════════════════════════════════════════════════════════════════╗
║         INFORMATIVA SUL TRATTAMENTO DEI DATI BIOMETRICI          ║
╠══════════════════════════════════════════════════════════════════╣
║  Ai sensi del GDPR (Reg. UE 2016/679), Art. 9:                  ║
║                                                                  ║
║  • Vengono acquisiti dati biometrici (template del volto)        ║
║  • I template sono cifrati con AES-128 + HMAC (Fernet)           ║
║  • Nessuna immagine originale viene conservata                   ║
║  • Conservazione massima: {_settings.data_retention_days} giorni dalla data di iscrizione  ║
║  • Diritto di cancellazione: scripts/delete_person.py --name ... ║
║  • Base giuridica: consenso esplicito dell'interessato           ║
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
    encodings: list[np.ndarray] = []

    print(f"\nAcquisizione volto per '{name}'.")
    print(f"Premi SPAZIO per catturare un campione ({num_samples} necessari). Q per annullare.\n")
    cv2.namedWindow("Enrollment — Face ID")

    while len(encodings) < num_samples:
        frame = cam.read(timeout=0.5)
        if frame is None:
            continue

        rgb = frame[:, :, ::-1]
        locs = face_recognition.face_locations(rgb, model="hog")
        display = frame.copy()

        for top, right, bottom, left in locs:
            cv2.rectangle(display, (left, top), (right, bottom), (34, 197, 94), 2)

        progress = f"Campioni: {len(encodings)}/{num_samples}"
        cv2.putText(display, progress, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (34, 197, 94), 2)
        cv2.imshow("Enrollment — Face ID", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Iscrizione annullata dall'utente.")
            cam.stop()
            cv2.destroyAllWindows()
            sys.exit(0)

        if key == ord(" "):
            if not locs:
                print("  Nessun volto rilevato. Avvicinati e riprova.")
                continue
            if len(locs) > 1:
                print("  Più volti nell'inquadratura. Assicurati di essere solo.")
                continue
            encs = face_recognition.face_encodings(rgb, known_face_locations=locs)
            if encs:
                encodings.append(encs[0])
                print(f"  Campione {len(encodings)}/{num_samples} acquisito.")

    cam.stop()
    cv2.destroyAllWindows()

    mean_encoding = np.mean(encodings, axis=0).astype(np.float32)

    with get_session() as session:
        repo = PersonRepository(session)
        person = repo.get_by_name(name)
        if person:
            print(f"Persona '{name}' già presente — aggiornamento template.")
        else:
            person = repo.create(name)
        repo.give_consent(person)
        repo.add_template(person, mean_encoding)

    logger.success(f"Iscrizione completata per '{name}'.")
    print(f"\n✓ Persona '{name}' iscritta con successo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iscrivi una nuova persona nel sistema Face ID")
    parser.add_argument("--name", "-n", required=True, help="Nome e cognome della persona")
    parser.add_argument("--samples", "-s", type=int, default=5, help="Numero di campioni (default: 5)")
    parser.add_argument("--camera", "-c", default="0", help="Sorgente camera (default: 0)")
    args = parser.parse_args()
    enroll(args.name, args.samples, args.camera)
