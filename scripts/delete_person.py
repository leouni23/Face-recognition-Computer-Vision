#!/usr/bin/env python
"""GDPR Art. 17 — Right to erasure: permanently delete all biometric data for a person."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from database.repository import PersonRepository
from database.session import get_session


def delete_person(name: str) -> None:
    confirm = input(
        f"Eliminare PERMANENTEMENTE tutti i dati biometrici per '{name}'? [s/N]: "
    ).strip().lower()
    if confirm not in ("s", "si", "sì", "y", "yes"):
        print("Operazione annullata.")
        sys.exit(0)

    with get_session() as session:
        count = PersonRepository(session).delete_person(name)

    if count:
        logger.success(f"Dati biometrici di '{name}' eliminati (GDPR Art. 17). Record rimossi: {count}")
        print(f"✓ {count} record eliminato/i per '{name}'.")
    else:
        print(f"Nessun record trovato per '{name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancella dati biometrici (GDPR Art. 17)")
    parser.add_argument("--name", "-n", required=True, help="Nome della persona da eliminare")
    args = parser.parse_args()
    delete_person(args.name)
