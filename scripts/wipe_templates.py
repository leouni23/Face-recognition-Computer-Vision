#!/usr/bin/env python
"""Delete ALL face templates (keeps persons + consent). One-off cleanup for a DB polluted with
mixed/untagged embeddings from different recognition models. Re-enroll afterwards.

  python scripts/wipe_templates.py            # asks confirmation
  python scripts/wipe_templates.py --yes      # no prompt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.repository import PersonRepository
from database.session import get_session


def main() -> None:
    ap = argparse.ArgumentParser(description="Wipe all face templates (clean re-enroll)")
    ap.add_argument("--yes", action="store_true", help="non chiedere conferma")
    args = ap.parse_args()
    if not args.yes:
        ans = input("Cancellare TUTTI i template volto? Le persone restano. [s/N] ").strip().lower()
        if ans not in ("s", "y"):
            print("Annullato.")
            return
    with get_session() as session:
        n = PersonRepository(session).delete_all_templates()
    print(f"Template eliminati: {n}. Ri-registra i volti sotto il profilo attivo.")


if __name__ == "__main__":
    main()
