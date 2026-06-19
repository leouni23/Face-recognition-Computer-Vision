#!/usr/bin/env python
"""Delete validation session artifacts (recorded footage contains images — clean up after analysis).

  python scripts/clear_validation.py --list                 # show sessions
  python scripts/clear_validation.py --session <id>          # delete one session
  python scripts/clear_validation.py --session <id> --video-only   # keep JSONL/metrics, drop the mp4
  python scripts/clear_validation.py --all                   # delete every session (asks confirmation)
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.validation import validation_root


def _sessions():
    root = validation_root()
    return sorted([d for d in root.iterdir() if d.is_dir()]) if root.is_dir() else []


def main() -> None:
    p = argparse.ArgumentParser(description="Pulizia artefatti sessioni di validazione")
    p.add_argument("--list", action="store_true", help="Elenca le sessioni")
    p.add_argument("--session", help="ID sessione da eliminare")
    p.add_argument("--all", action="store_true", help="Elimina tutte le sessioni")
    p.add_argument("--video-only", action="store_true", help="Cancella solo i video .mp4, tieni log e metriche")
    args = p.parse_args()

    sessions = _sessions()

    if args.list or not (args.session or args.all):
        if not sessions:
            print("Nessuna sessione di validazione.")
            return
        print("Sessioni:")
        for d in sessions:
            videos = list(d.rglob("*.mp4"))
            mb = sum(v.stat().st_size for v in videos) / 1e6
            print(f"  {d.name}  ({len(videos)} video, {mb:.1f} MB)")
        return

    targets = sessions if args.all else [validation_root() / args.session]

    if args.all:
        ans = input(f"Eliminare {'i video di ' if args.video_only else ''}TUTTE le {len(targets)} sessioni? [s/N]: ")
        if ans.strip().lower() not in ("s", "si", "sì", "y", "yes"):
            print("Annullato.")
            return

    for d in targets:
        if not d.is_dir():
            print(f"Non trovata: {d.name}")
            continue
        if args.video_only:
            for v in d.rglob("*.mp4"):
                v.unlink()
            print(f"Video eliminati: {d.name}")
        else:
            shutil.rmtree(d)
            print(f"Sessione eliminata: {d.name}")


if __name__ == "__main__":
    main()
