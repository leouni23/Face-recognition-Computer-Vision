#!/usr/bin/env python
"""Offline Phase-1 (Standard) vs Phase-2 (Optimized-TX2) comparison from saved sessions.

Runs purely from the validation artifacts on the external disk — no camera re-run. Groups
sessions by profile (optionally also by condition preset), recomputes the open-set 1:N metrics
per group, and emits a side-by-side comparison + deltas as JSON/CSV for the paper.

  python scripts/compare_sessions.py --list                  # list discovered sessions
  python scripts/compare_sessions.py                         # print comparison to stdout
  python scripts/compare_sessions.py --by-preset             # also split by condition preset
  python scripts/compare_sessions.py --json out.json --csv out.csv
  python scripts/compare_sessions.py --root /data/validation # explicit external root
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.compare import compare, scan_sessions, to_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Confronto cross-sessione Standard vs Optimized-TX2")
    ap.add_argument("--root", default=None, help="root validation (default: settings/VALIDATION_DIR)")
    ap.add_argument("--by-preset", action="store_true", help="suddividi anche per preset di condizione")
    ap.add_argument("--list", action="store_true", help="elenca solo le sessioni trovate")
    ap.add_argument("--json", default=None, help="scrivi il confronto completo in un file JSON")
    ap.add_argument("--csv", default=None, help="scrivi il riepilogo per profilo in un file CSV")
    args = ap.parse_args()
    root = Path(args.root) if args.root else None

    if args.list:
        rows = scan_sessions(root)
        for s in rows:
            print(f"{s['session_id']}\t{s['profile']}\t{s['session_type']}\t{s.get('start')}")
        print(f"\n{len(rows)} sessione/i.")
        return

    cmp = compare(root, by_preset=args.by_preset)
    if args.json:
        Path(args.json).write_text(json.dumps(cmp, indent=2, default=str), encoding="utf-8")
        print(f"JSON → {args.json}")
    if args.csv:
        Path(args.csv).write_text(to_csv(cmp), encoding="utf-8")
        print(f"CSV  → {args.csv}")

    print(f"\nRoot: {cmp['root']} — {cmp['n_sessions']} sessione/i\n")
    print(json.dumps(cmp["profiles"], indent=2, default=str))
    print("\nΔ (optimized − standard):", json.dumps(cmp["deltas_optimized_minus_standard"]))
    print("\n" + to_csv(cmp))


if __name__ == "__main__":
    main()
