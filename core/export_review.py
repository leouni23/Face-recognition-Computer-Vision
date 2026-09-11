"""Final export of a reviewed campaign — the dataset the paper is written from.

Runs only over the sessions the operator kept (excluded repetitions and excluded campaigns are
filtered out), recomputes the per-profile open-set metrics with the existing machinery
(`core.validation_metrics` via `core.compare._combine_metrics`) and writes a self-contained,
reproducible folder: same input → same output, with a MANIFEST that states exactly what was
included and what was left out.
"""
import csv
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from core.compare import _combine_metrics, scan_sessions
from core.review import excluded_sessions, load_state, review_root

_PROFILES = ("standard", "optimized-tx2")


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(Path(__file__).resolve().parent.parent),
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return out.stdout.decode().strip() or None
    except Exception:
        return None


def _pct(block: Optional[dict]) -> str:
    """A rate as 'x.xx% (k/N, IC95 lo–hi)' — a percentage without its denominator and CI is
    not reportable in a paper."""
    if not block or block.get("n") in (None, 0):
        return "n/d"
    val, k, n = block.get("value"), block.get("errors"), block.get("n")
    ci = block.get("ci95") or [None, None]
    txt = "{:.2f}% ({}/{})".format(100.0 * val, k, n) if val is not None else "n/d ({}/{})".format(k, n)
    if ci[0] is not None and ci[1] is not None:
        txt += ", IC95 {:.2f}–{:.2f}%".format(100.0 * ci[0], 100.0 * ci[1])
    if block.get("rule_of_3_upper") is not None:
        # 3/N is only meaningful while it stays below 1: with a handful of events it would print
        # an absurd "≤300%". Clamp and let the caveat about the tiny sample carry the message.
        txt += ", ≤{:.2f}% (regola del 3)".format(100.0 * min(1.0, block["rule_of_3_upper"]))
    return txt


def _num(v, digits=4):
    return round(v, digits) if isinstance(v, (int, float)) else None


def _det_csv(metrics: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["threshold", "fpir", "fnir", "mate_events", "non_mate_events"])
    for p in metrics.get("det_curve") or []:
        w.writerow([_num(p.get("threshold")), _num(p.get("fpir"), 6), _num(p.get("fnir"), 6),
                    p.get("mate_n"), p.get("non_n")])
    return buf.getvalue()


def _cmc_csv(metrics: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "identification_rate"])
    for p in metrics.get("cmc") or []:
        w.writerow([p.get("rank"), _num(p.get("rate"), 6)])
    return buf.getvalue()


def _comparison_csv(per_profile: Dict[str, dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["profile", "sessions", "mate_events", "non_mate_events",
                "fpir", "fpir_ci_lo", "fpir_ci_hi", "fnir", "fnir_ci_lo", "fnir_ci_hi",
                "eer", "rank1"])
    for prof in sorted(per_profile):
        e = per_profile[prof]
        m = e.get("metrics") or {}
        c = (m.get("counts") or {})
        fp, fn = m.get("fpir") or {}, m.get("fnir") or {}
        ci_p, ci_n = fp.get("ci95") or [None, None], fn.get("ci95") or [None, None]
        w.writerow([prof, e.get("n_sessions"), c.get("mate_events"), c.get("non_mate_events"),
                    _num(fp.get("value"), 6), _num(ci_p[0], 6), _num(ci_p[1], 6),
                    _num(fn.get("value"), 6), _num(ci_n[0], 6), _num(ci_n[1], 6),
                    _num((m.get("eer") or {}).get("value"), 6), _num(m.get("rank1"), 6)])
    return buf.getvalue()


def _telemetry(sessions: List[dict]) -> dict:
    """Per-profile hardware aggregates recorded in the session manifests (if present)."""
    out: Dict[str, dict] = {}
    for s in sessions:
        try:
            meta = json.loads((Path(s["dir"]) / "session.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        entry = out.setdefault(s.get("profile") or "standard",
                               {"sessions": 0, "energy_per_frame_uj": [], "hw": []})
        entry["sessions"] += 1
        epf = (meta.get("energy_per_frame") or {}).get("mean_uj")
        if isinstance(epf, (int, float)):
            entry["energy_per_frame_uj"].append(epf)
        if meta.get("hw_aggregate"):
            entry["hw"].append(meta["hw_aggregate"])
    for prof, entry in out.items():
        vals = entry.pop("energy_per_frame_uj")
        entry["energy_per_frame_uj_mean"] = round(sum(vals) / len(vals), 1) if vals else None
        hw = entry.pop("hw")
        merged: Dict[str, List[float]] = {}
        for agg in hw:
            for ch, st in agg.items():
                if isinstance(st, dict) and isinstance(st.get("mean"), (int, float)):
                    merged.setdefault(ch, []).append(st["mean"])
        entry["hw_mean"] = {ch: round(sum(v) / len(v), 2) for ch, v in sorted(merged.items())}
    return out


def _report_md(ctx: dict) -> str:
    per = ctx["per_profile"]
    lines = [
        "# Report di validazione — {}".format(ctx["campaign_name"]),
        "",
        "_Export: {} · commit `{}` · root dati: `{}`_".format(
            ctx["exported_at"], ctx["git_commit"] or "n/d", ctx["root"]),
        "",
        "Campagna `{}` — **{} sessioni incluse**, {} escluse.".format(
            ctx["campaign_folder"], ctx["n_included"], ctx["n_excluded"]),
        "",
        "## Risultati per profilo (open-set 1:N, per evento)",
        "",
        "| Profilo | Sessioni | Eventi mate | Eventi non-mate | FPIR | FNIR | EER | Rank-1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for prof in sorted(per):
        e = per[prof]
        m = e.get("metrics") or {}
        c = m.get("counts") or {}
        eer = (m.get("eer") or {}).get("value")
        r1 = m.get("rank1")
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
            prof, e.get("n_sessions"), c.get("mate_events"), c.get("non_mate_events"),
            _pct(m.get("fpir")), _pct(m.get("fnir")),
            "{:.2f}%".format(100.0 * eer) if isinstance(eer, (int, float)) else "n/d",
            "{:.2f}%".format(100.0 * r1) if isinstance(r1, (int, float)) else "n/d"))
    deltas = ctx.get("deltas") or {}
    if deltas:
        lines += ["", "## Confronto Optimized − Standard", "",
                  "| Metrica | Δ (punti percentuali) |", "| --- | --- |"]
        for k, v in sorted(deltas.items()):
            lines.append("| {} | {} |".format(
                k.upper(), "{:+.2f}".format(100.0 * v) if isinstance(v, (int, float)) else "n/d"))
    tel = ctx.get("telemetry") or {}
    if tel:
        lines += ["", "## Telemetria hardware (media per profilo)", "",
                  "| Profilo | Sessioni | Energia/frame (µJ) | Canali hw |", "| --- | --- | --- | --- |"]
        for prof in sorted(tel):
            t = tel[prof]
            hw = ", ".join("{}={}".format(k, v) for k, v in list(t.get("hw_mean", {}).items())[:4])
            lines.append("| {} | {} | {} | {} |".format(
                prof, t.get("sessions"), t.get("energy_per_frame_uj_mean") or "n/d", hw or "n/d"))
    caveats = []
    for prof in sorted(per):
        for c in ((per[prof].get("metrics") or {}).get("caveats") or []):
            if c not in caveats:
                caveats.append(c)
    lines += ["", "## Limiti dichiarati", ""]
    lines += ["- {}".format(c) for c in caveats] or ["- Nessun limite automatico segnalato."]
    lines += [
        "- Galleria di dimensioni ridotte: i tassi open-set dipendono dalla numerosità della "
        "galleria e non sono direttamente confrontabili con benchmark su gallerie grandi.",
        "",
        "## Sessioni escluse",
        "",
    ]
    if ctx["excluded_detail"]:
        lines += ["| Sessione | Motivo |", "| --- | --- |"]
        lines += ["| {} | {} |".format(sid, reason or "—")
                  for sid, reason in sorted(ctx["excluded_detail"].items())]
    else:
        lines.append("Nessuna sessione esclusa.")
    lines += ["", "---", "",
              "Ogni percentuale riporta numeratore/denominatore e intervallo di confidenza di "
              "Wilson al 95%. FPIR/FNIR sono calcolati per evento (un attraversamento = un "
              "evento), non per frame.", ""]
    return "\n".join(lines)


def export_campaign(folder: str, root: Optional[Path] = None) -> dict:
    """Build the export folder for a reviewed campaign. Returns a summary with its path."""
    root = Path(root) if root else review_root()
    camp_dir = root / folder
    if not (camp_dir.is_dir() and (camp_dir / "campaign.json").is_file()):
        raise ValueError("Campagna non trovata: {}".format(folder))
    try:
        camp_meta = json.loads((camp_dir / "campaign.json").read_text(encoding="utf-8"))
    except Exception:
        camp_meta = {}
    state = load_state(folder)
    excluded = excluded_sessions(folder)

    sessions = [s for s in scan_sessions(root) if s.get("group") == folder]
    included = [s for s in sessions if s["session_id"] not in excluded]
    if not included:
        raise ValueError("Nessuna sessione inclusa: non c'è nulla da esportare")

    per_profile: Dict[str, dict] = {}
    for prof in sorted({s.get("profile") or "standard" for s in included}):
        subset = sorted((s for s in included if (s.get("profile") or "standard") == prof),
                        key=lambda s: s["session_id"])
        metrics = _combine_metrics(subset)
        per_profile[prof] = {"n_sessions": len(subset), "metrics": metrics,
                             "sessions": [s["session_id"] for s in subset]}

    deltas = {}
    std, opt = per_profile.get("standard"), per_profile.get("optimized-tx2")
    if std and opt and std.get("metrics") and opt.get("metrics"):
        for k in ("fpir", "fnir"):
            a = (std["metrics"].get(k) or {}).get("value")
            b = (opt["metrics"].get(k) or {}).get("value")
            deltas[k] = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        for k in ("eer",):
            a = (std["metrics"].get(k) or {}).get("value")
            b = (opt["metrics"].get(k) or {}).get("value")
            deltas[k] = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        a, b = std["metrics"].get("rank1"), opt["metrics"].get("rank1")
        deltas["rank1"] = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None

    stamp = datetime.now()
    out_dir = root / "export" / "{}_{}".format(folder, stamp.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, text: str) -> None:
        (out_dir / name).write_text(text, encoding="utf-8")

    for prof, entry in per_profile.items():
        m = entry.get("metrics") or {}
        _write("metrics_{}.json".format(prof),
               json.dumps(m, indent=2, ensure_ascii=False, sort_keys=True))
        _write("det_{}.csv".format(prof), _det_csv(m))
        _write("cmc_{}.csv".format(prof), _cmc_csv(m))

    comparison = {
        "campaign": folder, "campaign_name": camp_meta.get("name", folder),
        "profiles": {p: {"n_sessions": e["n_sessions"], "sessions": e["sessions"],
                         "fpir": (e.get("metrics") or {}).get("fpir"),
                         "fnir": (e.get("metrics") or {}).get("fnir"),
                         "eer": (e.get("metrics") or {}).get("eer"),
                         "rank1": (e.get("metrics") or {}).get("rank1")}
                     for p, e in sorted(per_profile.items())},
        "deltas_optimized_minus_standard": deltas,
    }
    _write("comparison.json", json.dumps(comparison, indent=2, ensure_ascii=False, sort_keys=True))
    _write("comparison.csv", _comparison_csv(per_profile))
    _write("review_state.json", json.dumps(state, indent=2, ensure_ascii=False, sort_keys=True))

    telemetry = _telemetry(included)
    if telemetry:
        _write("telemetry.json", json.dumps(telemetry, indent=2, ensure_ascii=False, sort_keys=True))

    excluded_detail = {sid: (state["sessions"].get(sid) or {}).get("reason", "")
                       for sid in sorted(excluded)}
    manifest = {
        "exported_at": stamp.isoformat(),
        "root": str(root),
        "campaign": {"folder": folder, "name": camp_meta.get("name", folder),
                     "created_at": camp_meta.get("created_at"),
                     "closed_at": camp_meta.get("closed_at")},
        "git_commit": _git_commit(),
        "sessions_included": sorted(s["session_id"] for s in included),
        "sessions_excluded": excluded_detail,
        "n_included": len(included), "n_excluded": len(excluded),
        "profiles": sorted(per_profile.keys()),
        "reviewed": sum(1 for s in included
                        if (state["sessions"].get(s["session_id"]) or {}).get("reviewed")),
        "schema": "faceid-review-export/1",
    }
    _write("MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))
    _write("REPORT.md", _report_md({
        "campaign_name": camp_meta.get("name", folder), "campaign_folder": folder,
        "exported_at": stamp.strftime("%Y-%m-%d %H:%M:%S"), "git_commit": manifest["git_commit"],
        "root": str(root), "per_profile": per_profile, "deltas": deltas,
        "telemetry": telemetry, "n_included": len(included), "n_excluded": len(excluded),
        "excluded_detail": excluded_detail,
    }))
    logger.success("[Review] Export completato: {} ({} sessioni incluse)".format(out_dir, len(included)))
    return {"dir": str(out_dir), "n_included": len(included), "n_excluded": len(excluded),
            "profiles": sorted(per_profile.keys()), "manifest": manifest}
