#!/usr/bin/env python3
"""Genera performance_report.md dai JSON di benchmark + telemetria tegrastats.

Usage:
  python3 scripts/generate_report.py \
    --standard      perf_standard.json \
    --optimized     perf_optimized-tx2.json \
    --tegra-standard  tegra_standard_parsed.json \
    --tegra-optimized tegra_optimized_parsed.json \
    --output performance_report.md
"""
import argparse
import json
from datetime import datetime
from pathlib import Path


def _load(path):
    if path and Path(path).is_file():
        return json.loads(Path(path).read_text())
    return {}


def _fmt(v, suffix="", na="N/A"):
    return f"{v}{suffix}" if v is not None else na


def _perf_row(perf, tegra):
    providers_active = perf.get("providers_active", [])
    # Preferisce TRT > CUDA > CPU per la colonna "provider attivo"
    gpu_label = (
        next((p for p in providers_active if "Tensorrt" in p or "TensorRT" in p), None)
        or next((p for p in providers_active if "CUDA" in p), None)
        or next((p for p in providers_active), "N/A")
    )

    cyc = perf.get("cycles_per_inference")
    return [
        perf.get("profile", "N/A"),
        perf.get("model_pack", "N/A"),
        gpu_label,
        _fmt(perf.get("total", {}).get("median_ms"), " ms"),
        _fmt(perf.get("total", {}).get("p95_ms"), " ms"),
        _fmt(perf.get("fps_sustained"), " FPS"),
        ("{:.1f} M".format(cyc / 1e6) if cyc else "N/A"),
        _fmt(tegra.get("gpu_gr3d_pct_mean"), "%"),
        _fmt(tegra.get("emc_pct_mean"), "%"),
        _fmt(tegra.get("cpu_pct_mean"), "%"),
        _fmt(tegra.get("power_mw_mean"), " mW"),
        _fmt(tegra.get("temp_gpu_c_peak") or tegra.get("temp_cpu_c_peak"), "°C"),
    ]


def _table(rows, headers):
    # Calcola larghezze colonne
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells):
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    lines = [fmt_row(headers), sep] + [fmt_row(r) for r in rows]
    return "\n".join(lines)


def generate(args):
    std  = _load(args.standard)
    opt  = _load(args.optimized)
    ts   = _load(args.tegra_standard)
    to_  = _load(args.tegra_optimized)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    headers = [
        "Profilo", "Model pack", "Provider attivo",
        "Lat. mediana", "Lat. p95", "FPS",
        "Cicli/inf", "GPU% medio", "EMC% medio", "CPU% medio",
        "Potenza media", "Temp picco",
    ]

    rows = []
    if std:
        rows.append(_perf_row(std, ts))
    if opt:
        rows.append(_perf_row(opt, to_))

    md = f"""\
# Report Prestazioni — Jetson TX2 (JetPack 4.6 / L4T r32.7)

_Generato: {now}_
_Modalità potenza: MAXN (nvpmodel -m 0 + jetson_clocks)_

---

## Tabella riassuntiva

{_table(rows, headers)}

> **Nota colonne**: Latenza = detection + embedding per frame (un volto). FPS sostenuti =
> n_frame / wall_time totale misura (escluso warm-up). Cicli/inf = mediana dei cicli CPU per
> identificazione (perf_event; N/A se il kernel li nega → `--cap-add PERFMON`). EMC% = utilizzo
> del memory controller: su TX2 CPU e GPU condividono la banda LPDDR4 — **EMC saturo con GPU
> scarica = collo di bottiglia di memoria**, non di calcolo. Potenza = VDD_IN (intera scheda).
> Temperatura = picco GPU (o CPU se GPU non disponibile dal sensore).

### Intrusività della telemetria (Tier A vs Tier B)

- **Tier A (sempre attivo nelle sessioni)**: timing per stadio (`time.perf_counter`), lettura
  contatori perf (2 `read()` da 8 byte per frame), delta `/proc/self/io` e VmRSS a fine sessione.
  Overhead misurato dal benchmark: campo `telemetry_overhead_pct` in `perf_<profilo>.json`
  (tipicamente <1–2 %).
- **Tier B (SOLO benchmark dedicati, mai durante i test di accuratezza)**: profiler ONNX Runtime
  per-operatore (`--ort-profile`, sessioni raw con input sintetici) e, opzionale sull'host,
  `trtexec --loadEngine=/mnt/faceid-data/engines/... --dumpProfile` per i tempi per-layer degli
  engine TensorRT. Strumenti classe Nsight possono dimezzare il throughput: mai in sessione.

---

## Fase 1 — Diagnosi GPU vs CPU

### `platform: "arm64-cpu"` NON indica inferenza su CPU

Il campo `platform` nelle sessioni di validazione è generato da `detect_platform_label()`
in `core/metrics.py:307–319`. La funzione controlla il backend della telemetria di sistema
(`_TegrastatsProvider`). **Tegrastats non è accessibile dall'interno del container Docker**,
quindi la funzione cade nel ramo di fallback:

```python
return f"{{_arch()}}-cpu"   # → "arm64-cpu"
```

Questo è un **bug di telemetria**, non un indicatore di inferenza su CPU.

### Prove che la GPU era in uso durante la validazione

| Indicatore | Valore | Interpretazione |
|------------|--------|----------------|
| Container runtime | `nvidia` | GPU visibile nel container |
| `USE_GPU` | `true` | Provider GPU richiesti |
| Engine TRT in `/data/engines/` | `*_fp16.engine` presenti | TensorRT ha compilato su GPU |
| `warmup_last.json: prepare_s` | `0.0` | Engine TRT già cached → load istantaneo |

**Conclusione**: L'inferenza girava su GPU. Il campo `platform: "arm64-cpu"` è un falso
allarme dovuto a tegrastats non accessibile in Docker.

---

## Bug trovato: `OPT_MODEL_PACK=buffalo_l` nel `.env`

Il file `.env` aveva `OPT_MODEL_PACK=buffalo_l` invece del previsto `buffalo_s`. Questo
significa che **durante la campagna di validazione entrambi i profili usavano lo stesso
model pack** (`buffalo_l`). Il confronto accuratezza standard vs ottimizzato misurava
quindi:

- **Standard**: buffalo_l, FP32, CUDA
- **Ottimizzato**: buffalo_l, TRT FP16, CUDA

Anziché il confronto inteso:

- **Standard**: buffalo_l, FP32, CUDA
- **Ottimizzato**: buffalo_s, TRT FP16, CUDA

**Questo benchmark usa `OPT_MODEL_PACK=buffalo_s`** (corretto nel `.env` prima
dell'esecuzione). I risultati qui sono quindi il confronto corretto per il paper.

---

## Nota scientifica — Validità dei dati di accuratezza

I dati di **accuratezza** (sessioni `_standard` e `_optTX2` in `/data/validation/`) sono
stati raccolti con entrambi i profili su `buffalo_l`. Questo non invalida i dati in sé,
ma cambia l'interpretazione:

- I dati di accuratezza del profilo `optTX2` misurano **buffalo_l con TRT FP16**, non
  `buffalo_s`. Quindi confrontano la precisione numerica FP32 vs FP16 sullo stesso modello.
- Per un confronto accuratezza completo (modello pesante vs leggero), occorre rieseguire
  la validazione di accuratezza con `buffalo_s` (stesso video già registrato, nessun nuovo
  soggetto: `docker exec faceid python3 main.py --validate --profile optimized-tx2`).

**Raccomandazione**: Documentare nel paper che il confronto prestazioni (questa tabella)
usa il setup corretto buffalo_l vs buffalo_s, mentre il confronto accuratezza delle
sessioni di validazione è buffalo_l FP32 vs buffalo_l TRT-FP16.

---

## Dettaglio profilo standard

"""
    if std:
        md += f"""\
- ORT version: `{std.get('ort_version', 'N/A')}`
- Provider richiesti: `{std.get('providers_requested', [])}`
- Provider attivi: `{std.get('providers_active', [])}`
- GPU in uso: `{'SI' if std.get('gpu_used') else 'NO'}`
- Video sorgente: `{std.get('video_source', 'N/A')}`
- Frame misurati: {std.get('n_frames', 'N/A')} (warm-up: {std.get('n_warmup', 'N/A')})
- FPS sostenuti: **{std.get('fps_sustained', 'N/A')}**

| Stage | mean | mediana | p95 | min | max |
|-------|------|---------|-----|-----|-----|
| Detection | {std.get('detect',{}).get('mean_ms','N/A')} ms | {std.get('detect',{}).get('median_ms','N/A')} ms | {std.get('detect',{}).get('p95_ms','N/A')} ms | {std.get('detect',{}).get('min_ms','N/A')} ms | {std.get('detect',{}).get('max_ms','N/A')} ms |
| Embedding | {std.get('embed',{}).get('mean_ms','N/A')} ms | {std.get('embed',{}).get('median_ms','N/A')} ms | {std.get('embed',{}).get('p95_ms','N/A')} ms | {std.get('embed',{}).get('min_ms','N/A')} ms | {std.get('embed',{}).get('max_ms','N/A')} ms |
| Total E2E | {std.get('total',{}).get('mean_ms','N/A')} ms | {std.get('total',{}).get('median_ms','N/A')} ms | {std.get('total',{}).get('p95_ms','N/A')} ms | {std.get('total',{}).get('min_ms','N/A')} ms | {std.get('total',{}).get('max_ms','N/A')} ms |

"""
    else:
        md += "_Dati non disponibili._\n\n"

    md += "## Dettaglio profilo optimized-tx2\n\n"
    if opt:
        md += f"""\
- ORT version: `{opt.get('ort_version', 'N/A')}`
- Provider richiesti: `{opt.get('providers_requested', [])}`
- Provider attivi: `{opt.get('providers_active', [])}`
- GPU in uso: `{'SI' if opt.get('gpu_used') else 'NO'}`
- Video sorgente: `{opt.get('video_source', 'N/A')}`
- Frame misurati: {opt.get('n_frames', 'N/A')} (warm-up: {opt.get('n_warmup', 'N/A')})
- FPS sostenuti: **{opt.get('fps_sustained', 'N/A')}**

| Stage | mean | mediana | p95 | min | max |
|-------|------|---------|-----|-----|-----|
| Detection | {opt.get('detect',{}).get('mean_ms','N/A')} ms | {opt.get('detect',{}).get('median_ms','N/A')} ms | {opt.get('detect',{}).get('p95_ms','N/A')} ms | {opt.get('detect',{}).get('min_ms','N/A')} ms | {opt.get('detect',{}).get('max_ms','N/A')} ms |
| Embedding | {opt.get('embed',{}).get('mean_ms','N/A')} ms | {opt.get('embed',{}).get('median_ms','N/A')} ms | {opt.get('embed',{}).get('p95_ms','N/A')} ms | {opt.get('embed',{}).get('min_ms','N/A')} ms | {opt.get('embed',{}).get('max_ms','N/A')} ms |
| Total E2E | {opt.get('total',{}).get('mean_ms','N/A')} ms | {opt.get('total',{}).get('median_ms','N/A')} ms | {opt.get('total',{}).get('p95_ms','N/A')} ms | {opt.get('total',{}).get('min_ms','N/A')} ms | {opt.get('total',{}).get('max_ms','N/A')} ms |

"""
    else:
        md += "_Dati non disponibili._\n\n"

    md += "## Telemetria sistema (tegrastats host)\n\n"
    for label, t in [("standard", ts), ("optimized-tx2", to_)]:
        if t and t.get("n_samples", 0) > 0:
            md += f"""\
### Profilo `{label}`

| Metrica | Valore |
|---------|--------|
| Campioni | {t.get('n_samples', 'N/A')} |
| GPU GR3D medio | {_fmt(t.get('gpu_gr3d_pct_mean'), '%')} |
| GPU GR3D picco | {_fmt(t.get('gpu_gr3d_pct_peak'), '%')} |
| CPU medio | {_fmt(t.get('cpu_pct_mean'), '%')} |
| CPU picco | {_fmt(t.get('cpu_pct_peak'), '%')} |
| RAM picco | {_fmt(t.get('ram_used_mb_peak'), ' MB')} |
| Potenza media ({t.get('power_source', 'N/A')}) | {_fmt(t.get('power_mw_mean'), ' mW')} |
| Potenza picco | {_fmt(t.get('power_mw_peak'), ' mW')} |
| Temp GPU picco | {_fmt(t.get('temp_gpu_c_peak'), '°C')} |
| Temp CPU picco (BCPU) | {_fmt(t.get('temp_bcpu_c_peak') or t.get('temp_mcpu_c_peak') or t.get('temp_cpu_c_peak'), '°C')} |

"""
        else:
            md += f"### Profilo `{label}`\n\n_Telemetria non disponibile._\n\n"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Report scritto: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Genera performance_report.md")
    p.add_argument("--standard",        required=True)
    p.add_argument("--optimized",       required=True)
    p.add_argument("--tegra-standard",  default=None)
    p.add_argument("--tegra-optimized", default=None)
    p.add_argument("--output",          default="performance_report.md")
    args = p.parse_args()
    generate(args)
