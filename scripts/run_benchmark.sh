#!/bin/bash
# ============================================================
# Benchmark prestazionale profili InsightFace — Jetson TX2
# Eseguire sull'HOST (non dentro il container).
#
# Prerequisiti:
#   - Docker image faceid:jetson-tx2 presente
#   - Dati in /mnt/faceid-data (models + validation + engines)
#   - sudo senza password (o eseguire come root)
#
# Uso:
#   cd /home/leo/Face-recognition-Computer-Vision
#   bash scripts/run_benchmark.sh
#   bash scripts/run_benchmark.sh --n-frames 100   # run rapido per test
# ============================================================
set -euo pipefail

# ── Parametri configurabili ─────────────────────────────────
DATA="/mnt/faceid-data"
IMAGE="faceid:jetson-tx2"
N_FRAMES=300
WARMUP=20
TEGRA_INTERVAL=200        # ms campionamento tegrastats

# Directory output (su disco esterno per spazio)
OUT_ROOT="$DATA/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="$OUT_ROOT/bench_$TIMESTAMP"

# Path DENTRO il container (volumi montati sotto)
CONTAINER_APP="/app"
CONTAINER_DATA="/data"
CONTAINER_ENGINES="/data/engines"

# ── Parsing argomenti CLI ────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-frames) N_FRAMES="$2"; shift 2 ;;
        --warmup)   WARMUP="$2";   shift 2 ;;
        --out)      OUT="$2";      shift 2 ;;
        *) echo "Argomento sconosciuto: $1"; exit 1 ;;
    esac
done

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "============================================================"
echo " Benchmark Jetson TX2 — $(date)"
echo " N_FRAMES=$N_FRAMES  WARMUP=$WARMUP"
echo " Output: $OUT"
echo "============================================================"
echo ""

# ── Crea directory output ────────────────────────────────────
mkdir -p "$OUT"

# ── MAXN ─────────────────────────────────────────────────────
echo "[MAXN] Applico modalità potenza massima..."
sudo nvpmodel -m 0
sudo jetson_clocks
echo ""
echo "[MAXN] Modalità attiva:"
sudo nvpmodel -q
echo ""

# ── Fase 1: diagnosi provider ────────────────────────────────
echo "============================================================"
echo " FASE 1 — Diagnosi provider onnxruntime"
echo "============================================================"
sudo docker run --rm --runtime nvidia --entrypoint "" \
    -v "$DATA/models:/data/models:ro" \
    -e INSIGHTFACE_HOME=/data/models \
    "$IMAGE" python3 -c "
import onnxruntime as ort
import platform, sys
print('Python       :', sys.version.split()[0])
print('Platform     :', platform.machine())
print('ORT version  :', ort.__version__)
avail = ort.get_available_providers()
print('Providers    :', avail)
gpu = any('CUDA' in p or 'Tensorrt' in p or 'TensorRT' in p for p in avail)
print('GPU disponibile:', 'SI' if gpu else 'NO — onnxruntime CPU wheel, serve onnxruntime-gpu')
" 2>&1 | tee "$OUT/fase1_provider_check.txt"
echo ""

# Trova un video di validazione da usare come sorgente frame
VIDEO_HOST=$(find "$DATA/validation" -name "cam_0_000.mp4" 2>/dev/null | sort | head -1 || true)
if [ -z "$VIDEO_HOST" ]; then
    echo "[WARN] Nessun video trovato in $DATA/validation — il benchmark userà frame sintetici"
    VIDEO_CONTAINER=""
else
    # Costruisce il path DENTRO il container (volume: $DATA/validation → /data/validation)
    VIDEO_REL="${VIDEO_HOST#$DATA/validation/}"
    VIDEO_CONTAINER="$CONTAINER_DATA/validation/$VIDEO_REL"
    echo "[INFO] Video sorgente: $VIDEO_HOST"
    echo "[INFO] Path nel container: $VIDEO_CONTAINER"
fi
echo ""

# ── Funzione benchmark per un profilo ────────────────────────
run_profile() {
    local PROF="$1"
    local SAFE_PROF="${PROF//-/_}"   # "optimized-tx2" → "optimized_tx2" per nomi file

    echo "============================================================"
    echo " BENCHMARK: profilo=$PROF"
    echo "============================================================"

    PERF_OUT_CONTAINER="$CONTAINER_DATA/benchmarks/bench_$TIMESTAMP/perf_${SAFE_PROF}.json"
    PERF_OUT_HOST="$OUT/perf_${SAFE_PROF}.json"
    TEGRA_LOG="$OUT/tegra_${SAFE_PROF}.txt"

    # Avvia tegrastats sull'host mentre gira il container
    echo "[tegrastats] Avvio campionamento (interval=${TEGRA_INTERVAL}ms) → $TEGRA_LOG"
    sudo tegrastats --interval "$TEGRA_INTERVAL" --logfile "$TEGRA_LOG" &
    TPID=$!

    # Attendi qualche campione iniziale
    sleep 1

    # Video arg (opzionale)
    VIDEO_ARG=""
    if [ -n "$VIDEO_CONTAINER" ]; then
        VIDEO_ARG="--video $VIDEO_CONTAINER"
    fi

    echo "[docker] Avvio container benchmark profilo=$PROF..."
    sudo docker run --rm --runtime nvidia --entrypoint "" \
        -v "$DATA/models:/data/models:ro" \
        -v "$DATA/validation:/data/validation:ro" \
        -v "$DATA/engines:/data/engines" \
        -v "$OUT_ROOT:/data/benchmarks" \
        -v "$APP_DIR:/app:ro" \
        -e INSIGHTFACE_HOME=/data/models \
        -e DATA_DIR=/data \
        "$IMAGE" python3 /app/scripts/benchmark_profili.py \
            --profile "$PROF" \
            $VIDEO_ARG \
            --n-frames "$N_FRAMES" \
            --warmup "$WARMUP" \
            --engine-cache "$CONTAINER_ENGINES" \
            --output "$PERF_OUT_CONTAINER" \
        2>&1 | tee "$OUT/benchmark_${SAFE_PROF}.log"

    echo "[tegrastats] Stop..."
    sudo pkill tegrastats 2>/dev/null || true
    wait "$TPID" 2>/dev/null || true
    sleep 1  # flush file

    # Copia il JSON se non è già nel path host (potrebbe essere lo stesso percorso)
    if [ -f "$PERF_OUT_CONTAINER" ] && [ "$PERF_OUT_CONTAINER" != "$PERF_OUT_HOST" ]; then
        cp "$PERF_OUT_CONTAINER" "$PERF_OUT_HOST" 2>/dev/null || true
    fi

    echo "[parse] Parsing tegrastats → $OUT/tegra_${SAFE_PROF}_parsed.json"
    if [ -f "$TEGRA_LOG" ] && [ -s "$TEGRA_LOG" ]; then
        python3 "$APP_DIR/scripts/parse_tegrastats.py" "$TEGRA_LOG" \
            > "$OUT/tegra_${SAFE_PROF}_parsed.json" 2>/dev/null || \
            echo '{"error":"parse failed"}' > "$OUT/tegra_${SAFE_PROF}_parsed.json"
    else
        echo '{"error":"tegrastats log vuoto o assente"}' > "$OUT/tegra_${SAFE_PROF}_parsed.json"
    fi

    echo "[OK] Profilo $PROF completato."
    echo ""
}

# ── Esegui i due profili ─────────────────────────────────────
run_profile "standard"
run_profile "optimized-tx2"

# ── Report finale ────────────────────────────────────────────
REPORT="$OUT/performance_report.md"
echo "============================================================"
echo " FASE 5 — Generazione performance_report.md"
echo "============================================================"

STD_JSON="$OUT/perf_standard.json"
OPT_JSON="$OUT/perf_optimized_tx2.json"
TS_JSON="$OUT/tegra_standard_parsed.json"
TO_JSON="$OUT/tegra_optimized_tx2_parsed.json"

python3 "$APP_DIR/scripts/generate_report.py" \
    --standard        "$STD_JSON" \
    --optimized       "$OPT_JSON" \
    --tegra-standard  "$TS_JSON" \
    --tegra-optimized "$TO_JSON" \
    --output          "$REPORT" \
    2>&1 | tee -a "$OUT/generate_report.log"

echo ""
echo "============================================================"
echo " BENCHMARK COMPLETATO"
echo " Output directory: $OUT"
echo " Report:           $REPORT"
echo "============================================================"
echo ""
echo "File prodotti:"
ls -lh "$OUT/"
