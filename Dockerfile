# syntax=docker/dockerfile:1
#
# Face ID — immagine GPU (multi-stage).
#
# Base NVIDIA `cudnn-runtime`: impacchetta CUDA 12 runtime + cuDNN 9 (tutte le librerie
# che il provider CUDA di onnxruntime carica: cudart, cublas, cufft, curand, cusolver,
# cusparse, cudnn) in modo più compatto dei wheel pip equivalenti. Lo stage `builder`
# compila le dipendenze (InsightFace ha un'estensione Cython) in un venv; lo stage finale
# copia solo il venv → niente build tools nell'immagine. Il driver GPU arriva dall'host
# con `--gpus all`.
#
#   x86-64 + CUDA 12:
#     docker build -t faceid:x86-cuda .
#
#   Jetson (ARM64 / L4T, TX2 e Orin): NON usare questo Dockerfile — vedi Dockerfile.jetson
#   (parte da una base jetson-containers con onnxruntime già compilato per L4T) e il README.

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# ─────────────── builder: dipendenze in un venv ───────────────
FROM ${BASE_IMAGE} AS builder

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 DEBIAN_FRONTEND=noninteractive
# Pin onnxruntime-gpu to the last CUDA-12 release: the BASE_IMAGE is a CUDA 12
# cudnn-runtime, but onnxruntime-gpu ≥1.23 links CUDA 13 (libcudart.so.13) — an
# unpinned install there loads no CUDA and silently runs on CPU. Bumping the base
# to a CUDA 13 image? then move this pin up in lockstep.
ARG ONNXRUNTIME_PIP=onnxruntime-gpu==1.22.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt .
# onnxruntime (CPU) rimosso dai requirements: installiamo la variante GPU a parte
RUN pip install --upgrade pip wheel && \
    grep -vE '^onnxruntime' requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt && \
    if [ -n "$ONNXRUNTIME_PIP" ]; then pip install "$ONNXRUNTIME_PIP"; fi && \
    find /opt/venv -name '*.pyc' -delete

# ─────────────── runtime: solo venv + librerie a runtime ───────────────
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
# python3 (per il venv) + OpenCV (libGL/glib) + FFMPEG (RTSP) + curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 ffmpeg libgl1 libglib2.0-0 curl tzdata \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app
COPY . .

RUN useradd -m -u 1000 faceid && \
    mkdir -p /data /home/faceid/.insightface && \
    chown -R faceid:faceid /app /data /home/faceid
USER faceid

ENV DATA_DIR=/data \
    DATABASE_URL=sqlite:////data/face_id.db \
    USE_GPU=true \
    LOG_LEVEL=INFO

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/cameras || exit 1

CMD ["python3", "main.py", "--web", "--host", "0.0.0.0"]
