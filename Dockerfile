# syntax=docker/dockerfile:1
#
# Face ID — single parameterised Dockerfile for three GPU targets.
# The application layer is identical everywhere; only the base image and the
# ONNX Runtime build differ (CUDA userspace is provided by the base image).
#
#   x86-64 + CUDA 12 (RTX 2080):
#     docker build -t faceid:x86-cuda .
#       (defaults: BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04, ORT=onnxruntime-gpu)
#
#   Jetson Orin Nano (L4T r36, CUDA 12) — BUILD ON THE DEVICE:
#     docker build -t faceid:orin \
#       --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-base:r36.2.0 \
#       --build-arg ONNXRUNTIME_PIP="" --build-arg INSTALL_OPENCV=0 .
#
#   Jetson TX2 (L4T r32.7, CUDA 10.2) — BUILD ON THE DEVICE:
#     docker build -t faceid:tx2 \
#       --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.7.1 \
#       --build-arg ONNXRUNTIME_PIP="" --build-arg INSTALL_OPENCV=0 .
#
# On Jetson, ONNX Runtime and OpenCV come from the L4T / jetson-containers base
# (the pip wheels are not built for L4T) — hence ONNXRUNTIME_PIP="" INSTALL_OPENCV=0.
# Cross-building ARM images on x86 with buildx/QEMU can produce the image but
# CANNOT run or validate GPU code: build and test the Jetson images on the device.

ARG BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG PYTHON=python3
ARG ONNXRUNTIME_PIP=onnxruntime-gpu
ARG INSTALL_OPENCV=1

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs: OpenCV runtime (libGL/glib), FFMPEG for RTSP, python, curl for healthcheck.
# build-essential + python3-dev: InsightFace compiles a small Cython extension on install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ${PYTHON} ${PYTHON}-pip ${PYTHON}-dev build-essential \
        ffmpeg libgl1 libglib2.0-0 curl tzdata \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps. ONNX Runtime and (optionally) OpenCV are the platform-specific
# bits: strip them from requirements and install ORT separately so the same file
# serves every target. On Jetson the base image already provides both.
COPY requirements.txt .
RUN ${PYTHON} -m pip install --upgrade pip && \
    grep -vE '^onnxruntime' requirements.txt > /tmp/req.txt && \
    if [ "${INSTALL_OPENCV}" = "0" ]; then grep -v '^opencv-python' /tmp/req.txt > /tmp/req2.txt && mv /tmp/req2.txt /tmp/req.txt; fi && \
    ${PYTHON} -m pip install -r /tmp/req.txt && \
    if [ -n "${ONNXRUNTIME_PIP}" ]; then ${PYTHON} -m pip install "${ONNXRUNTIME_PIP}"; fi

# App code
COPY . .

# Non-root user; /data and the InsightFace model cache are writable volumes
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

# Web UI bound to all interfaces (container). Set WEB_PASSWORD when exposing beyond the host.
CMD ["python3", "main.py", "--web", "--host", "0.0.0.0"]
