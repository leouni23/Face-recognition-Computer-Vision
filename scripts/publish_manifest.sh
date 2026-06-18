#!/usr/bin/env bash
#
# Crea un image index multi-arch (manifest list / OCI image index) su Docker Hub
# combinando i tag per-architettura GIÀ pubblicati:
#
#   t018/faceid:latest   →   amd64 (x86 CUDA)  +  arm64 (Jetson Orin)
#
# Così `docker pull t018/faceid` scarica in automatico l'immagine giusta per
# l'architettura dell'host (amd64 sul PC, arm64 sul Jetson Orin).
#
# ── Prerequisiti ───────────────────────────────────────────────────────────────
#   1. `docker login` già eseguito.
#   2. I tag per-arch già pushati su Hub:
#        • x86  (linux/amd64):  t018/faceid:x86-cuda      → già pubblicato dal PC
#        • Orin (linux/arm64):  t018/faceid:jetson-orin   → buildare/pushare SUL device:
#            sudo docker build -f Dockerfile.jetson -t t018/faceid:jetson-orin \
#              --build-arg BASE_IMAGE=dustynv/onnxruntime:r36.2.0 .
#            sudo docker push t018/faceid:jetson-orin
#
# ── Nota sul Jetson TX2 ──────────────────────────────────────────────────────────
#   Il TX2 è anch'esso `linux/arm64`: un image index ha UN SOLO slot per
#   os/arch/variant, e lo spec OCI non distingue L4T r32 (TX2) da r36 (Orin).
#   Quindi il TX2 NON può stare nello stesso index dell'Orin → resta un tag
#   esplicito: t018/faceid:jetson-tx2  (gli utenti TX2 fanno il pull di quel tag).
#
# Uso:  ./scripts/publish_manifest.sh
#       REPO=tuo/repo ARM64_TAG=jetson-orin INDEX_TAG=latest ./scripts/publish_manifest.sh
set -euo pipefail

REPO="${REPO:-t018/faceid}"
INDEX_TAG="${INDEX_TAG:-latest}"
AMD64_TAG="${AMD64_TAG:-x86-cuda}"
ARM64_TAG="${ARM64_TAG:-jetson-orin}"

echo "Image index ${REPO}:${INDEX_TAG}  =  ${AMD64_TAG} (amd64)  +  ${ARM64_TAG} (arm64)"

# I tag per-arch devono già esistere sul registry
for t in "$AMD64_TAG" "$ARM64_TAG"; do
  if ! docker manifest inspect "${REPO}:${t}" >/dev/null 2>&1; then
    echo "ERRORE: ${REPO}:${t} non trovato su Docker Hub — buildalo e pushalo prima." >&2
    exit 1
  fi
done

docker buildx imagetools create -t "${REPO}:${INDEX_TAG}" \
  "${REPO}:${AMD64_TAG}" \
  "${REPO}:${ARM64_TAG}"

echo
echo "Fatto. Contenuto dell'index:"
docker buildx imagetools inspect "${REPO}:${INDEX_TAG}"
