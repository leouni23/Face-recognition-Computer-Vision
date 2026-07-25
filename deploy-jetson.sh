#!/usr/bin/env bash
# One-command rebuild + run on the Jetson TX2. No flags to remember.
#   ./deploy-jetson.sh            # build + (re)start, detached
#   EXT_DISK=/mnt/ssd ./deploy-jetson.sh
#   ./deploy-jetson.sh logs       # follow container logs
#   ./deploy-jetson.sh down       # stop the container
#
# Persists BIOMETRIC_SECRET_KEY + EXT_DISK to .env.jetson (gitignored) on first run, applies max
# performance (nvpmodel/jetson_clocks, best-effort), then `docker compose up --build`. The compose
# already mounts the Docker socket + restart:unless-stopped, so the UI Riavvia/Spegni work too.
set -euo pipefail
cd "$(dirname "$0")"

COMPOSE="docker compose -f docker-compose.jetson.yml --env-file .env.jetson"
ENV_FILE=".env.jetson"

# EXT_DISK resolution order: explicit env > value persisted in .env.jetson > auto-detect the first
# REAL external mount. No hardcoded default: a default path that isn't actually a mount silently
# sends models/DB/sessions to the internal eMMC (happened once — data split across two trees).
if [ -z "${EXT_DISK:-}" ] && [ -f "$ENV_FILE" ]; then
  EXT_DISK="$(sed -n 's/^EXT_DISK=//p' "$ENV_FILE" | tail -1)"
fi
if [ -z "${EXT_DISK:-}" ]; then
  for cand in /mnt/* /media/*/*; do
    [ -d "$cand" ] || continue
    if [ "$(stat -c %d "$cand" 2>/dev/null)" != "$(stat -c %d "$(dirname "$cand")" 2>/dev/null)" ]; then
      EXT_DISK="$cand"; break
    fi
  done
fi

# Sub-commands
case "${1:-up}" in
  logs) exec sudo -E $COMPOSE logs -f ;;
  down) exec sudo -E $COMPOSE down ;;
  up|"") ;;
  *) echo "uso: $0 [up|logs|down]"; exit 2 ;;
esac

# First-run secrets/config (never overwrite an existing key).
touch "$ENV_FILE"
grep -q '^BIOMETRIC_SECRET_KEY=' "$ENV_FILE" || \
  echo "BIOMETRIC_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')" >> "$ENV_FILE"
# HARD GATE: EXT_DISK must be a REAL mount point, otherwise the bootstrap below would happily
# create the data tree on the internal eMMC and the app would run on internal storage while
# looking perfectly healthy (models re-downloaded, a second DB, sessions written to the wrong
# disk). A directory on / is NOT an external disk: a mount point has a different device id than
# its parent. Override deliberately with ALLOW_INTERNAL_STORAGE=1.
_is_mount() {
  [ -d "$1" ] && [ "$(stat -c %d "$1" 2>/dev/null)" != "$(stat -c %d "$(dirname "$1")" 2>/dev/null)" ]
}
if [ -z "${EXT_DISK:-}" ] || ! _is_mount "$EXT_DISK"; then
  if [ "${ALLOW_INTERNAL_STORAGE:-0}" = "1" ]; then
    echo "ATTENZIONE: ${EXT_DISK:-(vuoto)} NON e' un disco montato — proseguo su storage interno (ALLOW_INTERNAL_STORAGE=1)." >&2
    : "${EXT_DISK:?EXT_DISK obbligatorio anche con ALLOW_INTERNAL_STORAGE=1}"
  else
    echo "ERRORE: EXT_DISK='${EXT_DISK:-(non impostato)}' non e' un disco esterno montato." >&2
    echo "        Dati/modelli/sessioni finirebbero sulla eMMC interna. Dischi montati disponibili:" >&2
    df -h --output=target,size,avail,fstype 2>/dev/null | awk 'NR==1 || $1 ~ /^\/(mnt|media)\//' >&2
    echo "        Riprova indicando il disco giusto, es.:  EXT_DISK=/mnt/<disco> $0" >&2
    echo "        (per forzare comunque lo storage interno: ALLOW_INTERNAL_STORAGE=1 $0)" >&2
    exit 1
  fi
fi
echo "Disco dati: ${EXT_DISK}  ($(df -h --output=avail "$EXT_DISK" 2>/dev/null | tail -1 | tr -d ' ') liberi)"

# Persist the (now validated) disk so the next run needs no flags.
grep -q '^EXT_DISK=' "$ENV_FILE" && sed -i "s#^EXT_DISK=.*#EXT_DISK=${EXT_DISK}#" "$ENV_FILE" \
  || echo "EXT_DISK=${EXT_DISK}" >> "$ENV_FILE"

# Bootstrap: le sottocartelle DEVONO esistere prima del primo avvio, altrimenti il symlink
# /root/.insightface -> /data/models punta nel vuoto e il warm-up modelli crasha.
mkdir -p "$EXT_DISK/models" "$EXT_DISK/engines" "$EXT_DISK/validation" 2>/dev/null \
  || sudo mkdir -p "$EXT_DISK/models" "$EXT_DISK/engines" "$EXT_DISK/validation"
[ -w "$EXT_DISK/models" ] || sudo chmod -R a+rwX "$EXT_DISK/models" "$EXT_DISK/engines" "$EXT_DISK/validation" || true

# USB camera: the host device node is NOT fixed (video0/video1/... depending on enumeration), and
# a missing node makes `compose up` fail outright. Auto-detect the first V4L2 capture node that is
# NOT the CSI one (/dev/video0 on Tegra needs nvargus-daemon, unavailable in the container); fall
# back to /dev/null so the container always starts (RTSP sources need no device).
if [ -z "${CAMERA_DEVICE:-}" ]; then
  CAMERA_DEVICE=/dev/null
  for d in /dev/video1 /dev/video2 /dev/video3 /dev/video0; do
    if [ -e "$d" ]; then CAMERA_DEVICE="$d"; break; fi
  done
fi
export CAMERA_DEVICE
# Persist it in the env-file too: `sudo -E` doesn't propagate the variable under every sudoers
# config, and compose reads the env-file regardless.
grep -q '^CAMERA_DEVICE=' "$ENV_FILE" && sed -i "s#^CAMERA_DEVICE=.*#CAMERA_DEVICE=${CAMERA_DEVICE}#" "$ENV_FILE" \
  || echo "CAMERA_DEVICE=${CAMERA_DEVICE}" >> "$ENV_FILE"
echo "Camera USB: ${CAMERA_DEVICE}$([ "$CAMERA_DEVICE" = /dev/null ] && echo '  (nessuna rilevata: usa sorgenti RTSP/IP)')"

# Max performance (best-effort; ignore if not a Jetson / no sudo).
sudo nvpmodel -m 0 >/dev/null 2>&1 || true
sudo jetson_clocks  >/dev/null 2>&1 || true

echo "Build + avvio (EXT_DISK=${EXT_DISK})..."
sudo -E $COMPOSE up --build -d
echo "Fatto. UI: http://localhost:8000  |  log: ./deploy-jetson.sh logs"
