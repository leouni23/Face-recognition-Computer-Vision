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
EXT_DISK="${EXT_DISK:-/mnt/faceid}"

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
grep -q '^EXT_DISK=' "$ENV_FILE" && sed -i "s#^EXT_DISK=.*#EXT_DISK=${EXT_DISK}#" "$ENV_FILE" \
  || echo "EXT_DISK=${EXT_DISK}" >> "$ENV_FILE"

if [ ! -d "$EXT_DISK" ]; then
  echo "ATTENZIONE: disco esterno $EXT_DISK non montato (modelli/engine/dati vanno li')." >&2
fi

# Bootstrap: le sottocartelle DEVONO esistere prima del primo avvio, altrimenti il symlink
# /root/.insightface -> /data/models punta nel vuoto e il warm-up modelli crasha.
mkdir -p "$EXT_DISK/models" "$EXT_DISK/engines" "$EXT_DISK/validation" 2>/dev/null \
  || sudo mkdir -p "$EXT_DISK/models" "$EXT_DISK/engines" "$EXT_DISK/validation"
[ -w "$EXT_DISK/models" ] || sudo chmod -R a+rwX "$EXT_DISK/models" "$EXT_DISK/engines" "$EXT_DISK/validation" || true

# Max performance (best-effort; ignore if not a Jetson / no sudo).
sudo nvpmodel -m 0 >/dev/null 2>&1 || true
sudo jetson_clocks  >/dev/null 2>&1 || true

echo "Build + avvio (EXT_DISK=${EXT_DISK})..."
sudo -E $COMPOSE up --build -d
echo "Fatto. UI: http://localhost:8000  |  log: ./deploy-jetson.sh logs"
