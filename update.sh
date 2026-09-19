#!/usr/bin/env bash
# update.sh — aggiorna MIC RES Sicilia dal repository e riavvia il servizio
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/update.log"
SERVICE="mic-res-sicilia"
VENV="$SCRIPT_DIR/.venv"
FORCE="${1:-}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

version_string() {
  local v
  v=$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null || echo "dev")
  local sha
  sha=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
  echo "$v ($sha)"
}

log "=== Avvio aggiornamento MIC RES Sicilia ==="
log "Versione attuale: $(version_string)"

# 1. Pull dal repository
cd "$SCRIPT_DIR"
git fetch origin
BEFORE=$(git rev-parse HEAD)
git pull --rebase origin "$(git rev-parse --abbrev-ref HEAD)"
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
  log "Già all'ultima versione — $(version_string)"
  if [ "$FORCE" != "--force" ]; then
    log "Usa --force per riavviare il servizio senza aggiornamenti."
    log "=== Nessuna modifica ==="
    exit 0
  fi
  log "Flag --force: procedo con il riavvio."
else
  log "Aggiornato: ${BEFORE:0:8} → ${AFTER:0:8}"
  log "Nuova versione: $(version_string)"
  CHANGES=$(git log --oneline "${BEFORE}..${AFTER}" 2>/dev/null || echo "(nessun log)")
  log "Modifiche incluse:"
  echo "$CHANGES" | while IFS= read -r line; do log "  · $line"; done
fi

# 2. Aggiorna dipendenze Python
log "Aggiornamento dipendenze Python..."
"$VENV/bin/pip" install -q --upgrade -e "$SCRIPT_DIR" 2>&1 | tail -3 | while IFS= read -r line; do log "  pip: $line"; done

# 3. Riavvio servizio
if command -v systemctl &>/dev/null && systemctl is-enabled "$SERVICE" &>/dev/null 2>&1; then
  log "Riavvio servizio systemd $SERVICE..."
  sudo systemctl restart "$SERVICE"
  sleep 2
  STATUS=$(systemctl is-active "$SERVICE" 2>/dev/null || echo "unknown")
  log "Stato servizio: $STATUS"
elif [ "$(uname)" = "Darwin" ]; then
  PLIST="$HOME/Library/LaunchAgents/com.mic-res-sicilia.plist"
  if [ -f "$PLIST" ]; then
    log "Riavvio servizio launchd..."
    launchctl unload "$PLIST" 2>/dev/null || true
    sleep 1
    launchctl load -w "$PLIST"
    log "Servizio launchd riavviato."
  else
    log "AVVISO: nessun servizio launchd. Riavvia con: bash start.sh"
  fi
else
  log "AVVISO: nessun servizio registrato. Riavvia con: bash start.sh"
fi

log "=== Aggiornamento completato — $(version_string) ==="
log "Log: $LOG"
