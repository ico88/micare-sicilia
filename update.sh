#!/usr/bin/env bash
# update.sh — aggiorna MIC RES Sicilia dal repository e riavvia il servizio
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/update.log"
SERVICE="mic-res-sicilia"
VENV="$SCRIPT_DIR/.venv"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Avvio aggiornamento MIC RES Sicilia ==="

# 1. Pull dal repository
log "Aggiornamento codice sorgente..."
cd "$SCRIPT_DIR"
git fetch origin
BEFORE=$(git rev-parse HEAD)
git pull --rebase origin "$(git rev-parse --abbrev-ref HEAD)"
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
  log "Nessun aggiornamento disponibile (già all'ultima versione)."
  log "Uso --force per forzare il riavvio del servizio comunque."
  FORCE="${1:-}"
  if [ "$FORCE" != "--force" ]; then
    log "Operazione terminata senza modifiche."
    exit 0
  fi
fi

log "Aggiornato: $BEFORE → $AFTER"

# 2. Log delle modifiche
CHANGES=$(git log --oneline "$BEFORE".."$AFTER" 2>/dev/null || echo "(nessun log disponibile)")
log "Modifiche:"
echo "$CHANGES" | while IFS= read -r line; do log "  $line"; done

# 3. Aggiorna dipendenze Python
log "Aggiornamento dipendenze Python..."
"$VENV/bin/pip" install -q --upgrade -e "$SCRIPT_DIR" 2>&1 | tail -5 | while IFS= read -r line; do log "  pip: $line"; done

# 4. Riavvio servizio
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
    log "AVVISO: nessun servizio launchd trovato. Riavvia manualmente con: bash start.sh"
  fi
else
  log "AVVISO: servizio systemd non trovato. Riavvia manualmente con: bash start.sh"
fi

log "=== Aggiornamento completato ==="
log "Log completo: $LOG"
