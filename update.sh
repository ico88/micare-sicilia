#!/usr/bin/env bash
# update.sh — aggiorna MIC RES Sicilia dal repository e riavvia il servizio
#
# Uso:
#   bash update.sh                   aggiorna il codice e riavvia
#   bash update.sh --force           riavvia anche senza nuovi commit
#   bash update.sh --branch main     switch a quel branch prima del pull
#   bash update.sh --check           mostra lo stato senza modificare nulla

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/update.log"
SERVICE="mic-res-sicilia"
VENV="$SCRIPT_DIR/.venv"

# ── argomenti ──────────────────────────────────────────────────────────────
FORCE=0; CHECK=0; TARGET_BRANCH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)   FORCE=1;               shift ;;
    --check)   CHECK=1;               shift ;;
    --branch)  TARGET_BRANCH="${2:?}"; shift 2 ;;
    *) echo "Argomento sconosciuto: $1" >&2; exit 1 ;;
  esac
done

# ── logging ────────────────────────────────────────────────────────────────
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG"
}

version_string() {
  local v sha branch
  v=$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null || echo "dev")
  sha=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
  branch=$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
  echo "$v (${branch}@${sha})"
}

cd "$SCRIPT_DIR"

log "══════════════════════════════════════════"
log " MIC RES Sicilia — Aggiornamento"
log "══════════════════════════════════════════"
log "Repository : $(git remote get-url origin 2>/dev/null || echo '?')"
log "Branch     : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
log "Commit     : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
log "Versione   : $(version_string)"
log "──────────────────────────────────────────"

# ── switch branch se richiesto ─────────────────────────────────────────────
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "HEAD")
if [[ -n "$TARGET_BRANCH" && "$TARGET_BRANCH" != "$CURRENT_BRANCH" ]]; then
  log "Switch branch: $CURRENT_BRANCH → $TARGET_BRANCH"
  if git show-ref --quiet --verify "refs/heads/$TARGET_BRANCH"; then
    git checkout "$TARGET_BRANCH"
  else
    git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH"
  fi
  CURRENT_BRANCH="$TARGET_BRANCH"
  log "Branch attuale: $CURRENT_BRANCH"
fi

# ── fetch ──────────────────────────────────────────────────────────────────
log "Fetch dal remote..."
if ! git fetch origin; then
  log "ERRORE: git fetch fallito — verificare connessione e credenziali."
  exit 1
fi
log "Fetch completato."

# ── --check: mostra stato e termina ───────────────────────────────────────
if [[ $CHECK -eq 1 ]]; then
  BEHIND=$(git rev-list "HEAD..origin/$CURRENT_BRANCH" --count 2>/dev/null || echo "?")
  AHEAD=$(git rev-list "origin/$CURRENT_BRANCH..HEAD" --count 2>/dev/null || echo "?")
  log "Commit da scaricare: $BEHIND | Commit locali non pushati: $AHEAD"
  if [[ "$BEHIND" =~ ^[0-9]+$ && "$BEHIND" -gt 0 ]]; then
    log "Commit in arrivo:"
    git log --oneline "HEAD..origin/$CURRENT_BRANCH" | while IFS= read -r l; do
      log "  · $l"
    done
  fi
  exit 0
fi

# ── confronto locale vs remoto ─────────────────────────────────────────────
LOCAL_SHA=$(git rev-parse HEAD)
REMOTE_SHA=$(git rev-parse "origin/$CURRENT_BRANCH" 2>/dev/null || echo "")

if [[ -z "$REMOTE_SHA" ]]; then
  log "ATTENZIONE: branch 'origin/$CURRENT_BRANCH' non trovato."
  log "Branch disponibili sul remote:"
  git branch -r | while IFS= read -r b; do log "  $b"; done
  if [[ $FORCE -eq 0 ]]; then
    log "Usa --branch <nome> per specificare il branch corretto."
    exit 1
  fi
fi

UPDATED=0
if [[ "$LOCAL_SHA" = "$REMOTE_SHA" ]]; then
  log "Già all'ultima versione (${LOCAL_SHA:0:8})."
  if [[ $FORCE -eq 0 ]]; then
    log "Usa --force per riavviare il servizio senza nuovi commit."
    log "══════════════════════════════════════════"
    exit 0
  fi
  log "Flag --force attivo: procedo con riavvio."
else
  BEHIND=$(git rev-list "HEAD..origin/$CURRENT_BRANCH" --count 2>/dev/null || echo "?")
  log "$BEHIND commit nuovi in arrivo:"
  git log --oneline "HEAD..origin/$CURRENT_BRANCH" | while IFS= read -r l; do
    log "  · $l"
  done

  log "Applico aggiornamenti (git pull --rebase)..."
  if ! git pull --rebase origin "$CURRENT_BRANCH"; then
    log "ERRORE: git pull fallito — risolvere i conflitti manualmente."
    exit 1
  fi

  NEW_SHA=$(git rev-parse HEAD)
  log "Aggiornato: ${LOCAL_SHA:0:8} → ${NEW_SHA:0:8}"
  log "Nuova versione: $(version_string)"
  UPDATED=1
fi

# ── dipendenze Python ──────────────────────────────────────────────────────
log "──────────────────────────────────────────"
log "Aggiornamento dipendenze Python..."

PIP_CMD=""
if [[ -f "$VENV/bin/pip" ]]; then
  PIP_CMD="$VENV/bin/pip"
elif command -v pip3 &>/dev/null; then
  PIP_CMD="pip3"
elif command -v pip &>/dev/null; then
  PIP_CMD="pip"
fi

if [[ -n "$PIP_CMD" ]]; then
  "$PIP_CMD" install -q --upgrade -e "$SCRIPT_DIR"
  log "Dipendenze aggiornate (pip: $PIP_CMD)."
else
  log "ATTENZIONE: pip non trovato — dipendenze non aggiornate."
fi

# ── riavvio servizio ───────────────────────────────────────────────────────
log "──────────────────────────────────────────"
log "Riavvio servizio..."

if command -v systemctl &>/dev/null && systemctl is-enabled "$SERVICE" &>/dev/null 2>&1; then
  sudo systemctl restart "$SERVICE"
  sleep 2
  STATUS=$(systemctl is-active "$SERVICE" 2>/dev/null || echo "unknown")
  log "Servizio systemd '$SERVICE': $STATUS"

elif [[ "$(uname)" = "Darwin" ]]; then
  PLIST="$HOME/Library/LaunchAgents/com.mic-res-sicilia.plist"
  if [[ -f "$PLIST" ]]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    sleep 1
    launchctl load -w "$PLIST"
    log "Servizio launchd riavviato."
  else
    log "ATTENZIONE: nessun servizio launchd configurato."
    log "  → Riavvia con: bash start.sh"
  fi

else
  log "ATTENZIONE: nessun servizio systemd '$SERVICE' trovato."
  log "  → Riavvia con: bash start.sh"
fi

log "══════════════════════════════════════════"
log "Completato — $(version_string)"
log "Log: $LOG"
