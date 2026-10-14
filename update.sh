#!/usr/bin/env bash
# update.sh — aggiorna MIC RES Sicilia dal repository e riavvia il servizio
#
# Uso:
#   bash update.sh                   aggiorna e riavvia
#   bash update.sh --force           riavvia senza aggiornare (anche se già aggiornato)
#   bash update.sh --branch main     fai checkout e pull del branch specificato
#   bash update.sh --check           mostra stato senza aggiornare
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$SCRIPT_DIR/update.log"
SERVICE="mic-res-sicilia"
VENV="$SCRIPT_DIR/.venv"

# ── argomenti ──────────────────────────────────────────────────────────────
FORCE=0; CHECK=0; TARGET_BRANCH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)   FORCE=1; shift ;;
    --check)   CHECK=1; shift ;;
    --branch)  TARGET_BRANCH="${2:-}"; shift 2 ;;
    *)         echo "Argomento sconosciuto: $1" >&2; exit 1 ;;
  esac
done

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
info() { log "ℹ  $*"; }
ok()   { log "✓  $*"; }
warn() { log "⚠  $*"; }
err()  { log "✗  $*"; }

# ── info versione ─────────────────────────────────────────────────────────
version_string() {
  local v sha branch
  v=$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null || echo "dev")
  sha=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo "?")
  branch=$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
  echo "$v (${branch}@${sha})"
}

log "══════════════════════════════════════════════"
log " MIC RES Sicilia — Aggiornamento"
log "══════════════════════════════════════════════"

cd "$SCRIPT_DIR"

# ── mostra stato corrente ──────────────────────────────────────────────────
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
LOCAL_SHA=$(git rev-parse HEAD 2>/dev/null || echo "?")
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "?")

info "Repository:    $REMOTE_URL"
info "Branch locale: $CURRENT_BRANCH"
info "Commit locale: ${LOCAL_SHA:0:8}"
info "Versione:      $(version_string)"

# ── se --check, mostra solo lo stato e esci ────────────────────────────────
if [[ $CHECK -eq 1 ]]; then
  git fetch origin 2>&1 | sed 's/^/   fetch: /'
  REMOTE_SHA=$(git rev-parse "origin/$CURRENT_BRANCH" 2>/dev/null || echo "")
  if [[ -n "$REMOTE_SHA" ]]; then
    AHEAD=$(git rev-list "origin/$CURRENT_BRANCH..HEAD" --count 2>/dev/null || echo "?")
    BEHIND=$(git rev-list "HEAD..origin/$CURRENT_BRANCH" --count 2>/dev/null || echo "?")
    info "Commit da scaricare: $BEHIND | Commit locali non pushati: $AHEAD"
    if [[ "$BEHIND" -gt 0 ]]; then
      info "Commit in arrivo:"
      git log --oneline "HEAD..origin/$CURRENT_BRANCH" | head -10 | while IFS= read -r l; do info "  · $l"; done
    fi
  fi
  exit 0
fi

# ── cambia branch se --branch specificato ─────────────────────────────────
if [[ -n "$TARGET_BRANCH" && "$TARGET_BRANCH" != "$CURRENT_BRANCH" ]]; then
  info "Switch branch: $CURRENT_BRANCH → $TARGET_BRANCH"
  if ! git show-ref --quiet --verify "refs/heads/$TARGET_BRANCH"; then
    git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH" 2>&1 | while IFS= read -r l; do log "  git: $l"; done
  else
    git checkout "$TARGET_BRANCH" 2>&1 | while IFS= read -r l; do log "  git: $l"; done
  fi
  CURRENT_BRANCH="$TARGET_BRANCH"
fi

# ── fetch ──────────────────────────────────────────────────────────────────
info "Fetch da origin..."
git fetch origin 2>&1 | grep -v '^$' | while IFS= read -r l; do log "  fetch: $l"; done

# ── confronto locale vs remoto DOPO il fetch ──────────────────────────────
REMOTE_SHA=$(git rev-parse "origin/$CURRENT_BRANCH" 2>/dev/null || echo "")
if [[ -z "$REMOTE_SHA" ]]; then
  warn "Branch 'origin/$CURRENT_BRANCH' non trovato sul remote."
  warn "Branch disponibili sul remote:"
  git branch -r | while IFS= read -r b; do warn "  $b"; done
  if [[ $FORCE -eq 0 ]]; then
    exit 1
  fi
fi

LOCAL_SHA=$(git rev-parse HEAD)
if [[ "$LOCAL_SHA" = "$REMOTE_SHA" ]]; then
  ok "Branch '$CURRENT_BRANCH' è già aggiornato all'ultimo commit remoto (${LOCAL_SHA:0:8})."
  info "Nessun aggiornamento disponibile sul branch '$CURRENT_BRANCH'."
  if [[ $FORCE -eq 0 ]]; then
    log "Usa --force per riavviare il servizio senza aggiornamenti."
    log "Usa --branch <nome> per passare a un altro branch."
    log "══════════════════════════════════════════════"
    exit 0
  fi
  info "Flag --force: procedo con il riavvio."
else
  # Mostra quanti e quali commit arriveranno
  BEHIND=$(git rev-list "HEAD..origin/$CURRENT_BRANCH" --count 2>/dev/null || echo "?")
  info "$BEHIND commit nuovi da applicare:"
  git log --oneline "HEAD..origin/$CURRENT_BRANCH" | head -20 | while IFS= read -r l; do info "  · $l"; done

  # Pull con rebase
  info "Applico aggiornamenti (git pull --rebase)..."
  git pull --rebase origin "$CURRENT_BRANCH" 2>&1 | while IFS= read -r l; do log "  pull: $l"; done

  NEW_SHA=$(git rev-parse HEAD)
  ok "Aggiornato: ${LOCAL_SHA:0:8} → ${NEW_SHA:0:8}"
  ok "Nuova versione: $(version_string)"
fi

# ── aggiorna dipendenze Python ────────────────────────────────────────────
PIP_CMD=""
if [[ -f "$VENV/bin/pip" ]]; then
  PIP_CMD="$VENV/bin/pip"
elif command -v pip3 &>/dev/null; then
  PIP_CMD="pip3"
elif command -v pip &>/dev/null; then
  PIP_CMD="pip"
fi

if [[ -n "$PIP_CMD" ]]; then
  info "Aggiornamento dipendenze Python ($PIP_CMD)..."
  "$PIP_CMD" install -q --upgrade -e "$SCRIPT_DIR" 2>&1 | grep -v '^$' | tail -5 | \
    while IFS= read -r l; do log "  pip: $l"; done
  ok "Dipendenze aggiornate."
else
  warn "pip non trovato — dipendenze non aggiornate."
fi

# ── riavvio servizio ──────────────────────────────────────────────────────
if command -v systemctl &>/dev/null && systemctl is-enabled "$SERVICE" &>/dev/null 2>&1; then
  info "Riavvio servizio systemd '$SERVICE'..."
  sudo systemctl restart "$SERVICE"
  sleep 2
  STATUS=$(systemctl is-active "$SERVICE" 2>/dev/null || echo "unknown")
  if [[ "$STATUS" = "active" ]]; then
    ok "Servizio attivo ($STATUS)."
  else
    warn "Servizio in stato: $STATUS"
  fi
elif [[ "$(uname)" = "Darwin" ]]; then
  PLIST="$HOME/Library/LaunchAgents/com.mic-res-sicilia.plist"
  if [[ -f "$PLIST" ]]; then
    info "Riavvio servizio launchd..."
    launchctl unload "$PLIST" 2>/dev/null || true
    sleep 1
    launchctl load -w "$PLIST"
    ok "Servizio launchd riavviato."
  else
    warn "Nessun servizio launchd. Riavvia con: bash start.sh"
  fi
else
  warn "Nessun servizio registrato. Riavvia con: bash start.sh"
fi

log "══════════════════════════════════════════════"
ok "Aggiornamento completato — $(version_string)"
log "Log completo: $LOG"
log "══════════════════════════════════════════════"
