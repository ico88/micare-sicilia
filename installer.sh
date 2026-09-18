#!/usr/bin/env bash
# MIC RES Sicilia — installer e avvio automatico
# Uso: bash installer.sh
# Porta di ascolto: 5555

set -euo pipefail

APP_PORT=5555
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/.venv"
SERVICE_NAME="micare-sicilia"

# ── Colori ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERR]${NC}   $*" >&2; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════╗"
echo -e "║      MIC RES Sicilia — Installer v1.0        ║"
echo -e "╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. Sistema operativo e dipendenze di sistema ──────────────────────────────
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
elif [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS="$ID"
else
    OS="unknown"
fi

info "Sistema rilevato: $OS"

install_system_deps() {
    info "Installo dipendenze di sistema..."
    if [[ "$OS" == "ubuntu" || "$OS" == "debian" ]]; then
        sudo apt-get update -q
        sudo apt-get install -y -q python3 python3-pip python3-venv python3-dev \
            build-essential libssl-dev libffi-dev curl git \
            libpq-dev gcc g++ cmake pkg-config
    elif [[ "$OS" == "fedora" || "$OS" == "rhel" || "$OS" == "centos" || "$OS" == "rocky" ]]; then
        sudo dnf install -y python3 python3-pip python3-devel \
            gcc gcc-c++ make openssl-devel libffi-devel cmake \
            curl git
    elif [[ "$OS" == "macos" ]]; then
        if ! command -v brew &>/dev/null; then
            warn "Homebrew non trovato. Installalo da https://brew.sh poi riesegui."
            exit 1
        fi
        brew install python@3.11 cmake pkg-config || true
    else
        warn "Distribuzione non riconosciuta. Assicurati di avere Python 3.10+, pip, venv e build tools."
    fi
}

# Controlla se python3 è disponibile e ha versione >= 3.10
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c 'import sys; print(sys.version_info >= (3,10))' 2>/dev/null || echo False)
        if [[ "$VER" == "True" ]]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    install_system_deps
    PYTHON_CMD="python3"
fi

ok "Python: $($PYTHON_CMD --version)"

# ── 2. Ambiente virtuale ──────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creo ambiente virtuale in .venv ..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    ok "Ambiente virtuale creato."
else
    info "Ambiente virtuale già presente, lo riuso."
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# ── 3. Upgrade pip ────────────────────────────────────────────────────────────
info "Aggiorno pip..."
"$PIP" install --upgrade pip --quiet

# ── 4. Dipendenze Python ──────────────────────────────────────────────────────
info "Installo dipendenze Python (può richiedere qualche minuto — Prophet compila C++)..."

# pystan è una dipendenza di prophet che a volte richiede build speciale
"$PIP" install --upgrade "setuptools" "wheel" "numpy<2" --quiet

# Installa il pacchetto in modalità editable (include src/micare_sicilia)
"$PIP" install -e "$APP_DIR" --quiet

# Aggiungi statsmodels (usato per baseline ARIMA in model.py)
"$PIP" install statsmodels --quiet

# Aggiungi gunicorn per produzione
"$PIP" install gunicorn --quiet

ok "Dipendenze installate."

# ── 5. Directory dati e modelli ───────────────────────────────────────────────
mkdir -p "$APP_DIR/data" "$APP_DIR/models" "$APP_DIR/instance"
ok "Directory data/, models/, instance/ pronte."

# ── 6. Configurazione Flask ───────────────────────────────────────────────────
CONFIG_FILE="$APP_DIR/instance/config.py"
if [[ ! -f "$CONFIG_FILE" ]]; then
    info "Genero chiave segreta e config in instance/config.py ..."
    SECRET=$($PYTHON_CMD -c "import secrets; print(secrets.token_hex(32))")
    cat > "$CONFIG_FILE" <<EOF
# Configurazione locale — NON committare questo file
SECRET_KEY = "$SECRET"
EOF
    ok "Config generata: $CONFIG_FILE"
else
    info "instance/config.py già presente, non sovrascritto."
fi

# ── 7. Script di avvio ────────────────────────────────────────────────────────
START_SCRIPT="$APP_DIR/start.sh"
cat > "$START_SCRIPT" <<'STARTEOF'
#!/usr/bin/env bash
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$APP_DIR/.venv/bin/activate"
source "$VENV"
export FLASK_APP=run.py
export FLASK_ENV=production
cd "$APP_DIR"
echo "Avvio MIC RES Sicilia su http://0.0.0.0:5555 ..."
exec gunicorn \
    --bind 0.0.0.0:5555 \
    --workers 2 \
    --threads 4 \
    --timeout 600 \
    --worker-class gthread \
    --log-level info \
    "run:app"
STARTEOF
chmod +x "$START_SCRIPT"
ok "Script di avvio creato: start.sh"

# ── 8. Systemd service (solo Linux) ──────────────────────────────────────────
if [[ "$OS" != "macos" ]] && command -v systemctl &>/dev/null; then
    UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
    CURRENT_USER="$(whoami)"

    if [[ ! -f "$UNIT_FILE" ]]; then
        info "Creo servizio systemd ${SERVICE_NAME}.service ..."
        sudo tee "$UNIT_FILE" > /dev/null <<EOF
[Unit]
Description=MIC RES Sicilia — AMR Prediction Platform
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/start.sh
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal
Environment=FLASK_ENV=production

[Install]
WantedBy=multi-user.target
EOF
        sudo systemctl daemon-reload
        sudo systemctl enable "$SERVICE_NAME" 2>/dev/null || true
        ok "Servizio systemd ${SERVICE_NAME} abilitato (avvio automatico al boot)."
    else
        info "Servizio systemd già presente."
    fi

    SYSTEMD_AVAILABLE=true
else
    SYSTEMD_AVAILABLE=false
fi

# ── 9. Firewall (solo Linux, opzionale) ───────────────────────────────────────
if command -v ufw &>/dev/null; then
    if sudo ufw status | grep -q "Status: active"; then
        info "UFW attivo — aggiungo regola per porta $APP_PORT ..."
        sudo ufw allow "$APP_PORT/tcp" --quiet || true
        ok "Porta $APP_PORT aperta in UFW."
    fi
fi

# ── 10. Riepilogo ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗"
echo -e "║          Installazione completata!           ║"
echo -e "╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Avvio manuale:      ${CYAN}bash start.sh${NC}"
if [[ "$SYSTEMD_AVAILABLE" == "true" ]]; then
    echo -e "  Avvio come servizio: ${CYAN}sudo systemctl start ${SERVICE_NAME}${NC}"
    echo -e "  Log in tempo reale:  ${CYAN}sudo journalctl -fu ${SERVICE_NAME}${NC}"
    echo -e "  Stop:                ${CYAN}sudo systemctl stop ${SERVICE_NAME}${NC}"
fi
echo ""
echo -e "  URL locale:    ${CYAN}http://localhost:${APP_PORT}${NC}"
echo -e "  URL rete:      ${CYAN}http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'IP_SERVER'):${APP_PORT}${NC}"
echo ""
echo -e "  ${YELLOW}Nota:${NC} al primo avvio carica i file Excel da 'Gestione dati',"
echo -e "        poi avvia l'addestramento (Prophet richiede ore la prima volta)."
echo ""

# ── 11. Avvio immediato (opzionale) ──────────────────────────────────────────
read -rp "Avviare l'applicazione adesso? [S/n] " LAUNCH
LAUNCH="${LAUNCH:-S}"
if [[ "$LAUNCH" =~ ^[Ss]$ ]]; then
    echo ""
    if [[ "$SYSTEMD_AVAILABLE" == "true" ]]; then
        sudo systemctl start "$SERVICE_NAME"
        ok "Servizio avviato. Aprire http://localhost:${APP_PORT}"
    else
        info "Avvio in foreground (Ctrl+C per fermare)..."
        exec bash "$START_SCRIPT"
    fi
fi
