#!/usr/bin/env bash
# MIC RES Sicilia — Installer v2.0
# Uso: bash installer.sh
# Porta di ascolto: 5555
#
# Supporta: Ubuntu 20.04/22.04/24.04, Debian 11/12, Fedora 38+,
#           RHEL/Rocky/AlmaLinux 8/9, macOS 12+
# Richiede: sudo (Linux) o Homebrew (macOS)

set -euo pipefail

APP_PORT=5555
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/.venv"
SERVICE_NAME="micare-sicilia"
LOG_FILE="$APP_DIR/install.log"

# ── Colori ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*" | tee -a "$LOG_FILE"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*" | tee -a "$LOG_FILE"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*" | tee -a "$LOG_FILE"; }
step()  { echo -e "\n${BOLD}▶ $*${NC}" | tee -a "$LOG_FILE"; }
die()   { echo -e "${RED}[ERRORE]${NC} $*" | tee -a "$LOG_FILE" >&2; echo "Log completo: $LOG_FILE"; exit 1; }

# Tutto l'output va anche nel log
exec > >(tee -a "$LOG_FILE") 2>&1

echo "" | tee -a "$LOG_FILE"
echo -e "${CYAN}╔══════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo -e "║     MIC RES Sicilia — Installer v2.0        ║" | tee -a "$LOG_FILE"
echo -e "╚══════════════════════════════════════════════╝${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
info "Log: $LOG_FILE"
info "Directory: $APP_DIR"

# ── 1. Rileva sistema operativo ───────────────────────────────────────────────
step "1/9 · Rilevamento sistema operativo"

if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    OS_NAME="macOS"
elif [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS="$ID"
    OS_NAME="${PRETTY_NAME:-$ID}"
else
    OS="unknown"
    OS_NAME="Sconosciuto"
fi

info "OS: $OS_NAME"

# ── 2. Dipendenze di sistema ──────────────────────────────────────────────────
step "2/9 · Dipendenze di sistema"
# Prophet richiede compilazione C/C++ (pystan) e cmake.
# pandas/numpy richiedono libc e libstdc++.
# gunicorn per produzione.

install_system_deps_deb() {
    info "apt-get update..."
    sudo apt-get update -y -q 2>&1 | tail -3
    info "Installo pacchetti di sistema..."
    sudo apt-get install -y -q \
        python3 python3-pip python3-venv python3-dev python3-full \
        build-essential gcc g++ \
        libssl-dev libffi-dev \
        cmake pkg-config \
        git curl wget \
        libopenblas-dev liblapack-dev \
        libbz2-dev zlib1g-dev \
        2>&1 | tail -5
}

install_system_deps_rpm() {
    info "Installo pacchetti di sistema (dnf)..."
    sudo dnf install -y \
        python3 python3-pip python3-devel \
        gcc gcc-c++ make cmake \
        openssl-devel libffi-devel \
        bzip2-devel zlib-devel \
        openblas-devel lapack-devel \
        git curl wget \
        2>&1 | tail -5
}

install_system_deps_macos() {
    if ! command -v brew &>/dev/null; then
        die "Homebrew non trovato. Installalo da https://brew.sh e riesegui."
    fi
    info "Installo dipendenze Homebrew..."
    brew install python@3.11 cmake pkg-config openblas 2>&1 | tail -5 || true
    brew link --overwrite python@3.11 2>/dev/null || true
}

case "$OS" in
    ubuntu|debian|linuxmint|pop)
        install_system_deps_deb ;;
    fedora)
        install_system_deps_rpm ;;
    rhel|centos|rocky|almalinux)
        # EPEL necessario per alcune dipendenze
        sudo dnf install -y epel-release 2>/dev/null || true
        install_system_deps_rpm ;;
    macos)
        install_system_deps_macos ;;
    *)
        warn "Distribuzione '$OS' non riconosciuta."
        warn "Assicurati di avere: python3 >= 3.10, gcc, g++, cmake, pip, venv"
        ;;
esac

ok "Dipendenze di sistema installate."

# ── 3. Python >= 3.10 ─────────────────────────────────────────────────────────
step "3/9 · Verifica Python"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" -c 'import sys; print(sys.version_info >= (3,10))' 2>/dev/null || echo "False")
        if [[ "$VER" == "True" ]]; then
            PYTHON_CMD="$cmd"
            break
        fi
    fi
done

[[ -z "$PYTHON_CMD" ]] && die "Python >= 3.10 non trovato. Installa python3.10 o superiore e riesegui."
ok "Python: $($PYTHON_CMD --version) → $PYTHON_CMD"

# ── 4. Ambiente virtuale ──────────────────────────────────────────────────────
step "4/9 · Ambiente virtuale Python"

if [[ -d "$VENV_DIR" ]]; then
    info "Ambiente virtuale già presente — verifico integrità..."
    # Rimuovi se corrotto (mancanza dell'eseguibile)
    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        warn "Ambiente virtuale corrotto, lo ricreo..."
        rm -rf "$VENV_DIR"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creo .venv con $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_DIR" || die "Impossibile creare il venv. Installa python3-venv."
    ok "Ambiente virtuale creato."
else
    ok "Ambiente virtuale esistente riusato."
fi

PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
GUNICORN="$VENV_DIR/bin/gunicorn"

# ── 5. Dipendenze Python ──────────────────────────────────────────────────────
step "5/9 · Dipendenze Python"
info "Aggiorno pip, setuptools, wheel..."
"$PIP" install --upgrade pip setuptools wheel --quiet

# numpy < 2 richiesto da prophet/pystan
info "Installo numpy (compatibile con Prophet)..."
"$PIP" install "numpy>=1.26,<2" --quiet

# pystan (dipendenza di prophet) compila Stan in C++ — può richiedere 5-15 min
info "Installo pystan (compilazione C++ Stan — può richiedere diversi minuti)..."
"$PIP" install "pystan>=3.0,<4" --quiet

# Prophet
info "Installo prophet..."
"$PIP" install "prophet>=1.1" --quiet

# Dipendenze scientifiche
info "Installo scikit-learn, statsmodels, pandas, matplotlib, plotly..."
"$PIP" install \
    "pandas>=2.1" \
    "scikit-learn>=1.4" \
    "statsmodels>=0.14" \
    "matplotlib>=3.8" \
    "plotly>=5.20" \
    "joblib>=1.3" \
    "tqdm>=4.66" \
    "openpyxl>=3.1" \
    --quiet

# Flask e server
info "Installo Flask, Flask-SQLAlchemy, gunicorn..."
"$PIP" install \
    "Flask>=3.0" \
    "Flask-SQLAlchemy>=3.1" \
    "gunicorn>=21.0" \
    --quiet

# Pacchetto locale (include src/micare_sicilia)
info "Installo pacchetto micare-sicilia in modalità editable..."
"$PIP" install -e "$APP_DIR" --quiet

ok "Tutte le dipendenze Python installate."

# Verifica importazioni critiche
info "Verifico importazioni critiche..."
"$PYTHON" -c "
import flask, sqlalchemy, prophet, sklearn, pandas, numpy, statsmodels, gunicorn
print('  flask', flask.__version__)
print('  pandas', pandas.__version__)
print('  numpy', numpy.__version__)
print('  prophet OK')
print('  scikit-learn', sklearn.__version__)
print('  statsmodels', statsmodels.__version__)
" || die "Alcune importazioni fallite. Controlla $LOG_FILE"
ok "Importazioni verificate."

# ── 6. Directory e struttura dati ─────────────────────────────────────────────
step "6/9 · Directory di lavoro"

mkdir -p "$APP_DIR/data" "$APP_DIR/models" "$APP_DIR/instance"
ok "data/, models/, instance/ pronti."

# ── 7. Configurazione Flask ───────────────────────────────────────────────────
step "7/9 · Configurazione applicazione"

CONFIG_FILE="$APP_DIR/instance/config.py"
if [[ ! -f "$CONFIG_FILE" ]]; then
    info "Genero SECRET_KEY casuale..."
    SECRET=$("$PYTHON" -c "import secrets; print(secrets.token_hex(32))")
    cat > "$CONFIG_FILE" <<EOF
# Configurazione locale — NON committare questo file
SECRET_KEY = "$SECRET"
EOF
    ok "instance/config.py generato."
else
    ok "instance/config.py già presente, non sovrascritto."
fi

# ── 8. Script di avvio ────────────────────────────────────────────────────────
step "8/9 · Script di avvio e servizio di sistema"

START_SCRIPT="$APP_DIR/start.sh"
cat > "$START_SCRIPT" <<STARTEOF
#!/usr/bin/env bash
# Avvia MIC RES Sicilia con gunicorn
APP_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$APP_DIR/.venv/bin/activate"
export FLASK_APP=run.py
export FLASK_ENV=production
export PYTHONPATH="\$APP_DIR/src:\$PYTHONPATH"
cd "\$APP_DIR"
exec "\$APP_DIR/.venv/bin/gunicorn" \\
    --bind 0.0.0.0:${APP_PORT} \\
    --workers 2 \\
    --threads 4 \\
    --worker-class gthread \\
    --timeout 3600 \\
    --graceful-timeout 120 \\
    --keep-alive 5 \\
    --log-level info \\
    --access-logfile - \\
    --error-logfile - \\
    "run:app"
STARTEOF
chmod +x "$START_SCRIPT"
ok "start.sh creato."

# ── Systemd (Linux) ───────────────────────────────────────────────────────────
SYSTEMD_AVAILABLE=false

if [[ "$OS" != "macos" ]] && command -v systemctl &>/dev/null; then
    UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
    CURRENT_USER="$(whoami)"
    CURRENT_GROUP="$(id -gn)"

    info "Creo/aggiorno unità systemd: $UNIT_FILE"
    sudo tee "$UNIT_FILE" > /dev/null <<EOF
[Unit]
Description=MIC RES Sicilia — Piattaforma previsione AMR
Documentation=https://github.com/ico88/micare-sicilia
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${CURRENT_USER}
Group=${CURRENT_GROUP}
WorkingDirectory=${APP_DIR}
ExecStart=${START_SCRIPT}
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10
StartLimitIntervalSec=60
StartLimitBurst=5

# Variabili ambiente
Environment=FLASK_ENV=production
Environment=PYTHONPATH=${APP_DIR}/src
Environment=PATH=${VENV_DIR}/bin:/usr/local/bin:/usr/bin:/bin

# Limiti risorse (Prophet usa molta RAM e CPU durante il training)
LimitNOFILE=65536
LimitNPROC=4096
TimeoutStartSec=120
TimeoutStopSec=60

# Output log verso systemd journal
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${SERVICE_NAME}

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable "${SERVICE_NAME}.service"
    ok "Servizio systemd '${SERVICE_NAME}' creato e abilitato (avvio automatico al boot)."
    SYSTEMD_AVAILABLE=true

    # Verifica che l'unità sia caricata correttamente
    STATUS=$(sudo systemctl is-enabled "$SERVICE_NAME" 2>/dev/null || echo "unknown")
    info "Stato servizio: $STATUS"

else
    # ── launchd (macOS) ───────────────────────────────────────────────────────
    if [[ "$OS" == "macos" ]]; then
        PLIST_DIR="$HOME/Library/LaunchAgents"
        PLIST_FILE="$PLIST_DIR/com.micare-sicilia.plist"
        mkdir -p "$PLIST_DIR"
        cat > "$PLIST_FILE" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.micare-sicilia</string>
    <key>ProgramArguments</key>
    <array>
        <string>${START_SCRIPT}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${APP_DIR}</string>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${APP_DIR}/logs/stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${APP_DIR}/logs/stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>FLASK_ENV</key>
        <string>production</string>
        <key>PYTHONPATH</key>
        <string>${APP_DIR}/src</string>
        <key>PATH</key>
        <string>${VENV_DIR}/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF
        mkdir -p "$APP_DIR/logs"
        launchctl unload "$PLIST_FILE" 2>/dev/null || true
        launchctl load -w "$PLIST_FILE"
        ok "launchd agent caricato (avvio automatico al login): $PLIST_FILE"
        SYSTEMD_AVAILABLE=macos
    fi
fi

# ── 9. Firewall ───────────────────────────────────────────────────────────────
step "9/9 · Firewall"

FIREWALL_OPENED=false

# UFW (Ubuntu/Debian)
if command -v ufw &>/dev/null; then
    if sudo ufw status 2>/dev/null | grep -q "Status: active"; then
        sudo ufw allow "${APP_PORT}/tcp" comment "MIC RES Sicilia" 2>/dev/null || true
        ok "UFW: porta $APP_PORT aperta."
        FIREWALL_OPENED=true
    else
        info "UFW non attivo, skip."
    fi
fi

# firewalld (Fedora/RHEL)
if command -v firewall-cmd &>/dev/null && ! $FIREWALL_OPENED; then
    if sudo firewall-cmd --state 2>/dev/null | grep -q "running"; then
        sudo firewall-cmd --permanent --add-port="${APP_PORT}/tcp" 2>/dev/null || true
        sudo firewall-cmd --reload 2>/dev/null || true
        ok "firewalld: porta $APP_PORT aperta."
        FIREWALL_OPENED=true
    fi
fi

if ! $FIREWALL_OPENED; then
    info "Nessun firewall attivo rilevato (o già configurato manualmente)."
fi

# ── Riepilogo ─────────────────────────────────────────────────────────────────
SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "IP_SERVER")

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗"
echo -e "║        Installazione completata!  ✓          ║"
echo -e "╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  ${BOLD}Accesso:${NC}"
echo -e "    Locale  →  ${CYAN}http://localhost:${APP_PORT}${NC}"
echo -e "    Rete    →  ${CYAN}http://${SERVER_IP}:${APP_PORT}${NC}"
echo ""
echo -e "  ${BOLD}Gestione servizio:${NC}"

if [[ "$SYSTEMD_AVAILABLE" == "true" ]]; then
    echo -e "    Avvia     →  ${CYAN}sudo systemctl start  ${SERVICE_NAME}${NC}"
    echo -e "    Ferma     →  ${CYAN}sudo systemctl stop   ${SERVICE_NAME}${NC}"
    echo -e "    Riavvia   →  ${CYAN}sudo systemctl restart ${SERVICE_NAME}${NC}"
    echo -e "    Stato     →  ${CYAN}sudo systemctl status ${SERVICE_NAME}${NC}"
    echo -e "    Log live  →  ${CYAN}sudo journalctl -fu ${SERVICE_NAME}${NC}"
elif [[ "$SYSTEMD_AVAILABLE" == "macos" ]]; then
    echo -e "    Stop      →  ${CYAN}launchctl unload ~/Library/LaunchAgents/com.micare-sicilia.plist${NC}"
    echo -e "    Log       →  ${CYAN}tail -f $APP_DIR/logs/stdout.log${NC}"
else
    echo -e "    Avvia     →  ${CYAN}bash start.sh${NC}"
fi

echo ""
echo -e "  ${BOLD}Log installazione:${NC} $LOG_FILE"
echo ""
echo -e "  ${YELLOW}Primo avvio:${NC}"
echo -e "    1. Apri http://localhost:${APP_PORT} → sezione 'Dati'"
echo -e "    2. Carica i file Excel annuali della Rete MIC"
echo -e "    3. Avvia addestramento (Prophet: ore di elaborazione — normale)"
echo ""

# ── Avvio immediato ───────────────────────────────────────────────────────────
read -rp "Avviare l'applicazione adesso? [S/n] " LAUNCH
LAUNCH="${LAUNCH:-S}"
if [[ "$LAUNCH" =~ ^[Ss]$ ]]; then
    echo ""
    if [[ "$SYSTEMD_AVAILABLE" == "true" ]]; then
        sudo systemctl start "$SERVICE_NAME"
        sleep 2
        if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
            ok "Servizio avviato con successo."
            echo -e "  Apri: ${CYAN}http://localhost:${APP_PORT}${NC}"
        else
            warn "Il servizio non sembra attivo. Controlla: sudo journalctl -fu ${SERVICE_NAME}"
        fi
    elif [[ "$SYSTEMD_AVAILABLE" == "macos" ]]; then
        ok "Servizio launchd già caricato e in esecuzione."
        echo -e "  Apri: ${CYAN}http://localhost:${APP_PORT}${NC}"
    else
        info "Avvio in foreground (Ctrl+C per fermare)..."
        exec bash "$START_SCRIPT"
    fi
fi
