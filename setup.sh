#!/usr/bin/env bash
# ============================================================================
# setup.sh — Automated VPS Deployment Script
# ============================================================================
# Run on a fresh Ubuntu 22.04+ VPS:
#   chmod +x setup.sh && sudo ./setup.sh
#
# What this script does:
#   1. Installs Python 3.10+ and system dependencies.
#   2. Creates a dedicated 'tradingbot' system user.
#   3. Sets up a Python virtual environment with all pip dependencies.
#   4. Creates a .env template for API keys.
#   5. Installs and enables a systemd service for 24/7 operation.
# ============================================================================

set -euo pipefail

# ---- Configuration ----
APP_NAME="trading-bot"
APP_DIR="/opt/${APP_NAME}"
APP_USER="tradingbot"
VENV_DIR="${APP_DIR}/venv"
SERVICE_FILE="/etc/systemd/system/${APP_NAME}.service"

echo "==========================================="
echo "  Trading Bot — VPS Setup Script"
echo "==========================================="

# ---- 1. System packages ----
echo "[1/6] Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    sqlite3 \
    > /dev/null 2>&1

# Ensure python3.11 is the default python3.
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 2>/dev/null || true

echo "   Python version: $(python3 --version)"

# ---- 2. Create system user ----
echo "[2/6] Creating system user '${APP_USER}'..."
if ! id "${APP_USER}" &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir "${APP_DIR}" "${APP_USER}"
    echo "   User '${APP_USER}' created."
else
    echo "   User '${APP_USER}' already exists."
fi

# ---- 3. Deploy application files ----
echo "[3/6] Deploying application to ${APP_DIR}..."
mkdir -p "${APP_DIR}"

# Copy project files (assumes this script is run from the project root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp -r "${SCRIPT_DIR}"/*.py "${APP_DIR}/"
cp -r "${SCRIPT_DIR}/requirements.txt" "${APP_DIR}/"

# ---- 4. Virtual environment ----
echo "[4/6] Setting up Python virtual environment..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip setuptools wheel -q
"${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt" -q
echo "   Dependencies installed."

# ---- 5. Create .env template ----
ENV_FILE="${APP_DIR}/.env"
if [ ! -f "${ENV_FILE}" ]; then
    echo "[5/6] Creating .env template..."
    cat > "${ENV_FILE}" <<'ENVEOF'
# ============================================================================
# Trading Bot — Environment Variables
# ============================================================================
# IMPORTANT: Fill in your actual API keys below.
# NEVER commit this file to version control.

# ---- Exchange ----
EXCHANGE_ID=binance
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here
EXCHANGE_SANDBOX=true

# ---- Paper Trading (set to false for live trading) ----
PAPER_TRADE=true

# ---- Trading Pairs (comma-separated) ----
TRADING_PAIRS=BTC/USDT,ETH/USDT

# ---- Timeframe ----
TIMEFRAME=1h

# ---- Risk ----
MAX_RISK_PER_TRADE=0.02
STOP_LOSS_PCT=0.02
TAKE_PROFIT_PCT=0.04
DAILY_DRAWDOWN_LIMIT=-0.05
MAX_OPEN_POSITIONS=3

# ---- Compounding ----
COMPOUND_MODE=true

# ---- ML ----
ML_THRESHOLD=0.65
ML_RETRAIN_INTERVAL_DAYS=7

# ---- Telegram ----
TELEGRAM_ENABLED=false
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# ---- Logging ----
LOG_LEVEL=INFO
ENVEOF
    echo "   .env template created at ${ENV_FILE}"
    echo "   >>> EDIT THIS FILE WITH YOUR API KEYS BEFORE STARTING <<<"
else
    echo "[5/6] .env already exists — skipping."
fi

# ---- 6. Set permissions & install systemd service ----
echo "[6/6] Setting permissions and installing systemd service..."
chown -R "${APP_USER}:${APP_USER}" "${APP_DIR}"
chmod 600 "${ENV_FILE}"

# Copy systemd service file.
if [ -f "${SCRIPT_DIR}/trading-bot.service" ]; then
    cp "${SCRIPT_DIR}/trading-bot.service" "${SERVICE_FILE}"
else
    # Generate inline if not present.
    cat > "${SERVICE_FILE}" <<SVCEOF
[Unit]
Description=Self-Learning AI Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${APP_USER}
Group=${APP_USER}
WorkingDirectory=${APP_DIR}
ExecStart=${VENV_DIR}/bin/python main.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=${APP_NAME}

# Security hardening.
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=${APP_DIR}
PrivateTmp=yes

# Environment.
EnvironmentFile=${ENV_FILE}

[Install]
WantedBy=multi-user.target
SVCEOF
fi

systemctl daemon-reload
systemctl enable "${APP_NAME}.service"

echo ""
echo "==========================================="
echo "  Setup Complete!"
echo "==========================================="
echo ""
echo "  Next steps:"
echo "  1. Edit ${ENV_FILE} with your API keys."
echo "  2. Start the bot:"
echo "       sudo systemctl start ${APP_NAME}"
echo "  3. Check status:"
echo "       sudo systemctl status ${APP_NAME}"
echo "  4. View logs:"
echo "       sudo journalctl -u ${APP_NAME} -f"
echo "  5. Stop the bot:"
echo "       sudo systemctl stop ${APP_NAME}"
echo ""
