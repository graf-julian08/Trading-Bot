# Deployment Guide — Self-Learning AI Trading Bot

## Prerequisites

- A Linux VPS (Ubuntu 22.04+ recommended) with at least 2 GB RAM
- Root or sudo access
- An exchange account with API key + secret (Binance recommended to start)
- (Optional) A Telegram bot token for notifications

---

## Quick Deploy (Automated)

```bash
# 1. Clone or copy the project to your VPS.
scp -r ./* user@your-vps:/tmp/trading-bot/

# 2. SSH into your VPS.
ssh user@your-vps

# 3. Run the setup script.
cd /tmp/trading-bot
chmod +x setup.sh
sudo ./setup.sh

# 4. Edit the .env file with your API keys.
sudo nano /opt/trading-bot/.env

# 5. Start the bot.
sudo systemctl start trading-bot

# 6. Verify it's running.
sudo systemctl status trading-bot
sudo journalctl -u trading-bot -f
```

---

## Manual Deploy

### Step 1: Install Python 3.10+

```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev build-essential sqlite3
```

### Step 2: Create a Virtual Environment

```bash
cd /opt/trading-bot
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
cp .env.example .env
nano .env
# Fill in:
#   EXCHANGE_API_KEY, EXCHANGE_API_SECRET
#   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (optional)
#   Set PAPER_TRADE=true for testing
```

### Step 4: Test with Dry Run

```bash
source venv/bin/activate
python main.py --dry-run
```

This runs a single iteration of the loop with paper trading and exits.
Check the output for any errors.

### Step 5: Run as a Service

```bash
# Copy the systemd unit file.
sudo cp trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

---

## Managing the Bot

| Action          | Command                                     |
|-----------------|---------------------------------------------|
| Start           | `sudo systemctl start trading-bot`          |
| Stop            | `sudo systemctl stop trading-bot`           |
| Restart         | `sudo systemctl restart trading-bot`        |
| Status          | `sudo systemctl status trading-bot`         |
| Live logs       | `sudo journalctl -u trading-bot -f`         |
| Last 100 lines  | `sudo journalctl -u trading-bot -n 100`     |

---

## Going Live (Real Money)

> **⚠️ WARNING**: Only do this after thorough paper-trading testing.

1. Edit `/opt/trading-bot/.env`:
   ```
   PAPER_TRADE=false
   EXCHANGE_SANDBOX=false
   ```
2. Restart: `sudo systemctl restart trading-bot`
3. Monitor closely for the first 24 hours via Telegram and logs.

---

## Setting Up Telegram Notifications

1. Create a bot with [@BotFather](https://t.me/BotFather) → copy the token.
2. Send any message to your bot → then visit:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
   to find your chat ID.
3. Update `.env`:
   ```
   TELEGRAM_ENABLED=true
   TELEGRAM_BOT_TOKEN=123456:ABC-DEF
   TELEGRAM_CHAT_ID=987654321
   ```
4. Restart the bot.

---

## Database

The SQLite database is stored at `/opt/trading-bot/trading_bot.db`.

You can inspect it with:

```bash
sqlite3 /opt/trading-bot/trading_bot.db
.tables
SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10;
SELECT * FROM daily_pnl ORDER BY date DESC LIMIT 7;
.quit
```

---

## Troubleshooting

| Symptom                         | Fix                                                        |
|---------------------------------|------------------------------------------------------------|
| Bot exits immediately           | Check `journalctl -u trading-bot -n 50` for errors         |
| "Exchange not available"        | Verify API keys and network connectivity                   |
| Kill switch keeps triggering    | Increase `DAILY_DRAWDOWN_LIMIT` in `.env`                  |
| Model accuracy very low         | Increase `ML_TRAINING_CANDLES`, check data quality          |
| No trades being placed          | Lower `ML_THRESHOLD` or check if spread is too high         |
