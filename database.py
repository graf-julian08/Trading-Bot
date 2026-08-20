"""
database.py — Async SQLite Persistence Layer
=============================================
Provides an async wrapper around SQLite (via aiosqlite) for:
  - Trade history logging
  - Daily PnL tracking
  - ML model checkpoint storage (serialised blobs)
  - OHLCV cache (optional, for faster restarts)

All database operations are non-blocking so they never stall the main
trading loop.

Usage:
    db = Database()
    await db.initialise()
    await db.log_trade(...)
    await db.close()
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import aiosqlite

from config import cfg

logger = logging.getLogger(__name__)

# ============================================================================
# Schema Definitions
# ============================================================================

_SCHEMA_SQL = """
-- Trade log: every order placed by the bot.
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,              -- 'buy' or 'sell'
    order_type      TEXT    NOT NULL,              -- 'limit' or 'market'
    price           REAL    NOT NULL,
    amount          REAL    NOT NULL,
    cost            REAL    NOT NULL,              -- price * amount
    fee             REAL    DEFAULT 0.0,
    stop_loss       REAL,
    take_profit     REAL,
    status          TEXT    DEFAULT 'open',        -- open / closed / cancelled
    pnl             REAL    DEFAULT 0.0,
    ml_probability  REAL    DEFAULT 0.0,           -- model confidence at entry
    entry_time      REAL    NOT NULL,              -- Unix timestamp
    exit_time       REAL,
    metadata        TEXT    DEFAULT '{}'            -- JSON blob for extra info
);

-- Daily profit/loss snapshots for the kill-switch and reporting.
CREATE TABLE IF NOT EXISTS daily_pnl (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL UNIQUE,            -- YYYY-MM-DD
    starting_equity  REAL NOT NULL,
    ending_equity    REAL,
    realised_pnl     REAL DEFAULT 0.0,
    trade_count      INTEGER DEFAULT 0,
    kill_switch_triggered INTEGER DEFAULT 0        -- boolean flag
);

-- Serialised ML model checkpoints.
CREATE TABLE IF NOT EXISTS model_states (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    model_blob  BLOB   NOT NULL,                   -- pickled model bytes
    accuracy    REAL   DEFAULT 0.0,
    trained_at  REAL   NOT NULL,                   -- Unix timestamp
    candle_count INTEGER DEFAULT 0,
    metadata    TEXT   DEFAULT '{}'
);

-- Index for fast lookups on trades by symbol and status.
CREATE INDEX IF NOT EXISTS idx_trades_symbol_status
    ON trades(symbol, status);

-- Index for fast daily PnL lookup.
CREATE INDEX IF NOT EXISTS idx_daily_pnl_date
    ON daily_pnl(date);
"""


class Database:
    """Async SQLite database manager for the trading bot."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or cfg.DB_PATH
        self._conn: Optional[aiosqlite.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialise(self) -> None:
        """Open the database connection and create tables if needed."""
        logger.info("Opening database at %s", self._db_path)
        self._conn = await aiosqlite.connect(self._db_path)
        # Enable WAL mode for better concurrent read performance.
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._conn.executescript(_SCHEMA_SQL)
        await self._conn.commit()
        logger.info("Database initialised successfully.")

    async def close(self) -> None:
        """Gracefully close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed.")

    # ------------------------------------------------------------------
    # Trade Operations
    # ------------------------------------------------------------------

    async def log_trade(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: float,
        amount: float,
        cost: float,
        fee: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        ml_probability: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a new trade record and return its ID."""
        sql = """
            INSERT INTO trades
                (symbol, side, order_type, price, amount, cost, fee,
                 stop_loss, take_profit, ml_probability, entry_time, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        meta_json = json.dumps(metadata or {})
        cursor = await self._conn.execute(
            sql,
            (
                symbol, side, order_type, price, amount, cost, fee,
                stop_loss, take_profit, ml_probability, time.time(), meta_json,
            ),
        )
        await self._conn.commit()
        trade_id = cursor.lastrowid
        logger.info("Logged trade #%d: %s %s %s @ %.8f", trade_id, side, amount, symbol, price)
        return trade_id

    async def close_trade(
        self, trade_id: int, exit_price: float, pnl: float, fee: float = 0.0
    ) -> None:
        """Mark a trade as closed with realised PnL."""
        sql = """
            UPDATE trades
               SET status = 'closed',
                   pnl = ?,
                   fee = fee + ?,
                   exit_time = ?
             WHERE id = ?
        """
        await self._conn.execute(sql, (pnl, fee, time.time(), trade_id))
        await self._conn.commit()
        logger.info("Closed trade #%d with PnL=%.4f", trade_id, pnl)

    async def get_open_trades(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve all currently open trades, optionally filtered by symbol."""
        if symbol:
            sql = "SELECT * FROM trades WHERE status = 'open' AND symbol = ?"
            cursor = await self._conn.execute(sql, (symbol,))
        else:
            sql = "SELECT * FROM trades WHERE status = 'open'"
            cursor = await self._conn.execute(sql)
        rows = await cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in rows]

    async def get_trade_by_id(self, trade_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single trade by its ID."""
        cursor = await self._conn.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    async def count_open_trades(self) -> int:
        """Return the total number of currently open trades."""
        cursor = await self._conn.execute(
            "SELECT COUNT(*) FROM trades WHERE status = 'open'"
        )
        row = await cursor.fetchone()
        return row[0]

    # ------------------------------------------------------------------
    # Daily PnL Operations
    # ------------------------------------------------------------------

    async def upsert_daily_pnl(
        self,
        date_str: str,
        starting_equity: float,
        ending_equity: Optional[float] = None,
        realised_pnl: float = 0.0,
        trade_count: int = 0,
        kill_switch_triggered: bool = False,
    ) -> None:
        """Insert or update the daily PnL row for a given date."""
        sql = """
            INSERT INTO daily_pnl
                (date, starting_equity, ending_equity, realised_pnl,
                 trade_count, kill_switch_triggered)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                ending_equity = excluded.ending_equity,
                realised_pnl = excluded.realised_pnl,
                trade_count = excluded.trade_count,
                kill_switch_triggered = excluded.kill_switch_triggered
        """
        await self._conn.execute(
            sql,
            (
                date_str, starting_equity, ending_equity, realised_pnl,
                trade_count, int(kill_switch_triggered),
            ),
        )
        await self._conn.commit()

    async def get_daily_pnl(self, date_str: str) -> Optional[Dict[str, Any]]:
        """Retrieve the daily PnL record for a given date string."""
        cursor = await self._conn.execute(
            "SELECT * FROM daily_pnl WHERE date = ?", (date_str,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    # ------------------------------------------------------------------
    # Model State Operations
    # ------------------------------------------------------------------

    async def save_model_state(
        self,
        symbol: str,
        model_blob: bytes,
        accuracy: float,
        candle_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist a serialised ML model checkpoint."""
        sql = """
            INSERT INTO model_states
                (symbol, model_blob, accuracy, trained_at, candle_count, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        cursor = await self._conn.execute(
            sql,
            (
                symbol, model_blob, accuracy, time.time(),
                candle_count, json.dumps(metadata or {}),
            ),
        )
        await self._conn.commit()
        model_id = cursor.lastrowid
        logger.info(
            "Saved model state #%d for %s (accuracy=%.4f, candles=%d)",
            model_id, symbol, accuracy, candle_count,
        )
        return model_id

    async def load_latest_model_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Load the most recent model checkpoint for a given symbol."""
        sql = """
            SELECT * FROM model_states
             WHERE symbol = ?
             ORDER BY trained_at DESC
             LIMIT 1
        """
        cursor = await self._conn.execute(sql, (symbol,))
        row = await cursor.fetchone()
        if row is None:
            return None
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    async def execute_raw(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute arbitrary SQL and return results as list of dicts."""
        cursor = await self._conn.execute(sql, params)
        rows = await cursor.fetchall()
        if cursor.description:
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        return []
