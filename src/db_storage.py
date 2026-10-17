"""SQLite storage for bot user data persistence."""

import sqlite3
import json
import os
import time
import logging
import threading
import uuid
from datetime import datetime, timezone, timedelta
from functools import wraps


def synchronized(method):
    """Serialize whole operations and roll back failed writes before releasing the lock."""

    @wraps(method)
    def locked(self, *args, **kwargs):
        with self.lock:
            if method.__name__ == "close":
                return method(self, *args, **kwargs)
            with self.conn:
                return method(self, *args, **kwargs)

    return locked


class BotStorage:
    """Handles persistent storage of user data in SQLite."""

    def __init__(self, db_path="data/bot.db"):
        """Initialize database connection and create tables."""
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        logging.getLogger(__name__).debug("Database initialized at %s", db_path)

    def _create_tables(self):
        """Create tables if they don't exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_data (
                user_id INTEGER PRIMARY KEY,
                conversation_context TEXT,
                rate_limit_timestamps TEXT,
                daily_count INTEGER DEFAULT 0,
                daily_date TEXT,
                last_seen REAL,
                img_gen_rate_limit_timestamps TEXT,
                img_gen_daily_count INTEGER DEFAULT 0,
                img_gen_daily_date TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_data_last_seen ON user_data(last_seen)")
        # Migrate existing tables that don't have image gen columns
        for col, definition in [
            ("img_gen_rate_limit_timestamps", "TEXT"),
            ("img_gen_daily_count", "INTEGER DEFAULT 0"),
            ("img_gen_daily_date", "TEXT"),
        ]:
            try:
                cursor.execute(f"ALTER TABLE user_data ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError as e:
                # Only suppress the error if the column already exists
                error_msg = str(e).lower()
                if "duplicate" not in error_msg and "already exists" not in error_msg:
                    raise
                # Column already exists, continue
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS search_quota (scope TEXT, day TEXT, count INTEGER NOT NULL, PRIMARY KEY(scope, day))"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS search_reservations (id TEXT PRIMARY KEY, user_id INTEGER, day TEXT, state TEXT)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS search_notices (user_id INTEGER, day TEXT, PRIMARY KEY(user_id, day))"
        )
        self.conn.commit()

    @synchronized
    def load_user_data(self, user_id):
        """Load user data from database."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT user_id, conversation_context, rate_limit_timestamps, daily_count, daily_date, last_seen,"
            " img_gen_rate_limit_timestamps, img_gen_daily_count, img_gen_daily_date"
            " FROM user_data WHERE user_id = ?",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "conversation_context": json.loads(row[1]) if row[1] else [],
                "rate_limit_timestamps": json.loads(row[2]) if row[2] else [],
                "daily_count": row[3],
                "daily_date": row[4],
                "last_seen": row[5],
                "img_gen_rate_limit_timestamps": json.loads(row[6]) if row[6] else [],
                "img_gen_daily_count": row[7] or 0,
                "img_gen_daily_date": row[8] or "",
            }
        return None

    @synchronized
    def save_user_data(
        self,
        user_id,
        conversation_context,
        rate_limit_timestamps,
        daily_count,
        daily_date,
        last_seen,
        img_gen_rate_limit_timestamps=None,
        img_gen_daily_count=0,
        img_gen_daily_date="",
    ):
        """Save user data to database."""
        self.conn.execute(
            """
            INSERT INTO user_data
            (user_id, conversation_context, rate_limit_timestamps, daily_count, daily_date, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                conversation_context=excluded.conversation_context,
                rate_limit_timestamps=excluded.rate_limit_timestamps,
                daily_count=excluded.daily_count, daily_date=excluded.daily_date,
                last_seen=MAX(COALESCE(user_data.last_seen, 0), excluded.last_seen)
            """,
            (
                user_id,
                json.dumps(conversation_context),
                json.dumps(rate_limit_timestamps),
                daily_count,
                daily_date,
                last_seen,
            ),
        )
        # Optional image arguments are retained for old callers, but LLM saves omit them.
        if img_gen_rate_limit_timestamps is not None:
            self.conn.execute(
                "UPDATE user_data SET img_gen_rate_limit_timestamps=?, img_gen_daily_count=?, img_gen_daily_date=? WHERE user_id=?",
                (json.dumps(img_gen_rate_limit_timestamps), img_gen_daily_count, img_gen_daily_date, user_id),
            )

    @synchronized
    def delete_user_data(self, user_id):
        """Delete user data from database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM user_data WHERE user_id = ?", (user_id,))
        self.conn.commit()

    @synchronized
    def update_user_image_limits(self, user_id, img_gen_rate_limit_timestamps, img_gen_daily_count, img_gen_daily_date):
        """Update or insert image generation limit fields.

        Ensures the user row exists with image limit fields set, either by updating
        an existing row or creating a new one with defaults for other fields.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_data (user_id, img_gen_rate_limit_timestamps, img_gen_daily_count, img_gen_daily_date)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                img_gen_rate_limit_timestamps = excluded.img_gen_rate_limit_timestamps,
                img_gen_daily_count = excluded.img_gen_daily_count,
                img_gen_daily_date = excluded.img_gen_daily_date
            """,
            (
                user_id,
                json.dumps(img_gen_rate_limit_timestamps or []),
                img_gen_daily_count,
                img_gen_daily_date,
            ),
        )
        self.conn.commit()

    @synchronized
    def get_stale_users(self, ttl_seconds):
        """Get list of user IDs that haven't been seen within TTL."""
        current_time = time.time()
        cursor = self.conn.cursor()
        cursor.execute("SELECT user_id FROM user_data WHERE last_seen < ?", (current_time - ttl_seconds,))
        return [row[0] for row in cursor.fetchall()]

    @synchronized
    def close(self):
        """Close database connection."""
        self.conn.close()

    @staticmethod
    def utc_day():
        """New search quotas use UTC independently of the legacy LLM quota timezone."""
        return datetime.now(timezone.utc).date().isoformat()

    @synchronized
    def reserve_search(self, user_id, user_limit, global_limit, day=None):
        """Atomically reserve both scopes; a durable ID makes later settlement idempotent."""
        day = day or self.utc_day()
        if min(user_limit, global_limit) <= 0:
            return None
        self.conn.execute("BEGIN IMMEDIATE")
        scopes = [(f"user:{user_id}", user_limit), ("global", global_limit)]
        for scope, limit in scopes:
            row = self.conn.execute("SELECT count FROM search_quota WHERE scope=? AND day=?", (scope, day)).fetchone()
            if row and row[0] >= limit:
                return None
        for scope, _ in scopes:
            self.conn.execute(
                "INSERT INTO search_quota VALUES (?, ?, 1) ON CONFLICT(scope, day) DO UPDATE SET count=count+1",
                (scope, day),
            )
        reservation = uuid.uuid4().hex
        self.conn.execute("INSERT INTO search_reservations VALUES (?, ?, ?, 'reserved')", (reservation, user_id, day))
        return reservation

    @synchronized
    def settle_search(self, reservation, refund=False):
        """Release verified zero-use requests only once, against their original UTC day."""
        self.conn.execute("BEGIN IMMEDIATE")
        row = self.conn.execute(
            "SELECT user_id, day, state FROM search_reservations WHERE id=?", (reservation,)
        ).fetchone()
        if row is None or row[2] != "reserved":
            return
        if refund:
            for scope in (f"user:{row[0]}", "global"):
                cursor = self.conn.execute(
                    "UPDATE search_quota SET count=count-1 WHERE scope=? AND day=? AND count>0", (scope, row[1])
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("Missing or inconsistent search reservation counter")
        self.conn.execute(
            "UPDATE search_reservations SET state=? WHERE id=?", ("released" if refund else "retained", reservation)
        )

    @synchronized
    def search_notice_shown(self, user_id, day):
        return (
            self.conn.execute("SELECT 1 FROM search_notices WHERE user_id=? AND day=?", (user_id, day)).fetchone()
            is not None
        )

    @synchronized
    def mark_search_notice(self, user_id, day):
        self.conn.execute("INSERT OR IGNORE INTO search_notices VALUES (?, ?)", (user_id, day))

    @synchronized
    def prune_search_history(self):
        """Keep recent UTC days, including every realistically active reservation."""
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
        for table in ("search_quota", "search_reservations", "search_notices"):
            self.conn.execute(f"DELETE FROM {table} WHERE day < ?", (cutoff,))
