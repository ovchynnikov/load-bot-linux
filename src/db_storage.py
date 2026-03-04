"""SQLite storage for bot user data persistence."""

import sqlite3
import json
import os
from logger import debug, error


class BotStorage:
    """Handles persistent storage of user data in SQLite."""

    def __init__(self, db_path="data/bot.db"):
        """Initialize database connection and create tables."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
        debug("Database initialized at %s", db_path)

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
                last_seen REAL
            )
        """)
        self.conn.commit()

    def load_user_data(self, user_id):
        """Load user data from database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM user_data WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return {
                "conversation_context": json.loads(row[1]) if row[1] else [],
                "rate_limit_timestamps": json.loads(row[2]) if row[2] else [],
                "daily_count": row[3],
                "daily_date": row[4],
                "last_seen": row[5],
            }
        return None

    def save_user_data(self, user_id, conversation_context, rate_limit_timestamps, daily_count, daily_date, last_seen):
        """Save user data to database."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO user_data 
            (user_id, conversation_context, rate_limit_timestamps, daily_count, daily_date, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
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
        self.conn.commit()

    def delete_user_data(self, user_id):
        """Delete user data from database."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM user_data WHERE user_id = ?", (user_id,))
        self.conn.commit()

    def get_stale_users(self, ttl_seconds):
        """Get list of user IDs that haven't been seen within TTL."""
        import time

        current_time = time.time()
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT user_id FROM user_data WHERE last_seen < ?", (current_time - ttl_seconds,)
        )
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        """Close database connection."""
        self.conn.close()
