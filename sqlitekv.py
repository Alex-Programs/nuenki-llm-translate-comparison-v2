import sqlite3
import threading


class SQLiteKVCache:
    def __init__(self, path):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """
        )
        self.conn.commit()

    def get(self, key):
        with self.lock:
            cur = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,))
            row = cur.fetchone()
            return row[0] if row else None

    def set(self, key, value):
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("Only string keys and values are supported.")
        with self.lock:
            self.conn.execute(
                "REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value)
            )
            self.conn.commit()

    def delete(self, key):
        with self.lock:
            self.conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            self.conn.commit()

    def keys(self):
        with self.lock:
            cur = self.conn.execute("SELECT key FROM kv")
            return [row[0] for row in cur.fetchall()]

    def close(self):
        self.conn.close()
