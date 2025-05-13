import sqlite3
import threading


class SQLiteKVCache:
    def __init__(self, path):
        self.path = path
        self.local = threading.local()
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """
        )
        conn.commit()
        conn.close()

    def _conn(self):
        if not hasattr(self.local, "conn"):
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self.local.conn = conn
        return self.local.conn

    def get(self, key):
        cur = self._conn().execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set(self, key, value):
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("Only string keys and values are supported.")
        conn = self._conn()
        conn.execute("REPLACE INTO kv (key, value) VALUES (?, ?)", (key, value))
        conn.commit()

    def delete(self, key):
        conn = self._conn()
        conn.execute("DELETE FROM kv WHERE key = ?", (key,))
        conn.commit()

    def keys(self):
        cur = self._conn().execute("SELECT key FROM kv")
        return [row[0] for row in cur.fetchall()]

    def close(self):
        if hasattr(self.local, "conn"):
            self.local.conn.close()
            del self.local.conn
