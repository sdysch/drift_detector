"""Local SQLite store for prediction requests and responses."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from sqlite3 import connect, register_adapter
from typing import Any, Self

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "predictions.db"
)

_FEATURE_COLUMNS = [
    ("feature_1", "REAL"),
    ("feature_2", "REAL"),
    ("feature_3", "REAL"),
    ("feature_gaussian", "REAL"),
    ("feature_lognormal", "REAL"),
    ("feature_exponential", "REAL"),
    ("category", "TEXT"),
    ("type", "TEXT"),
]

_FEATURE_NAMES = [c for c, _ in _FEATURE_COLUMNS]

register_adapter(np.float64, float)
register_adapter(np.float32, float)


class PredictionStore:
    def __init__(self, db_path: str | Path = _DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._conn = connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = None
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self) -> None:
        cols = ", ".join(f"{name} {dtype}" for name, dtype in _FEATURE_COLUMNS)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                {cols},
                prediction  REAL    NOT NULL,
                created_at  TEXT    NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_predictions_created_at
            ON predictions (created_at)
            """
        )
        self._conn.commit()

    def insert(self, features: dict[str, Any], prediction: float) -> int:
        now = datetime.now(timezone.utc).isoformat()
        names = _FEATURE_NAMES
        placeholders = ", ".join("?" for _ in names)
        values = [features[c] for c in names]
        cols = ", ".join(names)
        cur = self._conn.execute(
            f"INSERT INTO predictions ({cols}, prediction, created_at) VALUES ({placeholders}, ?, ?)",
            (*values, prediction, now),
        )
        self._conn.commit()
        logger.debug("Stored prediction id=%s", cur.lastrowid)
        return cur.lastrowid

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
