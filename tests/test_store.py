import sqlite3
import tempfile

import pytest

from drift_detector.api.store import PredictionStore, _FEATURE_NAMES


@pytest.fixture
def store():
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        yield PredictionStore(f.name)


_SAMPLE_FEATURES = {
    "feature_1": 0.5,
    "feature_2": 1.2,
    "feature_3": -0.3,
    "feature_gaussian": 0.1,
    "feature_lognormal": 2.5,
    "feature_exponential": 0.8,
    "category": "A",
    "type": "type_1",
}


class TestTableCreated:
    def test_table_exists(self, store):
        conn = sqlite3.connect(store.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        ).fetchall()
        assert len(tables) == 1

    def test_has_expected_columns(self, store):
        conn = sqlite3.connect(store.db_path)
        cols = {
            row[1]: row[2] for row in conn.execute("PRAGMA table_info(predictions)")
        }
        for name in _FEATURE_NAMES:
            assert name in cols
        assert cols["prediction"] == "REAL"
        assert cols["created_at"] == "TEXT"
        assert "id" in cols

    def test_feature_types_match_schema(self, store):
        conn = sqlite3.connect(store.db_path)
        cols = {
            row[1]: row[2] for row in conn.execute("PRAGMA table_info(predictions)")
        }
        assert cols["feature_1"] == "REAL"
        assert cols["feature_2"] == "REAL"
        assert cols["category"] == "TEXT"
        assert cols["type"] == "TEXT"


class TestInsert:
    def test_insert_returns_id(self, store):
        rid = store.insert(_SAMPLE_FEATURES, 0.42)
        assert isinstance(rid, int)
        assert rid > 0

    def test_insert_stores_values(self, store):
        store.insert(_SAMPLE_FEATURES, 0.42)
        conn = sqlite3.connect(store.db_path)
        row = conn.execute("SELECT * FROM predictions").fetchone()
        assert float(row[1]) == 0.5
        assert float(row[6]) == 0.8
        assert row[7] == "A"
        assert row[8] == "type_1"
        assert float(row[9]) == 0.42

    def test_insert_sets_timestamp(self, store):
        store.insert(_SAMPLE_FEATURES, 0.42)
        conn = sqlite3.connect(store.db_path)
        ts = conn.execute("SELECT created_at FROM predictions").fetchone()[0]
        assert ts.endswith("+00:00") or ts.endswith("Z") or ("T" in ts)

    def test_insert_increments_id(self, store):
        id1 = store.insert(_SAMPLE_FEATURES, 0.42)
        id2 = store.insert(_SAMPLE_FEATURES, 0.99)
        assert id2 == id1 + 1


class TestIndex:
    def test_created_at_index_exists(self, store):
        conn = sqlite3.connect(store.db_path)
        idx = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_predictions_created_at'"
        ).fetchone()
        assert idx is not None
