"""Controllable mock tools used by the verifier regression harness."""

import os
import sqlite3
from typing import Any, Dict

DB_PATH = "demo.db"


def setup_demo_db():
    """Creates a small real SQLite db to test against."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY,
                name TEXT,
                email TEXT
            )
        """)
        conn.executemany(
            "INSERT INTO customers (id, name, email) VALUES (?, ?, ?)",
            [
                (200, "Alice", "alice@newmail.com"),
                (2, "Bob", "bob@example.com"),
                (3, "Carol", "carol@example.com"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


def fake_agent_call(scenario: str):
    """Return deterministic SQL for the original SQL-agent scenarios."""
    if scenario == "honest_success":
        sql = "UPDATE customers SET email = 'alice@newmail.com' WHERE id = 1"
    elif scenario == "false_success":
        sql = "UPDATE customers SET email = 'ghost@nowhere.com' WHERE id = 999"
    elif scenario == "honest_failure":
        sql = "UPDATE customres SET email = 'x' WHERE id = 1"
    elif scenario == "hallucinated_table":
        sql = "UPDATE user_accounts SET email = 'x' WHERE id = 1"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return {"generated_sql": sql, "scenario_forced": scenario}


# ------------------------------------------------------------------
# SQL mutation mock tools
# ------------------------------------------------------------------
def sql_mock_honest_success(query: str, db_path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(query)
        conn.commit()
        return {"status": "SUCCESS", "rows_affected_by_tool": cur.rowcount}
    finally:
        conn.close()


def sql_mock_false_success(query: str, db_path: str) -> Dict[str, Any]:
    return {"status": "SUCCESS", "rows_affected_by_tool": 1}


def sql_mock_honest_failure(query: str, db_path: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("UPDATE missing_table SET value = 1")
        conn.commit()
        return {"status": "SUCCESS"}
    finally:
        conn.close()


def sql_mock_liar(query: str, db_path: str) -> Dict[str, Any]:
    return {"status": "SUCCESS", "rows_affected_by_tool": 999}


# ------------------------------------------------------------------
# SQL schema mock tools
# ------------------------------------------------------------------
def schema_mock_honest_success(query: str, db_path: str) -> Dict[str, Any]:
    return {"status": "SUCCESS", "message": "schema is valid"}


def schema_mock_false_success(query: str, db_path: str) -> Dict[str, Any]:
    return {"status": "SUCCESS", "message": "schema is valid"}


def schema_mock_honest_failure(query: str, db_path: str) -> Dict[str, Any]:
    raise sqlite3.OperationalError("no such table: missing_table")


def schema_mock_liar(query: str, db_path: str) -> Dict[str, Any]:
    return {"status": "SUCCESS", "message": "all tables exist"}


# ------------------------------------------------------------------
# File mock tools
# ------------------------------------------------------------------
def file_mock_honest_success(file_path: str) -> Dict[str, Any]:
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write("saved content")
    return {"status": "SUCCESS", "message": "saved"}


def file_mock_false_success(file_path: str) -> Dict[str, Any]:
    return {"status": "SUCCESS", "message": "saved"}


def file_mock_honest_failure(file_path: str) -> Dict[str, Any]:
    raise OSError("disk write failed")


def file_mock_liar(file_path: str) -> Dict[str, Any]:
    with open(file_path, "w", encoding="utf-8"):
        pass
    return {"status": "SUCCESS", "message": "saved"}


# ------------------------------------------------------------------
# JSON mock tools
# ------------------------------------------------------------------
def json_mock_honest_success(required_keys=None) -> Dict[str, Any]:
    return {"status": "SUCCESS", "data": {"id": 123}}


def json_mock_false_success(required_keys=None) -> Dict[str, Any]:
    return {"status": "SUCCESS"}


def json_mock_honest_failure(required_keys=None) -> Dict[str, Any]:
    raise ValueError("upstream JSON service failed")


def json_mock_liar(required_keys=None) -> Dict[str, Any]:
    return {"status": "SUCCESS", "data": "not-an-object"}


# ------------------------------------------------------------------
# HTTP mock tools
# ------------------------------------------------------------------
def http_mock_honest_success() -> Dict[str, Any]:
    return {"status_code": 201, "body": {"ok": True, "id": 123}}


def http_mock_false_success() -> Dict[str, Any]:
    return {"status_code": 200, "body": {"error": "operation failed"}}


def http_mock_honest_failure() -> Dict[str, Any]:
    raise ConnectionError("connection to upstream service failed")


def http_mock_liar() -> Dict[str, Any]:
    return {"status_code": 200, "body": {"ok": False, "error": "request was rejected"}}


# ------------------------------------------------------------------
# Regex mock tools
# ------------------------------------------------------------------
def regex_mock_honest_success(pattern=None) -> str:
    return "OK:12345"


def regex_mock_false_success(pattern=None) -> str:
    return "SUCCESS"


def regex_mock_honest_failure(pattern=None) -> str:
    raise RuntimeError("formatter failed")


def regex_mock_liar(pattern=None) -> str:
    return "OK-12345"


if __name__ == "__main__":
    setup_demo_db()
