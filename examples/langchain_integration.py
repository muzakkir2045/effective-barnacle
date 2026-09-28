"""
Step 6 - Real framework integration (LangChain)
-------------------------------------------------

Wraps LangChain's actual QuerySQLDataBaseTool with @verify_action.
This is a REAL LangChain tool, not one of our mocks. It is called
directly with args we control (same as honest_tool/phantom_tool in
test_harness.py) - we are not using an LLM to generate SQL here,
we are testing whether our verifier works on a tool we didn't write.

verify_sql_mutation and snapshot_sql_state are used completely
UNMODIFIED from universal_verifiers.py. No verifier code changed to
make this work - only a thin adapter around the LangChain call.

FINDING (see probe output / README below): LangChain's tool.run()
returns db.run_no_throw(), which:
  1. Returns '' for both a successful UPDATE and a false-success
     UPDATE (0 rows matched) - IDENTICAL output either way. There is
     no rows-affected count exposed at all. This is exactly the
     "self-report gives no usable signal" problem the whole project
     is built around - not a hypothetical, this is what a widely-used
     real tool actually does.
  2. Never raises on a DB error (missing table, bad SQL). It returns
     a string starting with "Error: ...". Our engine's crash-handling
     depends on the tool raising, so the adapter below re-raises when
     it sees that prefix - this is the one piece of translation code
     needed to plug a real tool in, and it lives in the adapter, not
     in the verifier or the engine.

Run:
    python langchain_integration.py
    python langchain_integration.py -v
"""
import argparse
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, List

warnings.filterwarnings("ignore")  # LangChain deprecation noise, not relevant here

from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool

from agent_reliability import registry, verify_action, FailureType as F, verify_sql_mutation, snapshot_sql_state

# ------------------------------------------------------------------
# Registration - identical call as every other harness in this project
# ------------------------------------------------------------------
registry.register("sql_db_query_mutation", verify_sql_mutation, snapshot=snapshot_sql_state)


BASE_SEED = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com');
INSERT INTO customers VALUES (2, 'Bob',   'bob@example.com');
INSERT INTO customers VALUES (3, 'Carol', 'carol@example.com');
"""


# ------------------------------------------------------------------
# Adapter: the ONLY new code. Exposes query/db_path as kwargs (so our
# existing snapshot_sql_state/verify_sql_mutation can read them, same
# as they do for our own mock tools) and translates LangChain's
# swallowed-error string back into a real exception.
# ------------------------------------------------------------------
@verify_action("sql_db_query_mutation")
def langchain_sql_tool(query: str, db_path: str):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    tool = QuerySQLDatabaseTool(db=db)
    try:
        result = tool.run(query)
        if isinstance(result, str) and result.startswith("Error:"):
            raise RuntimeError(result)
        return result
    finally:
        # SQLAlchemy keeps the sqlite file handle open via its connection
        # pool. On Windows this blocks the tempdir cleanup with a
        # PermissionError (WinError 32) unless we dispose the engine here.
        db._engine.dispose()


# ------------------------------------------------------------------
# Scenario model (same shape as test_harness.py / adversarial_harness.py)
# ------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    query: str
    expect_verified: bool
    expect_failure: str
    seed_sql: str = ""   # extra setup applied on top of BASE_SEED, for this scenario only


SCENARIOS: List[Scenario] = [
    # --- original 4 forced scenarios -------------------------------------
    Scenario("honest_success",
             "UPDATE customers SET email = 'alice@newmail.com' WHERE id = 1",
             True, F.NONE),
    Scenario("false_success",
             "UPDATE customers SET email = 'ghost@nowhere.com' WHERE id = 999",
             False, F.FALSE_SUCCESS),
    Scenario("honest_failure",
             "UPDATE customres SET email = 'x' WHERE id = 1",   # typo'd table
             False, F.SCHEMA_ERROR),
    Scenario("hallucinated_table",
             "UPDATE user_accounts SET email = 'x' WHERE id = 1",
             False, F.SCHEMA_ERROR),

    # --- 6 additional scenarios against the REAL tool ---------------------
    Scenario("multi_row_update",
             "UPDATE customers SET email = 'team@example.com' WHERE id IN (1, 2)",
             True, F.NONE),
    Scenario("insert_success",
             "INSERT INTO customers VALUES (4, 'Dave', 'dave@example.com')",
             True, F.NONE),
    Scenario("delete_success",
             "DELETE FROM customers WHERE id = 2",
             True, F.NONE),
    Scenario("delete_nothing",
             "DELETE FROM customers WHERE id = 999",
             False, F.FALSE_SUCCESS),
    Scenario("noop_idempotent",
             "UPDATE customers SET email = 'alice@newmail.com' WHERE id = 1",
             True, F.NONE,
             seed_sql="UPDATE customers SET email='alice@newmail.com' WHERE id=1;"),
    Scenario("constraint_violation",
             "INSERT INTO customers VALUES (1, 'Dup', 'dup@example.com')",   # id=1 already exists
             False, F.CRASH),
]


def label(verified: bool, failure: str) -> str:
    return f"{'verified' if verified else 'rejected'}/{failure}"


def run_one(sc: Scenario, tmpdir: str):
    db = os.path.join(tmpdir, f"{sc.name}.db")
    import sqlite3
    conn = sqlite3.connect(db)
    conn.executescript(BASE_SEED + sc.seed_sql)
    conn.close()
    return langchain_sql_tool(query=sc.query, db_path=db)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    opts = ap.parse_args()

    print(f"\n{'#':>2}  {'RESULT':<6}  {'SCENARIO':<20}  {'EXPECTED':<24}  ACTUAL")
    print("-" * 78)

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, sc in enumerate(SCENARIOS, 1):
            expected = label(sc.expect_verified, sc.expect_failure)
            result = run_one(sc, tmp)
            v = result["verification"]
            actual = label(v["verified"], v["failure_type"])
            ok = actual == expected

            print(f"{i:>2}  {'PASS' if ok else 'FAIL':<6}  {sc.name:<20}  {expected:<24}  {actual}")
            if opts.verbose or not ok:
                print(f"      tool_result: {result['tool_result']!r}")
                print(f"      reasoning:   {v['reasoning']}")
            if not ok:
                failures.append(sc.name)

    print("-" * 78)
    total = len(SCENARIOS)
    print(f"{total - len(failures)}/{total} passed against the REAL LangChain SQL tool"
          + (f"  |  FAILED: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
