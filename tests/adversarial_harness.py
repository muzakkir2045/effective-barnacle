"""
Step 4 - Adversarial verifier harness
--------------------------------------

This harness tests failure modes that ordinary happy-path tests
do not cover.

Covered cases:

1. Async tool succeeds and state changes.
2. Async tool claims success but changes nothing.
3. Async tool raises an exception.
4. Tool returns an error string without raising.
5. Tool returns {"isError": True} without changing state.
6. Tool returns {"isError": True} even though the requested
   database mutation actually happened.
7. Tool partially performs a multi-row update.
8. Tool performs a mutation but reports no row count.
9. Tool changes nothing and reports no row count.

The existing 40-test regression harness is intentionally left
untouched. This file is an additional adversarial layer.

Run:

    python adversarial_harness.py

Verbose:

    python adversarial_harness.py -v

List scenarios:

    python adversarial_harness.py --list

Run selected scenarios:

    python adversarial_harness.py --only async_success partial_success
"""

import argparse
import asyncio
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, List

from agent_reliability import (
    registry,
    verify_action,
    FailureType as F,

    verify_sql_mutation,
    snapshot_sql_state,
)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------
#
# The registry is process-local. This is safe because this harness
# is intended to be run as its own process.
#
registry.register(
    "sql_db_query_mutation",
    verify_sql_mutation,
    snapshot=snapshot_sql_state,
)


# ------------------------------------------------------------------
# Database seed
# ------------------------------------------------------------------

BASE_SEED = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT
);

INSERT INTO customers VALUES
    (1, 'Alice', 'alice@example.com');

INSERT INTO customers VALUES
    (2, 'Bob', 'bob@example.com');

INSERT INTO customers VALUES
    (3, 'Carol', 'carol@example.com');
"""


# ------------------------------------------------------------------
# Shared SQL helper
# ------------------------------------------------------------------

def _run_sql(query: str, db_path: str) -> int:
    """
    Execute SQL normally and return SQLite's rowcount.
    """
    conn = sqlite3.connect(db_path)

    try:
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


# ------------------------------------------------------------------
# 1. ASYNC HONEST TOOL
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
async def async_honest_tool(query: str, db_path: str):
    """
    Real async tool.

    It performs the mutation and reports the actual row count.
    """
    await asyncio.sleep(0)
    return {
        "status": "SUCCESS",
        "rows_affected_by_tool": _run_sql(query, db_path),
    }


# ------------------------------------------------------------------
# 2. ASYNC PHANTOM TOOL
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
async def async_phantom_tool(query: str, db_path: str):
    """
    Async tool that claims success but never touches the DB.
    """
    await asyncio.sleep(0)

    return {
        "status": "SUCCESS",
        "rows_affected_by_tool": 1,
    }


# ------------------------------------------------------------------
# 3. ASYNC CRASHING TOOL
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
async def async_crashing_tool(query: str, db_path: str):
    """
    Async tool that raises an actual SQLite error.
    """
    await asyncio.sleep(0)

    return _run_sql(
        "UPDATE table_that_does_not_exist "
        "SET email = 'x' WHERE id = 1",
        db_path,
    )


# ------------------------------------------------------------------
# 4. ERROR RETURNED AS A STRING
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
def returned_error_string_tool(query: str, db_path: str):
    """
    Tool does not raise an exception.

    Instead it returns an error-like string.

    The verifier must not treat the string itself as authoritative.
    It must inspect the actual database state.
    """
    return "ERROR: database operation failed"


# ------------------------------------------------------------------
# 5. MCP-STYLE isError WITHOUT STATE CHANGE
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
def iserror_phantom_tool(query: str, db_path: str):
    """
    Mimics a tool protocol response such as:

        {
            "isError": True,
            "message": "operation failed"
        }

    No database mutation actually occurs.
    """
    return {
        "isError": True,
        "message": "operation failed",
    }


# ------------------------------------------------------------------
# 6. isError=True BUT STATE ACTUALLY CHANGED
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
def iserror_but_mutation_succeeded(query: str, db_path: str):
    """
    The tool reports an error flag but actually performs the requested
    mutation.

    The independent state verifier should trust observed state rather
    than treating isError as proof that the mutation failed.
    """
    affected = _run_sql(query, db_path)

    return {
        "isError": True,
        "message": "tool reported an error",
        "rows_affected_by_tool": affected,
    }


# ------------------------------------------------------------------
# 7. PARTIAL SUCCESS
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
def partial_update_tool(query: str, db_path: str):
    """
    The requested query targets two rows:

        WHERE id IN (1, 2)

    But the tool only updates one of them.

    It then falsely reports that both rows were affected.
    """
    _run_sql(
        "UPDATE customers "
        "SET email = 'partial@example.com' "
        "WHERE id = 1",
        db_path,
    )

    return {
        "status": "SUCCESS",
        "rows_affected_by_tool": 2,
    }


# ------------------------------------------------------------------
# 8. SUCCESS WITHOUT A REPORTED COUNT
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
def no_count_success_tool(query: str, db_path: str):
    """
    Tool performs the mutation but deliberately does not provide
    rows_affected / rowcount.
    """
    _run_sql(query, db_path)

    return {
        "status": "SUCCESS",
    }


# ------------------------------------------------------------------
# 9. NO COUNT + NO MUTATION
# ------------------------------------------------------------------

@verify_action("sql_db_query_mutation")
def no_count_phantom_tool(query: str, db_path: str):
    """
    Tool reports success but provides no row count and does not
    modify the database.
    """
    return {
        "status": "SUCCESS",
    }


# ------------------------------------------------------------------
# Scenario model
# ------------------------------------------------------------------

@dataclass
class Scenario:
    name: str
    run: Callable[[str], dict]
    expect_verified: bool
    expect_failure: str
    seed_sql: str = ""


# ------------------------------------------------------------------
# Async helper
# ------------------------------------------------------------------

def _run_async(coro):
    """
    Run one async tool call from the synchronous scenario runner.
    """
    return asyncio.run(coro)


# ------------------------------------------------------------------
# Scenario builders
# ------------------------------------------------------------------

def build_scenarios() -> List[Scenario]:
    return [

        # ----------------------------------------------------------
        # ASYNC TOOL TESTS
        # ----------------------------------------------------------

        Scenario(
            name="async_success",
            run=lambda db: _run_async(
                async_honest_tool(
                    query=(
                        "UPDATE customers "
                        "SET email = 'async@example.com' "
                        "WHERE id = 1"
                    ),
                    db_path=db,
                )
            ),
            expect_verified=True,
            expect_failure=F.NONE,
        ),

        Scenario(
            name="async_false_success",
            run=lambda db: _run_async(
                async_phantom_tool(
                    query=(
                        "UPDATE customers "
                        "SET email = 'ghost@example.com' "
                        "WHERE id = 1"
                    ),
                    db_path=db,
                )
            ),
            expect_verified=False,
            expect_failure=F.FALSE_SUCCESS,
        ),

        Scenario(
            name="async_crash",
            run=lambda db: _run_async(
                async_crashing_tool(
                    query=(
                        "UPDATE customers "
                        "SET email = 'x@example.com' "
                        "WHERE id = 1"
                    ),
                    db_path=db,
                )
            ),
            expect_verified=False,
            expect_failure=F.SCHEMA_ERROR,
        ),

        # ----------------------------------------------------------
        # ERROR-STYLE OUTPUTS
        # ----------------------------------------------------------

        Scenario(
            name="returned_error_string",
            run=lambda db: returned_error_string_tool(
                query=(
                    "UPDATE customers "
                    "SET email = 'error-string@example.com' "
                    "WHERE id = 1"
                ),
                db_path=db,
            ),
            expect_verified=False,
            expect_failure=F.FALSE_SUCCESS,
        ),

        Scenario(
            name="iserror_without_mutation",
            run=lambda db: iserror_phantom_tool(
                query=(
                    "UPDATE customers "
                    "SET email = 'iserror@example.com' "
                    "WHERE id = 1"
                ),
                db_path=db,
            ),
            expect_verified=False,
            expect_failure=F.FALSE_SUCCESS,
        ),

        Scenario(
            name="iserror_but_state_changed",
            run=lambda db: iserror_but_mutation_succeeded(
                query=(
                    "UPDATE customers "
                    "SET email = 'actually-changed@example.com' "
                    "WHERE id = 1"
                ),
                db_path=db,
            ),
            expect_verified=True,
            expect_failure=F.NONE,
        ),

        # ----------------------------------------------------------
        # PARTIAL SUCCESS
        # ----------------------------------------------------------

        Scenario(
            name="partial_success",
            run=lambda db: partial_update_tool(
                query=(
                    "UPDATE customers "
                    "SET email = 'partial@example.com' "
                    "WHERE id IN (1, 2)"
                ),
                db_path=db,
            ),
            expect_verified=False,
            expect_failure=F.PARTIAL_SUCCESS,
        ),

        # ----------------------------------------------------------
        # NO REPORTED COUNT
        # ----------------------------------------------------------

        Scenario(
            name="no_count_success",
            run=lambda db: no_count_success_tool(
                query=(
                    "UPDATE customers "
                    "SET email = 'nocount@example.com' "
                    "WHERE id = 1"
                ),
                db_path=db,
            ),
            expect_verified=True,
            expect_failure=F.NONE,
        ),

        Scenario(
            name="no_count_false_success",
            run=lambda db: no_count_phantom_tool(
                query=(
                    "UPDATE customers "
                    "SET email = 'never-written@example.com' "
                    "WHERE id = 1"
                ),
                db_path=db,
            ),
            expect_verified=False,
            expect_failure=F.FALSE_SUCCESS,
        ),
    ]


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def label(verified: bool, failure: str) -> str:
    return (
        f"{'verified' if verified else 'rejected'}"
        f"/{failure}"
    )


def run_one(sc: Scenario, tmpdir: str):
    db = os.path.join(
        tmpdir,
        f"{sc.name}.db"
    )

    conn = sqlite3.connect(db)

    try:
        conn.executescript(
            BASE_SEED + sc.seed_sql
        )
        conn.commit()
    finally:
        conn.close()

    return sc.run(db)["verification"]


# ------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print verifier reasoning for every scenario",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="run only these scenarios",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="list scenario names and exit",
    )

    options = parser.parse_args()

    scenarios = build_scenarios()

    if options.list:
        print(
            "\n".join(
                scenario.name
                for scenario in scenarios
            )
        )
        return 0

    if options.only:
        known = {
            scenario.name
            for scenario in scenarios
        }

        unknown = set(options.only) - known

        if unknown:
            print(
                "Unknown scenario(s): "
                + ", ".join(sorted(unknown))
            )
            return 2

        scenarios = [
            scenario
            for scenario in scenarios
            if scenario.name in options.only
        ]

    if not scenarios:
        print("No scenarios selected.")
        return 2

    width = max(
        len(scenario.name)
        for scenario in scenarios
    )

    print(
        f"\n{'#':>2}  "
        f"{'RESULT':<6}  "
        f"{'SCENARIO':<{width}}  "
        f"{'EXPECTED':<30}  "
        f"ACTUAL"
    )

    print("-" * (width + 82))

    failures = []

    with tempfile.TemporaryDirectory() as tmpdir:

        for index, scenario in enumerate(
            scenarios,
            1,
        ):
            expected = label(
                scenario.expect_verified,
                scenario.expect_failure,
            )

            try:
                verdict = run_one(
                    scenario,
                    tmpdir,
                )

                actual = label(
                    verdict["verified"],
                    verdict["failure_type"],
                )

                ok = actual == expected
                reasoning = verdict["reasoning"]

            except Exception as error:
                actual = (
                    f"ERROR/{type(error).__name__}"
                )
                ok = False
                reasoning = str(error)

            print(
                f"{index:>2}  "
                f"{'PASS' if ok else 'FAIL':<6}  "
                f"{scenario.name:<{width}}  "
                f"{expected:<30}  "
                f"{actual}"
            )

            if options.verbose or not ok:
                print(
                    f"{'':>10}{reasoning}"
                )

            if not ok:
                failures.append(
                    scenario.name
                )

    total = len(scenarios)
    passed = total - len(failures)

    print("-" * (width + 82))

    print(
        f"{passed}/{total} passed"
        + (
            "  |  FAILED: "
            + ", ".join(failures)
            if failures
            else ""
        )
    )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())