"""V1 verifier regression harness, including Step 3 mock-tool scenarios."""

import argparse
import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from typing import Callable, List

from agent_reliability import registry, verify_action, FailureType as F
from mock_agent_harness import (
    fake_agent_call,
    sql_mock_honest_success, sql_mock_false_success, sql_mock_honest_failure, sql_mock_liar,
    schema_mock_honest_success, schema_mock_false_success, schema_mock_honest_failure, schema_mock_liar,
    file_mock_honest_success, file_mock_false_success, file_mock_honest_failure, file_mock_liar,
    json_mock_honest_success, json_mock_false_success, json_mock_honest_failure, json_mock_liar,
    http_mock_honest_success, http_mock_false_success, http_mock_honest_failure, http_mock_liar,
    regex_mock_honest_success, regex_mock_false_success, regex_mock_honest_failure, regex_mock_liar,
)
from agent_reliability import (
    verify_file_written,
    verify_json_payload,
    verify_http_status,
    verify_regex_match,
    verify_sql_mutation,
    verify_sql_schema,
    snapshot_sql_state,
    snapshot_file_state,
)


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------
registry.register("sql_db_query_mutation", verify_sql_mutation, snapshot=snapshot_sql_state)
registry.register("sql_schema_check", verify_sql_schema)
registry.register("write_file", verify_file_written, snapshot=snapshot_file_state)
registry.register("parse_json", verify_json_payload)
registry.register("make_http_request", verify_http_status)
registry.register("check_format", verify_regex_match)


BASE_SEED = """
CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
INSERT INTO customers VALUES (1, 'Alice', 'alice@example.com');
INSERT INTO customers VALUES (2, 'Bob',   'bob@example.com');
INSERT INTO customers VALUES (3, 'Carol', 'carol@example.com');
"""


# ------------------------------------------------------------------
# Existing SQL tools
# ------------------------------------------------------------------
def _run_sql(query: str, db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(query)
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


@verify_action("sql_db_query_mutation")
def honest_tool(query: str, db_path: str):
    return {"status": "SUCCESS", "rows_affected_by_tool": _run_sql(query, db_path)}


@verify_action("sql_db_query_mutation")
def phantom_tool(query: str, db_path: str):
    return {"status": "SUCCESS", "rows_affected_by_tool": 1}


@verify_action("sql_db_query_mutation")
def inflated_count_tool(query: str, db_path: str):
    return {"status": "SUCCESS", "rows_affected_by_tool": _run_sql(query, db_path) + 2}


@verify_action("unregistered_tool")
def unregistered_tool(query: str, db_path: str):
    return {"status": "SUCCESS", "rows_affected_by_tool": _run_sql(query, db_path)}


# ------------------------------------------------------------------
# Decorate each Step 3 mock so every scenario passes through the real engine.
# ------------------------------------------------------------------
sql_step3_tools = {
    "honest_success": verify_action("sql_db_query_mutation")(sql_mock_honest_success),
    "false_success": verify_action("sql_db_query_mutation")(sql_mock_false_success),
    "honest_failure": verify_action("sql_db_query_mutation")(sql_mock_honest_failure),
    "liar": verify_action("sql_db_query_mutation")(sql_mock_liar),
}

schema_step3_tools = {
    "honest_success": verify_action("sql_schema_check")(schema_mock_honest_success),
    "false_success": verify_action("sql_schema_check")(schema_mock_false_success),
    "honest_failure": verify_action("sql_schema_check")(schema_mock_honest_failure),
    "liar": verify_action("sql_schema_check")(schema_mock_liar),
}

file_step3_tools = {
    "honest_success": verify_action("write_file")(file_mock_honest_success),
    "false_success": verify_action("write_file")(file_mock_false_success),
    "honest_failure": verify_action("write_file")(file_mock_honest_failure),
    "liar": verify_action("write_file")(file_mock_liar),
}

json_step3_tools = {
    "honest_success": verify_action("parse_json")(json_mock_honest_success),
    "false_success": verify_action("parse_json")(json_mock_false_success),
    "honest_failure": verify_action("parse_json")(json_mock_honest_failure),
    "liar": verify_action("parse_json")(json_mock_liar),
}

http_step3_tools = {
    "honest_success": verify_action("make_http_request")(http_mock_honest_success),
    "false_success": verify_action("make_http_request")(http_mock_false_success),
    "honest_failure": verify_action("make_http_request")(http_mock_honest_failure),
    "liar": verify_action("make_http_request")(http_mock_liar),
}

regex_step3_tools = {
    "honest_success": verify_action("check_format")(regex_mock_honest_success),
    "false_success": verify_action("check_format")(regex_mock_false_success),
    "honest_failure": verify_action("check_format")(regex_mock_honest_failure),
    "liar": verify_action("check_format")(regex_mock_liar),
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


def sql_case(name, query, verified, failure, tool=honest_tool, seed_sql="") -> Scenario:
    return Scenario(name, lambda db: tool(query=query, db_path=db), verified, failure, seed_sql)


def step3_case(name, tool, verified, failure, *args, seed_sql="", **kwargs) -> Scenario:
    return Scenario(name, lambda db: tool(*args, **kwargs), verified, failure, seed_sql)


def agent_case(name, verified, failure) -> Scenario:
    sql = fake_agent_call(name)["generated_sql"]
    return sql_case(name, sql, verified, failure)


def build_scenarios() -> List[Scenario]:
    scenarios = [
        # Existing SQL regression scenarios.
        agent_case("honest_success", True, F.NONE),
        agent_case("false_success", False, F.FALSE_SUCCESS),
        agent_case("honest_failure", False, F.SCHEMA_ERROR),
        agent_case("hallucinated_table", False, F.SCHEMA_ERROR),
        sql_case("noop_idempotent", "UPDATE customers SET email = 'alice@newmail.com' WHERE id = 1",
                 True, F.NONE, seed_sql="UPDATE customers SET email='alice@newmail.com' WHERE id=1;"),
        sql_case("noop_unverifiable", "UPDATE customers SET email = lower(email) WHERE id = 1", False, F.NO_OP),
        sql_case("partial_already_set", "UPDATE customers SET email = 'team@example.com' WHERE id IN (1, 2)",
                 True, F.NONE, seed_sql="UPDATE customers SET email='team@example.com' WHERE id=2;"),
        sql_case("multi_row_update", "UPDATE customers SET email = 'team@example.com' WHERE id IN (1, 2)", True, F.NONE),
        sql_case("update_all_rows", "UPDATE customers SET name = 'X'", True, F.NONE),
        sql_case("insert_success", "INSERT INTO customers VALUES (4, 'Dave', 'dave@example.com')", True, F.NONE),
        sql_case("delete_success", "DELETE FROM customers WHERE id = 2", True, F.NONE),
        sql_case("delete_nothing", "DELETE FROM customers WHERE id = 999", False, F.FALSE_SUCCESS),
        sql_case("constraint_violation", "INSERT INTO customers VALUES (1, 'Dup', 'dup@example.com')", False, F.CRASH),
        sql_case("phantom_tool", "UPDATE customers SET email = 'ghost@example.com' WHERE id = 1",
                 False, F.FALSE_SUCCESS, tool=phantom_tool),
        sql_case("inflated_count", "UPDATE customers SET email = 'new@example.com' WHERE id = 1",
                 False, F.MISMATCH, tool=inflated_count_tool),
        sql_case("no_verifier_registered", "UPDATE customers SET email = 'z@example.com' WHERE id = 1",
                 False, F.NO_VERIFIER, tool=unregistered_tool),
    ]

    # SQL mutation: honest success / false success / honest failure / liar.
    scenarios += [
        sql_case("step3_sql_honest_success", "UPDATE customers SET email='step3@example.com' WHERE id=1",
                 True, F.NONE, tool=sql_step3_tools["honest_success"]),
        sql_case("step3_sql_false_success", "UPDATE customers SET email='ghost@example.com' WHERE id=999",
                 False, F.FALSE_SUCCESS, tool=sql_step3_tools["false_success"]),
        sql_case("step3_sql_honest_failure", "UPDATE customers SET email='x' WHERE id=1",
                 False, F.SCHEMA_ERROR, tool=sql_step3_tools["honest_failure"]),
        sql_case("step3_sql_liar", "UPDATE customers SET email='liar@example.com' WHERE id=1",
                 False, F.FALSE_SUCCESS, tool=sql_step3_tools["liar"]),
    ]

    # SQL schema: the liar/false-success cases use a missing table in the query.
    scenarios += [
        Scenario("step3_schema_honest_success",
                  lambda db: schema_step3_tools["honest_success"]("SELECT * FROM customers", db), True, F.NONE),
        Scenario("step3_schema_false_success",
                  lambda db: schema_step3_tools["false_success"]("SELECT * FROM missing_table", db), False, F.SCHEMA_ERROR),
        Scenario("step3_schema_honest_failure",
                  lambda db: schema_step3_tools["honest_failure"]("SELECT * FROM missing_table", db), False, F.SCHEMA_ERROR),
        Scenario("step3_schema_liar",
                  lambda db: schema_step3_tools["liar"]("UPDATE missing_table SET value=1", db), False, F.SCHEMA_ERROR),
    ]

    # File verifier: the two bad-success forms are "nothing written" and "empty file".
    scenarios += [
        Scenario("step3_file_honest_success", lambda db: file_step3_tools["honest_success"](
            os.path.join(os.path.dirname(db), "honest.txt")), True, F.NONE),
        Scenario("step3_file_false_success", lambda db: file_step3_tools["false_success"](
            os.path.join(os.path.dirname(db), "missing.txt")), False, F.FALSE_SUCCESS),
        Scenario("step3_file_honest_failure", lambda db: file_step3_tools["honest_failure"](
            os.path.join(os.path.dirname(db), "failure.txt")), False, F.CRASH),
        Scenario("step3_file_liar", lambda db: file_step3_tools["liar"](
            os.path.join(os.path.dirname(db), "empty.txt")), False, F.FALSE_SUCCESS),
    ]

    # JSON verifier: it checks required keys. The liar has the key but the wrong payload type,
    # which deliberately documents the current verifier boundary: it passes because the key exists.
    scenarios += [
        Scenario("step3_json_honest_success", lambda db: json_step3_tools["honest_success"](required_keys=["status", "data"]
        ), True, F.NONE),
        Scenario("step3_json_false_success", lambda db: json_step3_tools["false_success"](required_keys=["status", "data"]
        ), False, F.FALSE_SUCCESS),
        Scenario("step3_json_honest_failure", lambda db: json_step3_tools["honest_failure"](required_keys=["status", "data"]
        ), False, F.CRASH),
        Scenario("step3_json_liar", lambda db: json_step3_tools["liar"](required_keys=["status", "data"]
        ), True, F.NONE),
    ]

    # HTTP verifier: both bad-success variants return an HTTP 200 but expose an application error.
    scenarios += [
        Scenario("step3_http_honest_success", lambda db: http_step3_tools["honest_success"](), True, F.NONE),
        Scenario("step3_http_false_success", lambda db: http_step3_tools["false_success"](), False, F.FALSE_SUCCESS),
        Scenario("step3_http_honest_failure", lambda db: http_step3_tools["honest_failure"](), False, F.CRASH),
        Scenario("step3_http_liar", lambda db: http_step3_tools["liar"](), False, F.FALSE_SUCCESS),
    ]

    # Regex verifier: required output format is OK:<digits>.
    regex_kwargs = {"pattern": r"^OK:\d+$"}
    scenarios += [
        Scenario("step3_regex_honest_success", lambda db: regex_step3_tools["honest_success"](**regex_kwargs), True, F.NONE),
        Scenario("step3_regex_false_success", lambda db: regex_step3_tools["false_success"](**regex_kwargs), False, F.FALSE_SUCCESS),
        Scenario("step3_regex_honest_failure", lambda db: regex_step3_tools["honest_failure"](**regex_kwargs), False, F.CRASH),
        Scenario("step3_regex_liar", lambda db: regex_step3_tools["liar"](**regex_kwargs), False, F.FALSE_SUCCESS),
    ]

    return scenarios


def label(verified: bool, failure: str) -> str:
    return f"{'verified' if verified else 'rejected'}/{failure}"


def run_one(sc: Scenario, tmpdir: str):
    db = os.path.join(tmpdir, f"{sc.name}.db")
    conn = sqlite3.connect(db)
    try:
        conn.executescript(BASE_SEED + sc.seed_sql)
    finally:
        conn.close()
    return sc.run(db)["verification"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="print verifier reasoning for every scenario")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="run only these scenarios")
    ap.add_argument("--list", action="store_true", help="list scenario names and exit")
    opts = ap.parse_args()

    scenarios = build_scenarios()
    if opts.list:
        print("\n".join(s.name for s in scenarios))
        return 0
    if opts.only:
        unknown = set(opts.only) - {s.name for s in scenarios}
        if unknown:
            print(f"Unknown scenario(s): {', '.join(sorted(unknown))}")
            return 2
        scenarios = [s for s in scenarios if s.name in opts.only]

    width = max(len(s.name) for s in scenarios)
    print(f"\n{'#':>2}  {'RESULT':<6}  {'SCENARIO':<{width}}  {'EXPECTED':<28}  ACTUAL")
    print("-" * (width + 78))

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, sc in enumerate(scenarios, 1):
            expected = label(sc.expect_verified, sc.expect_failure)
            try:
                v = run_one(sc, tmp)
                actual = label(v["verified"], v["failure_type"])
                ok = actual == expected
                reasoning = v["reasoning"]
            except Exception as e:
                actual, ok, reasoning = f"ERROR/{type(e).__name__}", False, str(e)

            print(f"{i:>2}  {'PASS' if ok else 'FAIL':<6}  {sc.name:<{width}}  {expected:<28}  {actual}")
            if opts.verbose or not ok:
                print(f"{'':>10}{reasoning}")
            if not ok:
                failures.append(sc.name)

    total = len(scenarios)
    print("-" * (width + 78))
    print(f"{total - len(failures)}/{total} passed" + (f"  |  FAILED: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
