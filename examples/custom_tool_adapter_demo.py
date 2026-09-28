"""
Demo - integrating a tool that uses NONE of our naming conventions
---------------------------------------------------------------------

Simulates handing the SDK to a user whose tool:
  - takes kwargs named nothing like ours ("sql_text", "database_file"
    instead of "query", "db_path")
  - returns a dict shaped nothing like ours ("http_status", "payload"
    instead of "status_code", "body")
  - must NOT have its return shape changed by wrapping it (some other
    part of their codebase calls this function directly)

Shows the two features that remove the need for a hand-written adapter
in the common case:
  1. field_map / output_field_map on registry.register() - translate
     names for the verifier without touching the tool's real call/return.
  2. wrap=False on @verify_action - the tool's return value is passed
     through completely unchanged; the verdict is delivered via a
     callback and via get_verdict()/get_last_verdict() instead.

No verifier code changes for either of these - same verify_sql_mutation
and verify_http_status used everywhere else in this project.

Run:
    python custom_tool_adapter_demo.py
"""
import os
import sqlite3
import sys
import tempfile

from agent_reliability import (
    registry,
    verify_action,
    get_last_verdict,
    verify_sql_mutation,
    snapshot_sql_state,
    verify_http_status,
)


# ------------------------------------------------------------------
# A SQL tool with completely different kwarg names than our verifier expects.
# ------------------------------------------------------------------
registry.register(
    "their_sql_tool",
    verify_sql_mutation,
    snapshot=snapshot_sql_state,
    field_map={"query": "sql_text", "db_path": "database_file"},   # our_name -> their_name
)

collected_verdicts = []


@verify_action("their_sql_tool", wrap=False, on_verdict=collected_verdicts.append)
def their_sql_tool(sql_text: str, database_file: str):
    """Their function, their names. Return shape is untouched by us."""
    conn = sqlite3.connect(database_file)
    try:
        cur = conn.execute(sql_text)
        conn.commit()
        return {"outcome": "done", "affected": cur.rowcount}   # their own shape, not ours
    finally:
        conn.close()


# ------------------------------------------------------------------
# An HTTP-style tool with completely different output field names.
# ------------------------------------------------------------------
registry.register(
    "their_http_tool",
    verify_http_status,
    output_field_map={"status_code": "http_status", "body": "payload"},  # our_name -> their_name
)


@verify_action("their_http_tool", wrap=False)
def their_http_tool(succeed: bool):
    """Simulates their API client's own response shape."""
    if succeed:
        return {"http_status": 200, "payload": {"ok": True}}
    return {"http_status": 200, "payload": {"ok": False, "error": "rejected"}}  # false success


def main() -> int:
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "demo.db")
        conn = sqlite3.connect(db)
        conn.executescript("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT); "
                            "INSERT INTO t VALUES (1, 'old');")
        conn.close()

        # --- honest SQL update -----------------------------------------
        result = their_sql_tool(sql_text="UPDATE t SET v='new' WHERE id=1", database_file=db)
        print("their_sql_tool honest  -> return value unchanged:", result)
        verdict = get_last_verdict()
        print("                          verdict:", verdict["verified"], verdict["failure_type"])
        if not verdict["verified"]:
            failures.append("sql_honest")

        # --- false success: WHERE matches nothing -----------------------
        result = their_sql_tool(sql_text="UPDATE t SET v='ghost' WHERE id=999", database_file=db)
        print("their_sql_tool liar    -> return value unchanged:", result)
        verdict = get_last_verdict()
        print("                          verdict:", verdict["verified"], verdict["failure_type"])
        if verdict["verified"] or verdict["failure_type"] != "false_success":
            failures.append("sql_liar")

    # --- HTTP honest success ---------------------------------------------
    result = their_http_tool(succeed=True)
    print("their_http_tool honest -> return value unchanged:", result)
    if "__verification__" not in result:
        failures.append("http_honest_no_verdict_key")
    elif not result["__verification__"]["verified"]:
        failures.append("http_honest")

    # --- HTTP false success (200 but error body) --------------------------
    result = their_http_tool(succeed=False)
    print("their_http_tool liar   -> return value unchanged:", result)
    v = result.get("__verification__", {})
    if v.get("verified") or v.get("failure_type") != "false_success":
        failures.append("http_liar")

    print(f"\ncollected_verdicts callback fired {len(collected_verdicts)} time(s) for the SQL tool.")

    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
        return 1
    print("\nAll cases behaved as expected - no adapter function was written for either tool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
