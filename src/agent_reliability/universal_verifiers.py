"""
Pure verifier functions. They know nothing about the registry or the engine.
Signature: (args, kwargs, tool_output[, pre_state]) -> Verdict
"""
import hashlib
import os
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Optional

from .verdict import Verdict, FailureType as F


# =====================================================================
# SQL helpers
# =====================================================================
_TABLE_RE = re.compile(
    r'\b(?:UPDATE|INSERT\s+(?:OR\s+\w+\s+)?INTO|REPLACE\s+INTO|DELETE\s+FROM)\s+["`\[]?(\w+)',
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r'^\s*UPDATE\s+(?:OR\s+\w+\s+)?["`\[]?(\w+)["`\]]?\s+SET\s+(.+?)(?:\s+WHERE\s+(.+?))?\s*;?\s*$',
    re.IGNORECASE | re.DOTALL,
)
_DELETE_RE = re.compile(
    r'^\s*DELETE\s+FROM\s+["`\[]?(\w+)["`\]]?(?:\s+WHERE\s+(.+?))?\s*;?\s*$',
    re.IGNORECASE | re.DOTALL,
)
_LIT = r"(?:'(?:[^']|'')*'|-?\d+(?:\.\d+)?|NULL)"
_SET_FULL = re.compile(rf"^\s*\w+\s*=\s*{_LIT}(?:\s*,\s*\w+\s*=\s*{_LIT})*\s*$", re.IGNORECASE)
_SET_ITEM = re.compile(rf"(\w+)\s*=\s*({_LIT})", re.IGNORECASE)


def _query_and_db(args, kwargs):
    query = kwargs.get("query") or (args[0] if args else "")
    db_path = kwargs.get("db_path") or (args[1] if len(args) > 1 else "demo.db")
    return query, db_path


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Read-only connection: verifiers can observe state but never change it."""
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)


def _read_rows(db_path: str, table: str, where: Optional[str] = None) -> list:
    """
    Reads only the rows in scope for the operation being verified:
    WHERE-bounded for UPDATE/DELETE, the whole table only when the
    statement itself has no WHERE (in which case the whole table
    genuinely IS the affected scope, not an inefficiency).
    """
    conn = _connect_ro(db_path)
    try:
        sql = f'SELECT * FROM "{table}"' + (f" WHERE {where}" if where else "")
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _count_rows(db_path: str, table: str, where: Optional[str], extra_conds: List[str] = ()) -> int:
    """O(1)-ish via COUNT(*) - never copies rows, safe on large tables."""
    conds = ([f"({where})"] if where else []) + list(extra_conds)
    sql = f'SELECT COUNT(*) FROM "{table}"' + (" WHERE " + " AND ".join(conds) if conds else "")
    conn = _connect_ro(db_path)
    try:
        return conn.execute(sql).fetchone()[0]
    finally:
        conn.close()


def _parse_mutation(query: str) -> Optional[Dict[str, Any]]:
    m = _UPDATE_RE.match(query)
    if m:
        clause = m.group(2)
        pairs = _SET_ITEM.findall(clause) if _SET_FULL.match(clause) else None
        return {"op": "update", "table": m.group(1), "where": m.group(3), "set": pairs, "parsed": True}
    m = _DELETE_RE.match(query)
    if m:
        return {"op": "delete", "table": m.group(1), "where": m.group(2), "set": None, "parsed": True}
    m = _TABLE_RE.search(query)
    if m:
        op = "insert" if re.match(r"\s*(INSERT|REPLACE)\b", query, re.IGNORECASE) else "other"
        return {"op": op, "table": m.group(1), "where": None, "set": None, "parsed": False}
    return None


def _claimed_rows(tool_output: Any) -> Optional[int]:
    if isinstance(tool_output, dict):
        for key in ("rows_affected_by_tool", "rows_affected", "rowcount"):
            val = tool_output.get(key)
            if isinstance(val, int) and not isinstance(val, bool) and val >= 0:
                return val
    return None


def snapshot_sql_state(args, kwargs):
    """
    Runs BEFORE the tool.

    Scale note: UPDATE/DELETE only ever read the rows the statement's
    own WHERE clause targets (bounded by the operation, not the table
    size). INSERT and any statement we can't parse fall back to a
    single COUNT(*) - never a full row copy - so a multi-million-row
    table costs one aggregate query, not a snapshot of everything.
    """
    query, db_path = _query_and_db(args, kwargs)
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)
    parsed = _parse_mutation(query)
    if parsed is None:
        raise ValueError("no target table found in query")

    table, op, where, set_pairs = parsed["table"], parsed["op"], parsed["where"], parsed["set"]

    if op in ("update", "delete"):
        matched = None
        if parsed["parsed"]:
            try:
                matched = _count_rows(db_path, table, where)
            except Exception:
                matched = None
        pre_rows = _read_rows(db_path, table, where)
        return {"table": table, "op": op, "where": where, "set": set_pairs,
                "matched": matched, "pre_rows": pre_rows, "pre_count": None}

    # insert / unparsed "other": count-only, no row copy regardless of table size
    pre_count = _count_rows(db_path, table, None)
    return {"table": table, "op": op, "where": where, "set": set_pairs,
            "matched": None, "pre_rows": None, "pre_count": pre_count}


def _count_mismatch(op: str, claimed: Optional[int], changed: int, matched: Optional[int]) -> Optional[str]:
    if claimed is None:
        return None
    if op == "update":
        if matched is not None and claimed != matched:
            return f"Self-report mismatch: tool claimed {claimed} row(s), WHERE targets {matched}."
        if matched is None and claimed < changed:
            return f"Self-report mismatch: tool claimed {claimed} row(s), but {changed} actually changed."
        return None
    if claimed != changed:
        return f"Self-report mismatch: tool claimed {claimed} row(s), actual state change is {changed}."
    return None


def _verdict_state_unchanged(db_path: str, pre: Dict[str, Any], claimed: Optional[int]) -> Verdict:
    table, op, matched = pre["table"], pre["op"], pre["matched"]
    claim = "no row count reported" if claimed is None else f"claimed {claimed} row(s)"

    if op != "update":
        return Verdict(False, "receipt-confirmed", "high",
                       f"FALSE SUCCESS: {op.upper()} reported success ({claim}) but '{table}' is unchanged.",
                       F.FALSE_SUCCESS)
    if matched == 0 or (matched is None and claimed == 0):
        return Verdict(False, "receipt-confirmed", "high",
                       f"FALSE SUCCESS: UPDATE reported success ({claim}) but WHERE matched no rows; "
                       f"'{table}' is unchanged.", F.FALSE_SUCCESS)

    pairs = pre["set"]
    if matched is None or pairs is None:
        return Verdict(False, "no-check-possible", "low",
                       f"State of '{table}' unchanged after UPDATE ({claim}); SQL too complex to confirm "
                       f"the requested values are already in place.", F.NO_OP)
    try:
        holding = _count_rows(db_path, table, pre["where"], [f"{c} IS {lit}" for c, lit in pairs])
    except Exception as e:
        return Verdict(False, "no-check-possible", "low",
                       f"State unchanged; post-condition check failed to run: {e}", F.NO_OP)

    if holding == matched:
        if claimed is not None and claimed != matched:
            return Verdict(False, "receipt-confirmed", "high",
                           f"Self-report mismatch: tool claimed {claimed} row(s), WHERE targets {matched}.",
                           F.MISMATCH)
        return Verdict(True, "receipt-confirmed", "high",
                       f"No state change, but all {matched} targeted row(s) already hold the requested "
                       f"values (idempotent no-op).")
    return Verdict(False, "receipt-confirmed", "high",
                   f"FALSE SUCCESS: tool reported success ({claim}) but only {holding} of {matched} targeted "
                   f"row(s) hold the requested values; '{table}' is unchanged.", F.FALSE_SUCCESS)


# =====================================================================
# VERIFIER 1: SQL Mutation (bounded before/after diff; never re-executes)
# =====================================================================
def verify_sql_mutation(args: List[Any], kwargs: Dict[str, Any], tool_output: Any, pre_state: Any = None) -> Verdict:
    _, db_path = _query_and_db(args, kwargs)
    if pre_state is None:
        return Verdict(False, "no-check-possible", "low",
                       "No pre-state snapshot available; cannot verify independently.", F.MISSING_INPUT)

    table = pre_state["table"]
    claimed = _claimed_rows(tool_output)

    # ---- UPDATE / DELETE: diff only the WHERE-bounded rows ----------
    if pre_state["pre_rows"] is not None:
        try:
            post_rows = _read_rows(db_path, table, pre_state["where"])
        except Exception as e:
            return Verdict(False, "no-check-possible", "low",
                           f"Could not read post-state of '{table}': {e}", F.VERIFIER_ERROR)

        pre_c, post_c = Counter(pre_state["pre_rows"]), Counter(post_rows)
        changed = max(sum((post_c - pre_c).values()), sum((pre_c - post_c).values()))

        if changed == 0:
            return _verdict_state_unchanged(db_path, pre_state, claimed)

        if pre_state["op"] == "update" and pre_state["matched"] is not None and pre_state["set"] is not None:
            try:
                holding = _count_rows(db_path, table, pre_state["where"],
                                       [f"{c} IS {lit}" for c, lit in pre_state["set"]])
            except Exception as e:
                return Verdict(False, "no-check-possible", "low",
                               f"State changed, but post-condition check failed to run: {e}", F.VERIFIER_ERROR)
            matched = pre_state["matched"]
            if holding < matched:
                return Verdict(False, "receipt-confirmed", "high",
                               f"PARTIAL SUCCESS: UPDATE changed database state, but only {holding} of "
                               f"{matched} targeted row(s) hold the requested values in '{table}'.",
                               F.PARTIAL_SUCCESS)

        mismatch = _count_mismatch(pre_state["op"], claimed, changed, pre_state["matched"])
        if mismatch:
            return Verdict(False, "receipt-confirmed", "high", mismatch, F.MISMATCH)
        return Verdict(True, "receipt-confirmed", "high",
                       f"State verified: {changed} row(s) changed in '{table}' ({pre_state['op']}).")

    # ---- INSERT / unparsed: COUNT(*) only, no row copy at any table size ----
    try:
        post_count = _count_rows(db_path, table, None)
    except Exception as e:
        return Verdict(False, "no-check-possible", "low",
                       f"Could not read post-state count of '{table}': {e}", F.VERIFIER_ERROR)

    changed = max(post_count - pre_state["pre_count"], 0)
    if changed == 0:
        claim = "no row count reported" if claimed is None else f"claimed {claimed} row(s)"
        return Verdict(False, "receipt-confirmed", "high",
                       f"FALSE SUCCESS: {pre_state['op'].upper()} reported success ({claim}) but row count "
                       f"in '{table}' is unchanged.", F.FALSE_SUCCESS)

    mismatch = _count_mismatch(pre_state["op"], claimed, changed, None)
    if mismatch:
        return Verdict(False, "receipt-confirmed", "high", mismatch, F.MISMATCH)
    return Verdict(True, "receipt-confirmed", "high",
                   f"State verified: {changed} row(s) changed in '{table}' ({pre_state['op']}).")


# =====================================================================
# VERIFIER 2: SQL Schema Check
# =====================================================================
def verify_sql_schema(args: List[Any], kwargs: Dict[str, Any], tool_output: Any) -> Verdict:
    query, db_path = _query_and_db(args, kwargs)
    if not os.path.exists(db_path):
        return Verdict(False, "receipt-confirmed", "high",
                       f"Database path '{db_path}' does not exist.", F.MISSING_INPUT)

    tables = re.findall(r'(?:FROM|UPDATE|INTO)\s+([a-zA-Z_][a-zA-Z0-9_]*)', query, re.IGNORECASE)
    try:
        conn = sqlite3.connect(db_path)
        try:
            existing = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
        finally:
            conn.close()
    except Exception as e:
        return Verdict(False, "no-check-possible", "low", f"Schema check error: {e}", F.VERIFIER_ERROR)

    missing = [t for t in tables if t not in existing]
    if missing:
        return Verdict(False, "receipt-confirmed", "high",
                       f"Referenced non-existent table(s): {', '.join(missing)}.", F.SCHEMA_ERROR)
    return Verdict(True, "receipt-confirmed", "high", "Schema validation passed. All referenced tables exist.")


# =====================================================================
# VERIFIER 3: File Written (fingerprint-based; catches an untouched
# pre-existing file, not just "does it exist")
# =====================================================================
_MAX_HASH_BYTES = 10 * 1024 * 1024  # bigger files: compare size + mtime only, skip the hash


def _file_path(args, kwargs):
    return kwargs.get("file_path") or (args[0] if args else None)


def _file_fingerprint(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {"exists": False}
    st = os.stat(path)
    digest = None
    if st.st_size <= _MAX_HASH_BYTES:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
    return {"exists": True, "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}


def snapshot_file_state(args, kwargs):
    """Runs BEFORE the tool: fingerprint of the target file (or exists=False)."""
    path = _file_path(args, kwargs)
    if not path:
        raise ValueError("no file_path in tool call")
    return _file_fingerprint(path)


def verify_file_written(args: List[Any], kwargs: Dict[str, Any], tool_output: Any, pre_state: Any = None) -> Verdict:
    path = _file_path(args, kwargs)
    if not path:
        return Verdict(False, "no-check-possible", "low", "Missing 'file_path' argument.", F.MISSING_INPUT)

    post = _file_fingerprint(path)
    if not post["exists"]:
        return Verdict(False, "receipt-confirmed", "high",
                       f"FALSE SUCCESS: '{path}' was not created on disk.", F.FALSE_SUCCESS)
    size = post["size"]
    if size == 0:
        return Verdict(False, "receipt-confirmed", "high",
                       f"FALSE SUCCESS: '{path}' exists but is 0 bytes.", F.FALSE_SUCCESS)

    pre_existed = bool(pre_state and pre_state.get("exists"))
    if pre_existed and all(pre_state.get(k) == post.get(k) for k in ("size", "mtime_ns", "sha256")):
        return Verdict(False, "receipt-confirmed", "high",
                       f"FALSE SUCCESS: '{path}' already existed and is untouched "
                       f"(same content and modification time).", F.FALSE_SUCCESS)

    claimed = tool_output.get("bytes_written") if isinstance(tool_output, dict) else None
    if isinstance(claimed, int) and not isinstance(claimed, bool):
        known_new = pre_state is not None and not pre_state.get("exists")
        if (known_new and claimed != size) or (not known_new and claimed > size):
            return Verdict(False, "receipt-confirmed", "high",
                           f"Self-report mismatch: tool claimed {claimed} byte(s) written, "
                           f"file holds {size}.", F.MISMATCH)

    action = "updated" if pre_existed else "created"
    return Verdict(True, "receipt-confirmed", "high",
                   f"File {action} and verified on disk at '{path}' ({size} bytes).")


# =====================================================================
# VERIFIER 4: JSON Payload
# =====================================================================
def verify_json_payload(args: List[Any], kwargs: Dict[str, Any], tool_output: Any) -> Verdict:
    required_keys = kwargs.get("required_keys", [])
    data = tool_output
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (TypeError, ValueError, json.JSONDecodeError) as e:
            return Verdict(False, "receipt-confirmed", "high", f"Output is not valid JSON text: {e}.", F.BAD_OUTPUT)
    if not isinstance(data, dict):
        return Verdict(False, "receipt-confirmed", "high",
                       f"Expected JSON object, got {type(data).__name__}.", F.BAD_OUTPUT)
    missing = [k for k in required_keys if k not in data]
    if missing:
        return Verdict(False, "receipt-confirmed", "high",
                       f"FALSE SUCCESS: payload missing required key(s): {', '.join(missing)}.", F.FALSE_SUCCESS)
    return Verdict(True, "receipt-confirmed", "high", "JSON payload structure and required keys verified.")


# =====================================================================
# VERIFIER 5: HTTP Status + body (a 200 with an error body is NOT success)
# =====================================================================
def _extract_status(tool_output: Any) -> Optional[int]:
    val = None
    if isinstance(tool_output, dict):
        val = tool_output.get("status_code", tool_output.get("status"))
    elif hasattr(tool_output, "status_code"):
        val = tool_output.status_code
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.strip().isdigit():
        return int(val)
    return None


def _extract_body(tool_output: Any) -> Any:
    if isinstance(tool_output, dict):
        return tool_output.get("body", tool_output.get("json"))
    try:
        return tool_output.json()
    except Exception:
        return getattr(tool_output, "text", None)


def _body_error(body: Any) -> Optional[str]:
    """Catches Slack-style ok:false, GraphQL-style errors, generic error/errors fields -
    not just the 'ok'/'error' pair the original verifier checked."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            return None
    if not isinstance(body, dict):
        return None
    if body.get("ok") is False:
        return f"'ok' is false (error: {str(body.get('error'))[:80]})"
    if body.get("success") is False:
        return "'success' is false"
    if body.get("error"):
        return f"'error' field set: {str(body['error'])[:80]}"
    if body.get("errors"):
        return f"'errors' field set: {str(body['errors'])[:80]}"
    return None


def verify_http_status(args: List[Any], kwargs: Dict[str, Any], tool_output: Any) -> Verdict:
    expected = kwargs.get("expected_status", [200, 201, 202, 204])
    if isinstance(expected, int):
        expected = [expected]

    status = _extract_status(tool_output)
    if status is None:
        return Verdict(False, "self-report-only", "low",
                       "Could not extract an HTTP status code from tool output.", F.MISSING_INPUT)
    if status not in expected:
        return Verdict(False, "receipt-confirmed", "high",
                       f"HTTP request failed with status code: {status}.", F.HTTP_ERROR)

    problem = _body_error(_extract_body(tool_output))
    if problem:
        return Verdict(False, "receipt-confirmed", "high",
                       f"FALSE SUCCESS: HTTP {status} but the response body reports failure: {problem}.",
                       F.FALSE_SUCCESS)
    return Verdict(True, "receipt-confirmed", "high",
                   f"HTTP status {status} is an expected success code and the body reports no error.")


# =====================================================================
# VERIFIER 6: Regex Match (whole output must match by default)
# =====================================================================
def verify_regex_match(args: List[Any], kwargs: Dict[str, Any], tool_output: Any) -> Verdict:
    pattern = kwargs.get("pattern")
    if not pattern:
        return Verdict(False, "no-check-possible", "low", "No 'pattern' provided to check against.", F.MISSING_INPUT)
    if tool_output is None:
        return Verdict(False, "receipt-confirmed", "high", "Tool returned no output.", F.BAD_OUTPUT)

    mode = kwargs.get("match_mode", "full")   # "full" (default) or "search"
    try:
        if mode == "search":
            ok = re.search(pattern, str(tool_output)) is not None
        else:
            ok = re.fullmatch(pattern, str(tool_output).strip()) is not None
    except re.error as e:
        return Verdict(False, "no-check-possible", "low", f"Invalid regex pattern r'{pattern}': {e}.", F.VERIFIER_ERROR)

    if ok:
        return Verdict(True, "receipt-confirmed", "high", f"Output matches pattern r'{pattern}'.")
    return Verdict(False, "receipt-confirmed", "high",
                   f"FALSE SUCCESS: output failed pattern r'{pattern}'.", F.FALSE_SUCCESS)
