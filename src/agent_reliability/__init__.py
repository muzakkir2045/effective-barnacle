"""
agent_reliability
==================

A reliability layer for AI agent tools: independently verify that a tool's
claimed success actually happened, by checking real system state (database,
file, HTTP response) rather than trusting the tool's own self-report.

Quickstart
----------
    from agent_reliability import verify_action, registry, verify_sql_mutation, snapshot_sql_state

    registry.register("run_sql", verify_sql_mutation, snapshot=snapshot_sql_state)

    @verify_action("run_sql")
    def run_sql(query: str, db_path: str):
        ...  # your existing tool, unchanged

    result = run_sql(query="UPDATE ...", db_path="app.db")
    result["verification"]["verified"]       # True/False - is it actually true?
    result["verification"]["failure_type"]    # "false_success", "self_report_mismatch", ...

If your tool's kwargs or return value don't already match a verifier's
expected names, see `registry.register(..., field_map=..., output_field_map=...)`
and `verify_action(..., wrap=False)` to avoid writing a custom adapter.
"""

from .verdict import Verdict, FailureType
from .verify_agent import (
    VerifierRegistry,
    registry,
    verify_action,
    get_verdict,
    get_last_verdict,
)
from .universal_verifiers import (
    verify_sql_mutation,
    verify_sql_schema,
    verify_file_written,
    verify_json_payload,
    verify_http_status,
    verify_regex_match,
    snapshot_sql_state,
    snapshot_file_state,
)

__version__ = "0.1.0"

__all__ = [
    "Verdict",
    "FailureType",
    "VerifierRegistry",
    "registry",
    "verify_action",
    "get_verdict",
    "get_last_verdict",
    "verify_sql_mutation",
    "verify_sql_schema",
    "verify_file_written",
    "verify_json_payload",
    "verify_http_status",
    "verify_regex_match",
    "snapshot_sql_state",
    "snapshot_file_state",
]
