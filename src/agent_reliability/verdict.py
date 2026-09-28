"""Single shared Verdict model (used by the engine AND the verifiers)."""
from dataclasses import dataclass, asdict
from typing import Any, Dict


class FailureType:
    NONE = "none"
    FALSE_SUCCESS = "false_success"          # tool said OK, real state says no
    NO_OP = "no_op"                          # state unchanged, desired state NOT confirmable
    MISMATCH = "self_report_mismatch"        # tool's claim != observed state
    CRASH = "crash"                          # tool raised an exception
    SCHEMA_ERROR = "schema_error"            # missing table/column
    NO_VERIFIER = "no_verifier"              # nothing registered for this tool
    VERIFIER_ERROR = "verifier_error"        # verifier itself failed
    MISSING_INPUT = "missing_input"          # verifier lacked what it needs
    BAD_OUTPUT = "bad_output"                # malformed / wrong output
    HTTP_ERROR = "http_error"
    PARTIAL_SUCCESS = "partial_success"


@dataclass
class Verdict:
    verified: bool
    basis: str        # "receipt-confirmed" | "self-report-only" | "no-check-possible"
    confidence: str   # "high" | "low" | "unknown"
    reasoning: str
    failure_type: str = FailureType.NONE
    execution_time_ms: float = 0.0
    call_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
