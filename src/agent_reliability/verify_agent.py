import functools
import inspect
import time
import uuid
from collections import OrderedDict
from typing import Callable, Dict, Optional

from .verdict import Verdict, FailureType


# ---------------------------------------------------------
# VERIFIER REGISTRY
# ---------------------------------------------------------
class VerifierRegistry:
    def __init__(self):
        self._verifiers: Dict[str, Callable] = {}
        self._snapshots: Dict[str, Callable] = {}
        self._field_maps: Dict[str, Dict[str, str]] = {}
        self._output_field_maps: Dict[str, Dict[str, str]] = {}

    def register(self, tool_name: str, verifier_fn: Callable = None, snapshot: Callable = None,
                 field_map: Dict[str, str] = None, output_field_map: Dict[str, str] = None):
        """
        Manual:    registry.register("tool", verifier_fn, snapshot=snap_fn)
        Decorator: @registry.register("tool", snapshot=snap_fn)
        `snapshot(args, kwargs)` runs BEFORE the tool and its result is passed
        to the verifier as `pre_state` (if the verifier accepts that param).

        `field_map` lets a tool use its own kwarg names instead of matching a
        verifier's expected names exactly. Keys are the names the VERIFIER
        expects (e.g. "query", "db_path", "file_path"); values are the names
        the TOOL actually uses in its own call signature, e.g.:
            field_map={"query": "sql_text", "db_path": "database_file"}
        The tool is still called with its own original kwargs unchanged -
        only the copy handed to the snapshot/verifier is translated.

        `output_field_map` does the same for the TOOL'S RETURN VALUE, when it
        is a dict, e.g. a tool that returns {"status": 200, "result": {...}}
        instead of the {"status_code": ..., "body": ...} the HTTP verifier
        expects: output_field_map={"status_code": "status", "body": "result"}.
        The verifier sees the translated copy; the caller still gets the
        tool's real, unmodified output.
        """
        def _store(fn: Callable):
            self._verifiers[tool_name] = fn
            if snapshot is not None:
                self._snapshots[tool_name] = snapshot
            if field_map is not None:
                self._field_maps[tool_name] = field_map
            if output_field_map is not None:
                self._output_field_maps[tool_name] = output_field_map
            return fn

        if verifier_fn is not None:
            return _store(verifier_fn)
        return _store

    def get_verifier(self, tool_name: str) -> Optional[Callable]:
        return self._verifiers.get(tool_name)

    def get_snapshot(self, tool_name: str) -> Optional[Callable]:
        return self._snapshots.get(tool_name)

    def get_field_map(self, tool_name: str) -> Dict[str, str]:
        return self._field_maps.get(tool_name, {})

    def get_output_field_map(self, tool_name: str) -> Dict[str, str]:
        return self._output_field_maps.get(tool_name, {})


registry = VerifierRegistry()


# ---------------------------------------------------------
# Small bounded store of recent verdicts, for wrap=False callers who
# don't want a callback. Not a replacement for real logging/tracing.
# ---------------------------------------------------------
_MAX_RECENT_VERDICTS = 200
_recent_verdicts: "OrderedDict[str, dict]" = OrderedDict()


def _record_verdict(call_id: str, verdict_dict: dict) -> None:
    _recent_verdicts[call_id] = verdict_dict
    while len(_recent_verdicts) > _MAX_RECENT_VERDICTS:
        _recent_verdicts.popitem(last=False)


def get_verdict(call_id: str) -> Optional[dict]:
    """Look up a specific past verdict by its call_id."""
    return _recent_verdicts.get(call_id)


def get_last_verdict() -> Optional[dict]:
    """Convenience for single-threaded/demo use: the most recent verdict recorded."""
    if not _recent_verdicts:
        return None
    return next(reversed(_recent_verdicts.values()))


# ---------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------
def _classify_error(error: Exception) -> str:
    if "no such table" in str(error).lower():
        return FailureType.SCHEMA_ERROR
    return FailureType.CRASH


def _translate_kwargs(tool_name, kwargs):
    """Builds the kwargs dict the snapshot/verifier see, using field_map to pull
    values from the tool's own kwarg names into the names the verifier expects.
    The tool itself is always called with the ORIGINAL kwargs, unchanged."""
    field_map = registry.get_field_map(tool_name)
    if not field_map:
        return kwargs
    translated = dict(kwargs)
    for verifier_name, tool_name_key in field_map.items():
        if tool_name_key in kwargs:
            translated[verifier_name] = kwargs[tool_name_key]
    return translated


def _translate_output(tool_name, tool_output):
    """Same idea as _translate_kwargs but for the tool's return value, when
    it's a dict. The caller still receives the tool's real, unmodified output;
    only the copy handed to the verifier is translated."""
    output_field_map = registry.get_output_field_map(tool_name)
    if not output_field_map or not isinstance(tool_output, dict):
        return tool_output
    translated = dict(tool_output)
    for verifier_key, tool_key in output_field_map.items():
        if tool_key in tool_output:
            translated[verifier_key] = tool_output[tool_key]
    return translated


def _take_pre_state(tool_name, args, kwargs):
    snap = registry.get_snapshot(tool_name)
    if snap is None:
        return None
    try:
        return snap(args=args, kwargs=_translate_kwargs(tool_name, kwargs))
    except Exception:
        return None  # verifier decides what to do without a pre-state


def _run_verification(tool_name, args, kwargs, tool_output, error, pre_state, elapsed_ms, call_id) -> Verdict:
    def make(verified, basis, conf, reasoning, ftype):
        return Verdict(verified, basis, conf, reasoning, ftype, elapsed_ms, call_id)

    if error is not None:
        return make(False, "receipt-confirmed", "high",
                    f"Tool crashed during execution: {type(error).__name__} - {error}",
                    _classify_error(error))

    verifier_fn = registry.get_verifier(tool_name)
    if verifier_fn is None:
        return make(False, "no-check-possible", "unknown",
                    f"No verification function registered for tool '{tool_name}'.",
                    FailureType.NO_VERIFIER)

    try:
        translated_kwargs = _translate_kwargs(tool_name, kwargs)
        translated_output = _translate_output(tool_name, tool_output)
        extra = {"pre_state": pre_state} if "pre_state" in inspect.signature(verifier_fn).parameters else {}
        v = verifier_fn(args=args, kwargs=translated_kwargs, tool_output=translated_output, **extra)
        v.execution_time_ms = elapsed_ms
        v.call_id = call_id
        return v
    except Exception as v_err:
        return make(False, "no-check-possible", "unknown",
                    f"Verifier failed to run: {v_err}", FailureType.VERIFIER_ERROR)


# ---------------------------------------------------------
# INTERCEPTOR DECORATOR (SYNC + ASYNC)
# ---------------------------------------------------------
def verify_action(tool_name: str, wrap: bool = True, on_verdict: Callable[[dict], None] = None):
    """
    SDK decorator wrapping sync or async agent tools.

    wrap=True (default, unchanged from before): the wrapped function returns
        {"tool_result": ..., "error": ..., "verification": {...}}.
        This is a breaking change to the function's return shape - fine for
        new code, but it means anything else calling this function directly
        (outside the reliability layer) sees a different return value.

    wrap=False (non-invasive mode): the wrapped function returns EXACTLY
        what the original tool returns - same value on success, same
        exception raised on failure. Nothing about the tool's contract
        changes for any other caller. The verdict is instead delivered via:
          - `on_verdict(verdict_dict)`, called synchronously right after
            verification runs, if provided;
          - `verify_agent.get_verdict(call_id)` / `get_last_verdict()`, a
            small in-memory recent-verdicts store, for cases with no callback;
          - if the tool's own output is a dict, a `"__verification__"` key is
            also added to a COPY of it (a new key, so nothing existing is
            overwritten) purely as a convenience - ignore it if unwanted.
    """
    def decorator(func: Callable):

        def _finalize(tool_output, error, verdict: Verdict):
            verdict_dict = verdict.to_dict()
            if on_verdict is not None:
                try:
                    on_verdict(verdict_dict)
                except Exception:
                    pass  # a broken callback must never break the wrapped tool
            _record_verdict(verdict_dict["call_id"], verdict_dict)

            if wrap:
                return {"tool_result": tool_output, "error": str(error) if error else None,
                        "verification": verdict_dict}

            if error is not None:
                raise error  # non-invasive mode: preserve the tool's own failure contract
            if isinstance(tool_output, dict):
                annotated = dict(tool_output)
                annotated["__verification__"] = verdict_dict
                return annotated
            return tool_output

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            call_id = str(uuid.uuid4())[:8]
            pre_state = _take_pre_state(tool_name, args, kwargs)
            start = time.perf_counter()
            tool_output, error = None, None
            try:
                tool_output = await func(*args, **kwargs)
            except Exception as e:
                error = e
            elapsed_ms = (time.perf_counter() - start) * 1000
            verdict = _run_verification(tool_name, args, kwargs, tool_output, error, pre_state, elapsed_ms, call_id)
            return _finalize(tool_output, error, verdict)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            call_id = str(uuid.uuid4())[:8]
            pre_state = _take_pre_state(tool_name, args, kwargs)
            start = time.perf_counter()
            tool_output, error = None, None
            try:
                tool_output = func(*args, **kwargs)
            except Exception as e:
                error = e
            elapsed_ms = (time.perf_counter() - start) * 1000
            verdict = _run_verification(tool_name, args, kwargs, tool_output, error, pre_state, elapsed_ms, call_id)
            return _finalize(tool_output, error, verdict)

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator
