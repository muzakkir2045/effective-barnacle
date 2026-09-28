"""
Real-tool test #2 - LangChain's RequestsGetTool (HTTP verifier)
----------------------------------------------------------------

Wraps LangChain's actual RequestsGetTool with @verify_action. This
makes REAL HTTP calls over REAL sockets - not mocked dicts - against
a small local HTTP server we spin up for the test, so the result is
deterministic and doesn't depend on a third-party API's rate limits.
(A live probe against api.github.com hit exactly that problem: this
sandbox's shared egress IP was already rate-limited, which would make
a regression suite flaky through no fault of our own code - so the
target here is a real server we control the responses of, not a real
external endpoint.)

verify_http_status is used completely UNMODIFIED from
universal_verifiers.py.

FINDING (bigger than the step-6 SQL finding): RequestsGetTool's
output is `response.text` - ONLY the response body. The HTTP status
code is never exposed at all, on ANY response, including a 404. An
agent using this tool as-is has no way to distinguish a 200 from a
404 unless it happens to notice wording differences in the body text.
This isn't a contrived edge case; confirmed directly against a real
404 from a real socket (see probe output below the scenarios).

Because of this, the adapter does the same thing verify_sql_mutation
does for SQL: it independently OBSERVES real state rather than
trusting the tool's self-report. A GET is not a mutation, so issuing
a second real GET purely to read the status code is consistent with
the project's "verifiers only read, never re-execute a mutation"
rule - re-running a GET has no side effect to duplicate. No verifier
code changed; the independent status check lives in the adapter.

Run:
    python real_http_integration.py
    python real_http_integration.py -v
"""
import argparse
import json
import sys
import threading
import warnings
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, List

warnings.filterwarnings("ignore")

import requests
from langchain_community.tools.requests.tool import RequestsGetTool
from langchain_community.utilities.requests import TextRequestsWrapper

from agent_reliability import registry, verify_action, FailureType as F, verify_http_status

registry.register("make_http_request", verify_http_status)


# ------------------------------------------------------------------
# A tiny real HTTP server (loopback only) so the test is deterministic.
# ------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    ROUTES = {
        "/ok":       (200, {"ok": True, "id": 42}),
        "/degraded": (200, {"ok": False, "error": "upstream unavailable"}),  # 200 but body says failure
        "/missing":  (404, {"message": "not found"}),
    }

    def do_GET(self):
        status, payload = self.ROUTES.get(self.path, (404, {"message": "no route"}))
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # keep test output clean


class LocalServer:
    def __enter__(self):
        self.srv = HTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self.srv.server_port
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        return self

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def __exit__(self, *exc):
        self.srv.shutdown()


# ------------------------------------------------------------------
# Adapter: real LangChain tool for the body + an independent real GET
# for the status code, since the tool itself discards it.
# ------------------------------------------------------------------
_wrapper = TextRequestsWrapper()
_tool = RequestsGetTool(requests_wrapper=_wrapper, allow_dangerous_requests=True)


@verify_action("make_http_request")
def langchain_http_get_tool(url: str):
    body_text = _tool.run(url)                    # the real, widely-used agent tool
    status_code = requests.get(url).status_code    # independent real observation (read-only, not a mutation)
    return {"status_code": status_code, "body": body_text}


# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    path: str
    expect_verified: bool
    expect_failure: str


SCENARIOS: List[Scenario] = [
    Scenario("honest_success",    "/ok",       True,  F.NONE),
    Scenario("error_status_code", "/missing",  False, F.HTTP_ERROR),
    Scenario("false_success_body", "/degraded", False, F.FALSE_SUCCESS),
]


def label(verified: bool, failure: str) -> str:
    return f"{'verified' if verified else 'rejected'}/{failure}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    opts = ap.parse_args()

    print(f"\n{'#':>2}  {'RESULT':<6}  {'SCENARIO':<20}  {'EXPECTED':<22}  ACTUAL")
    print("-" * 76)

    failures = []
    with LocalServer() as server:
        # Probe first, printed once, showing exactly what the real tool loses.
        raw_404 = _tool.run(server.url("/missing"))
        print(f"Probe - what RequestsGetTool alone returns for a real 404: {raw_404!r}")
        print(f"        (no status code anywhere in that output)\n")

        for i, sc in enumerate(SCENARIOS, 1):
            expected = label(sc.expect_verified, sc.expect_failure)
            result = langchain_http_get_tool(url=server.url(sc.path))
            v = result["verification"]
            actual = label(v["verified"], v["failure_type"])
            ok = actual == expected

            print(f"{i:>2}  {'PASS' if ok else 'FAIL':<6}  {sc.name:<20}  {expected:<22}  {actual}")
            if opts.verbose or not ok:
                print(f"      tool_result: {result['tool_result']!r}")
                print(f"      reasoning:   {v['reasoning']}")
            if not ok:
                failures.append(sc.name)

    print("-" * 76)
    total = len(SCENARIOS)
    print(f"{total - len(failures)}/{total} passed against the REAL LangChain RequestsGetTool"
          + (f"  |  FAILED: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
