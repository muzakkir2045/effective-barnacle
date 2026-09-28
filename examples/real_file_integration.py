"""
Real-tool test #1 - LangChain's WriteFileTool (file verifier)
----------------------------------------------------------------

Wraps LangChain's actual WriteFileTool - the tool LangChain's own
FileManagementToolkit hands to agents - with @verify_action. Not a
mock: this is the real tool class, doing real filesystem writes,
including its own real root_dir sandboxing logic.

verify_file_written and snapshot_file_state are used completely
UNMODIFIED from universal_verifiers.py.

FINDING: WriteFileTool never raises. Every failure path (sandbox
escape, bad path, disk error) is caught internally and returned as a
string prefixed "Error: ..." - the same swallowed-error pattern the
LangChain SQL tool used in step 6. Unlike the SQL case, our file
verifier doesn't even need the adapter to translate that string into
an exception: verify_file_written only checks the real filesystem, so
a blocked write is caught correctly (false_success) whether or not
the adapter looks at the tool's return string at all. That's the
verify-the-side-effect design paying off on a second, unrelated tool.

Adapter note: WriteFileTool resolves relative file_path arguments
against its own root_dir internally. Our verifier needs to check the
REAL absolute path on disk, so the adapter takes both: `file_path`
(the real absolute path our verifier checks) and `tool_input_path`
(whatever string - honest or an escape attempt - gets handed to the
real tool). This is the same shape of translation as step 6's
query/db_path kwargs: the tool's own calling convention differs from
what a generic verifier needs, and that gap is bridged in the
adapter, not in the verifier.

Run:
    python real_file_integration.py
    python real_file_integration.py -v
"""
import argparse
import os
import shutil
import sys
import tempfile
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional

warnings.filterwarnings("ignore")

from langchain_community.tools.file_management import WriteFileTool

from agent_reliability import registry, verify_action, FailureType as F, verify_file_written, snapshot_file_state

registry.register("write_file", verify_file_written, snapshot=snapshot_file_state)


# ------------------------------------------------------------------
# Adapter: real WriteFileTool call + the absolute-path translation
# described above. No verifier code touched.
# ------------------------------------------------------------------
@verify_action("write_file")
def langchain_write_file_tool(file_path: str, tool_input_path: str, text: str,
                               root_dir: str, append: bool = False):
    tool = WriteFileTool(root_dir=root_dir)
    return tool.run({"file_path": tool_input_path, "text": text, "append": append})


# ------------------------------------------------------------------
# Scenarios
# ------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    expect_verified: bool
    expect_failure: str
    build: Callable[[str], dict]   # receives root_dir, returns kwargs for the adapter


def build_scenarios() -> List[Scenario]:
    return [
        Scenario(
            "honest_success", True, F.NONE,
            lambda root: dict(file_path=os.path.join(root, "notes.txt"),
                               tool_input_path="notes.txt", text="hello", root_dir=root),
        ),
        Scenario(
            # WriteFileTool blocks the escape and returns an "Error: ..." string;
            # nothing is written at the real target -> caught purely by checking disk.
            "sandbox_escape_relative", False, F.FALSE_SUCCESS,
            lambda root: dict(file_path=os.path.join(os.path.dirname(root), "escape_rel.txt"),
                               tool_input_path="../escape_rel.txt", text="leak", root_dir=root),
        ),
        Scenario(
            "sandbox_escape_absolute", False, F.FALSE_SUCCESS,
            lambda root: dict(file_path="/tmp/escape_abs_target.txt",
                               tool_input_path="/tmp/escape_abs_target.txt", text="leak", root_dir=root),
        ),
        Scenario(
            # pre-seed the file, then have the tool's write get blocked -> file exists
            # but is provably untouched. Exercises the fingerprint-based "already existed,
            # unchanged" branch (step 1's reconciled file verifier) against a real tool.
            "sandbox_escape_leaves_target_untouched", False, F.FALSE_SUCCESS,
            lambda root: _preseed_then_block(root),
        ),
        Scenario(
            "append_grows_existing_file", True, F.NONE,
            lambda root: _preseed_then_append(root),
        ),
    ]


def _preseed_then_block(root: str) -> dict:
    target = os.path.join(os.path.dirname(root), "already_there.txt")
    with open(target, "w") as f:
        f.write("original content")
    return dict(file_path=target, tool_input_path="../already_there.txt",
                text="attempted overwrite", root_dir=root)


def _preseed_then_append(root: str) -> dict:
    target = os.path.join(root, "log.txt")
    with open(target, "w") as f:
        f.write("line1\n")
    return dict(file_path=target, tool_input_path="log.txt", text="line2\n",
                root_dir=root, append=True)


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------
def label(verified: bool, failure: str) -> str:
    return f"{'verified' if verified else 'rejected'}/{failure}"


def run_one(sc: Scenario, tmpdir: str):
    root = os.path.join(tmpdir, sc.name, "sandbox")
    os.makedirs(root, exist_ok=True)
    kwargs = sc.build(root)
    return langchain_write_file_tool(**kwargs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    opts = ap.parse_args()

    scenarios = build_scenarios()
    width = max(len(s.name) for s in scenarios)
    print(f"\n{'#':>2}  {'RESULT':<6}  {'SCENARIO':<{width}}  {'EXPECTED':<24}  ACTUAL")
    print("-" * (width + 78))

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, sc in enumerate(scenarios, 1):
            expected = label(sc.expect_verified, sc.expect_failure)
            result = run_one(sc, tmp)
            v = result["verification"]
            actual = label(v["verified"], v["failure_type"])
            ok = actual == expected

            print(f"{i:>2}  {'PASS' if ok else 'FAIL':<6}  {sc.name:<{width}}  {expected:<24}  {actual}")
            if opts.verbose or not ok:
                print(f"      tool_result: {result['tool_result']!r}")
                print(f"      reasoning:   {v['reasoning']}")
            if not ok:
                failures.append(sc.name)

    print("-" * (width + 78))
    total = len(scenarios)
    print(f"{total - len(failures)}/{total} passed against the REAL LangChain WriteFileTool"
          + (f"  |  FAILED: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
