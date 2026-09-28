"""
Step 5 - Detection rate / false-alarm rate metrics
----------------------------------------------------

PASS/FAIL in the harnesses only tells you whether the verifier matched
the label YOU assigned to a scenario. It does not say how good the
verifier actually is at catching lies vs. leaving honest tools alone.

This file reclassifies every scenario from both harnesses along a
different axis:

    - Was the tool's self-report ACTUALLY correct or not?
        expect_verified == True   -> tool's report was HONEST
        expect_verified == False  -> tool's report was a LIE
          (false_success / self_report_mismatch / partial_success /
           crash / schema_error / no_verifier - anything the tool
           over-claimed, and the engine SHOULD reject)

    - Did our engine's ACTUAL verdict agree?

That gives a standard 2x2 per scenario:

    Tool lied,  we rejected it   -> True Positive  (caught the lie)
    Tool lied,  we verified it   -> False Negative (missed the lie - dangerous)
    Tool honest, we verified it  -> True Negative  (correctly let through)
    Tool honest, we rejected it  -> False Positive (false alarm)

Detection rate   = TP / (TP + FN)   -- of all the lies, what % did we catch
False-alarm rate = FP / (FP + TN)   -- of all honest tools, what % did we wrongly flag

Both are reported overall AND broken down per failure_type, and per
verifier/tool family (sql / schema / file / json / http / regex /
engine-level), pulled straight from each scenario's EXPECTED label so
the numbers are independent of whether the harness's own PASS/FAIL
check happens to be green.

Run:
    python metrics.py                 # table + summary
    python metrics.py -v              # also print every scenario row
    python metrics.py --json out.json # write the raw numbers to a file
"""
import argparse
import json as json_mod
import sqlite3
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import test_harness as th
import adversarial_harness as ah


# ------------------------------------------------------------------
# Which harness a scenario's own BASE_SEED/run_one belong to, so we
# seed and execute it the way that harness expects.
# ------------------------------------------------------------------
@dataclass
class Source:
    label: str
    base_seed: str
    run_one: callable


SOURCES = [
    Source("test_harness", th.BASE_SEED, th.run_one),
    Source("adversarial_harness", ah.BASE_SEED, ah.run_one),
]

# Rough scenario-name -> tool family, for the per-family breakdown.
# Anything not matched falls into "sql" (today everything not file/json/
# http/regex/schema IS a SQL mutation scenario, across both harnesses).
_FAMILY_HINTS = [
    ("no_verifier", "engine"),
    ("schema", "sql_schema"),
    ("file", "file"),
    ("json", "json"),
    ("http", "http"),
    ("regex", "regex"),
]


def _family(name: str) -> str:
    for hint, fam in _FAMILY_HINTS:
        if hint in name:
            return fam
    return "sql_mutation"


# ------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------
@dataclass
class Row:
    source: str
    name: str
    family: str
    tool_was_honest: bool     # ground truth: was the tool's self-report correct?
    expected_label: str       # verified/none, rejected/false_success, ...
    actual_label: str
    outcome: str              # TP / FN / TN / FP / ERROR
    reasoning: str


def _label(verified: bool, failure: str) -> str:
    return f"{'verified' if verified else 'rejected'}/{failure}"


def _classify(expect_verified: bool, actual_verified: bool) -> str:
    if not expect_verified and not actual_verified:
        return "TP"   # tool lied, we caught it
    if not expect_verified and actual_verified:
        return "FN"   # tool lied, we missed it  <- worst case
    if expect_verified and actual_verified:
        return "TN"   # tool honest, correctly let through
    return "FP"        # tool honest, we wrongly flagged it


def collect_rows(verbose: bool) -> List[Row]:
    rows: List[Row] = []
    for src in SOURCES:
        scenarios = th.build_scenarios() if src.label == "test_harness" else ah.build_scenarios()
        with tempfile.TemporaryDirectory() as tmp:
            for sc in scenarios:
                expected_label = _label(sc.expect_verified, sc.expect_failure)
                try:
                    v = src.run_one(sc, tmp)
                    actual_verified = v["verified"]
                    actual_label = _label(actual_verified, v["failure_type"])
                    reasoning = v["reasoning"]
                except Exception as e:
                    actual_verified = None
                    actual_label = f"ERROR/{type(e).__name__}"
                    reasoning = str(e)

                outcome = "ERROR" if actual_verified is None else _classify(sc.expect_verified, actual_verified)
                rows.append(Row(
                    source=src.label,
                    name=sc.name,
                    family=_family(sc.name),
                    tool_was_honest=sc.expect_verified,
                    expected_label=expected_label,
                    actual_label=actual_label,
                    outcome=outcome,
                    reasoning=reasoning,
                ))
    return rows


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------
def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def aggregate(rows: List[Row], keyfn) -> Dict[str, Dict[str, int]]:
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        buckets[keyfn(r)][r.outcome] += 1
    return buckets


def print_table(title: str, buckets: Dict[str, Dict[str, int]]):
    print(f"\n{title}")
    width = max([len(k) for k in buckets] + [10])
    print(f"{'':<{width}}  {'TP':>4} {'FN':>4} {'TN':>4} {'FP':>4}   "
          f"{'DETECTION':>10}  {'FALSE-ALARM':>12}")
    print("-" * (width + 44))
    for key in sorted(buckets):
        c = buckets[key]
        tp, fn, tn, fp = c["TP"], c["FN"], c["TN"], c["FP"]
        det = _rate(tp, tp + fn)
        far = _rate(fp, fp + tn)
        print(f"{key:<{width}}  {tp:>4} {fn:>4} {tn:>4} {fp:>4}   {det:>10}  {far:>12}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="print every scenario row")
    ap.add_argument("--json", metavar="FILE", help="write raw metrics as JSON to FILE")
    opts = ap.parse_args()

    rows = collect_rows(opts.verbose)

    if opts.verbose:
        print(f"\n{'SOURCE':<20}{'SCENARIO':<30}{'FAMILY':<14}{'OUTCOME':<8}{'EXPECTED':<26}ACTUAL")
        for r in rows:
            print(f"{r.source:<20}{r.name:<30}{r.family:<14}{r.outcome:<8}{r.expected_label:<26}{r.actual_label}")

    errors = [r for r in rows if r.outcome == "ERROR"]

    by_failure_type = aggregate(
        [r for r in rows if not r.tool_was_honest],
        lambda r: r.expected_label.split("/", 1)[1],
    )
    print_table("Detection rate by failure_type (lies only; false-alarm column is n/a here)", by_failure_type)

    by_family = aggregate(rows, lambda r: r.family)
    print_table("Detection rate & false-alarm rate by verifier family", by_family)

    by_source = aggregate(rows, lambda r: r.source)
    print_table("By harness (source file)", by_source)

    overall = aggregate(rows, lambda r: "overall")
    print_table("Overall", overall)

    total = len(rows)
    tp = sum(r.outcome == "TP" for r in rows)
    fn = sum(r.outcome == "FN" for r in rows)
    tn = sum(r.outcome == "TN" for r in rows)
    fp = sum(r.outcome == "FP" for r in rows)

    print(f"\n{total} scenarios total ({len(errors)} harness errors).")
    print(f"Overall detection rate   : {_rate(tp, tp + fn)}  ({tp}/{tp + fn} lies caught)")
    print(f"Overall false-alarm rate : {_rate(fp, fp + tn)}  ({fp}/{fp + tn} honest tools wrongly flagged)")

    if opts.json:
        payload = {
            "rows": [r.__dict__ for r in rows],
            "overall": {"tp": tp, "fn": fn, "tn": tn, "fp": fp},
        }
        with open(opts.json, "w") as f:
            json_mod.dump(payload, f, indent=2)
        print(f"\nWrote raw metrics to {opts.json}")

    if errors:
        print(f"\n{len(errors)} scenario(s) raised harness errors (not counted in rates):")
        for r in errors:
            print(f"  {r.source}/{r.name}: {r.reasoning}")
        return 1

    if fn > 0:
        print(f"\nWARNING: {fn} missed lie(s) (False Negatives) - see -v for details.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
