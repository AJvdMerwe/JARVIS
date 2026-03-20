#!/usr/bin/env python
"""
evaluation/run_evals.py
────────────────────────
CLI runner for the evaluation harness.

Usage:
    python evaluation/run_evals.py --suite smoke
    python evaluation/run_evals.py --suite code --save
    python evaluation/run_evals.py --suite all --stop-on-fail
    python evaluation/run_evals.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent evaluation suites.")
    parser.add_argument(
        "--suite", default="smoke",
        help="Suite to run: smoke | code | news | search | document | routing | all",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available suites and exit.",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save JSON report to data/eval_reports/.",
    )
    parser.add_argument(
        "--stop-on-fail", action="store_true",
        help="Abort after first failing test.",
    )
    parser.add_argument(
        "--session", default="eval",
        help="Orchestrator session ID for the eval run.",
    )
    args = parser.parse_args()

    from evaluation.builtin_suites import SUITES, get_all_suites

    if args.list:
        print("\nAvailable evaluation suites:")
        for name in SUITES:
            suite = SUITES[name]()
            print(f"  {name:<12} ({len(suite)} cases)")
        print(f"  {'all':<12} (all cases combined)")
        return

    if args.suite == "all":
        suite = get_all_suites()
    elif args.suite in SUITES:
        suite = SUITES[args.suite]()
    else:
        print(f"Unknown suite '{args.suite}'. Use --list to see available suites.")
        sys.exit(1)

    from evaluation.eval_harness import Evaluator

    evaluator = Evaluator(
        session_id=args.session,
        stop_on_fail=args.stop_on_fail,
    )
    report = evaluator.run(suite)
    report.print_summary()

    if args.save:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("data/eval_reports") / f"{suite.name}_{ts}.json"
        report.save(out)

    sys.exit(0 if report.failed == 0 else 1)


if __name__ == "__main__":
    main()
