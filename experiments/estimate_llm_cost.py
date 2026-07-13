"""Estimate LLM benchmark cost before spending anything (no API calls).

Usage:
    python experiments/estimate_llm_cost.py --scenarios 486 --horizon 90
    python experiments/estimate_llm_cost.py --scenarios 162 --model claude-opus-4-8
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply.agents.cost import format_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", type=int, default=486,
                    help="number of scenarios the LLM methods will run on")
    ap.add_argument("--horizon", type=int, default=90)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--methods", nargs="+", default=["llm_only", "hybrid"])
    args = ap.parse_args()
    print(format_table(args.methods, args.scenarios, args.model, args.horizon))
    print("\n(BATCH = Anthropic Batch API, 50% off; intro = promo Sonnet-5 rate "
          "through 2026-08-31.)")


if __name__ == "__main__":
    main()
