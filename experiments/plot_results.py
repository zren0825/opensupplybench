"""Visualize a benchmark results CSV (Phase 6, offline).

Produces publication-style figures under experiments/figures/:
  1. cost_by_policy.png      — normalized cost (cost/oracle), uniform vs prevalence
  2. cost_by_demand.png      — heatmap: policy × demand pattern
  3. cost_by_sku.png         — heatmap: policy × SKU archetype
  4. fillrate_and_vbc.png    — fill rate + Virtual-Best-Classical win share

Design: categorical policy colors use the Okabe-Ito colorblind-safe palette
(fixed per policy, not by rank); heatmaps use a single-hue sequential ramp where
darker = higher cost = worse. Lower cost/oracle is better throughout.

Usage:
    python experiments/plot_results.py --results data/results.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opensupply.policies import CLASSICAL_FOR_VBC
from opensupply.evaluation import virtual_best_classical
from opensupply.evaluation.weighting import cost_ratio, weighted_metric

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# Okabe-Ito colorblind-safe palette, assigned to policy identity (never by rank).
POLICY_COLOR = {
    "rule_based": "#0072B2",       # blue
    "forecast_safety": "#E69F00",  # orange
    "periodic_review": "#009E73",  # green
    "cost_min": "#D55E00",         # vermillion
    "tuned_ss": "#CC79A7",         # purple
    "hybrid": "#000000",           # black (when present)
    "llm_only": "#56B4E9",         # sky blue (when present)
}
SCHEME_COLOR = {"uniform": "#0072B2", "prevalence": "#E69F00"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
_NUM = ("total_cost", "oracle_cost", "regret", "fill_rate")


def load_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows or "oracle_cost" not in rows[0]:
        raise SystemExit("results CSV missing 'oracle_cost' — regenerate with run_benchmark.py")
    for r in rows:
        for k in _NUM:
            if r.get(k, "") != "":
                r[k] = float(r[k])
    return rows


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_axisbelow(True)


def fig_cost_by_policy(by_policy, order, path):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    _style(ax)
    x = np.arange(len(order))
    w = 0.38
    for i, scheme in enumerate(("uniform", "prevalence")):
        vals = [weighted_metric(by_policy[p], cost_ratio, scheme) for p in order]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=scheme,
                      color=SCHEME_COLOR[scheme], zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.axhline(1.0, color=MUTED, lw=1, ls="--", zorder=2)
    ax.text(len(order) - 0.5, 1.02, "oracle", ha="right", va="bottom",
            fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9, color=INK)
    ax.set_ylabel("cost ÷ clairvoyant oracle  (lower is better)", color=INK, fontsize=10)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_title("Normalized cost by policy — uniform vs. prevalence weighting",
                 color=INK, fontsize=12, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, title="scenario weighting", title_fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _heatmap(by_policy, order, rows, key, title, path):
    levels = sorted({r[key] for r in rows if r.get(key)})
    M = np.full((len(order), len(levels)), np.nan)
    for i, p in enumerate(order):
        for j, lv in enumerate(levels):
            sub = [r for r in by_policy[p] if r.get(key) == lv]
            if sub:
                M[i, j] = weighted_metric(sub, cost_ratio, "uniform")

    fig, ax = plt.subplots(figsize=(1.3 * len(levels) + 2.5, 0.7 * len(order) + 2))
    vmax = np.nanpercentile(M, 95)
    im = ax.imshow(M, cmap="Reds", aspect="auto", vmin=1.0, vmax=vmax)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels, rotation=30, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9, color=INK)
    for i in range(len(order)):
        for j in range(len(levels)):
            if not np.isnan(M[i, j]):
                shade = "white" if M[i, j] > (1.0 + 0.62 * (vmax - 1.0)) else INK
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        fontsize=8, color=shade)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("cost ÷ oracle", color=INK, fontsize=9)
    cb.ax.tick_params(colors=MUTED, length=0)
    ax.set_title(title, color=INK, fontsize=12, weight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_fillrate_and_vbc(by_policy, order, rows, classical, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = [POLICY_COLOR.get(p, "#888888") for p in order]

    _style(ax1)
    fills = [weighted_metric(by_policy[p], lambda r: r["fill_rate"], "uniform") for p in order]
    bars = ax1.bar(range(len(order)), fills, color=colors, zorder=3, width=0.6)
    for b, v in zip(bars, fills):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.1%}",
                 ha="center", va="bottom", fontsize=8, color=INK)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("fill rate (achieved service level)", color=INK, fontsize=10)
    ax1.yaxis.grid(True, color=GRID, lw=0.8)
    ax1.set_xticks(range(len(order)))
    ax1.set_xticklabels(order, rotation=20, ha="right", fontsize=9, color=INK)
    ax1.set_title("Fill rate by policy", color=INK, fontsize=12, weight="bold", loc="left")

    # VBC win share: how often each classical policy is the per-scenario best
    _style(ax2)
    vbc = virtual_best_classical(rows, classical)
    wins = defaultdict(int)
    for v in vbc.values():
        wins[v["policy"]] += 1
    tot = sum(wins.values()) or 1
    corder = [p for p in order if p in classical]
    shares = [wins[p] / tot for p in corder]
    bars = ax2.bar(range(len(corder)), shares,
                   color=[POLICY_COLOR.get(p, "#888888") for p in corder],
                   zorder=3, width=0.6)
    for b, v in zip(bars, shares):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.0%}",
                 ha="center", va="bottom", fontsize=8, color=INK)
    ax2.set_ylabel("share of scenarios where this is the best classical", color=INK, fontsize=10)
    ax2.yaxis.grid(True, color=GRID, lw=0.8)
    ax2.set_xticks(range(len(corder)))
    ax2.set_xticklabels(corder, rotation=20, ha="right", fontsize=9, color=INK)
    ax2.set_title("Virtual Best Classical — who wins where\n(no single policy dominates → VBC is a strong bar)",
                  color=INK, fontsize=11, weight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/results.csv")
    args = ap.parse_args()

    rows = load_rows(args.results)
    classical = [p for p in CLASSICAL_FOR_VBC if any(r["policy"] == p for r in rows)]
    by_policy = defaultdict(list)
    for r in rows:
        by_policy[r["policy"]].append(r)
    order = sorted(by_policy, key=lambda p: weighted_metric(by_policy[p], cost_ratio, "uniform"))

    os.makedirs(FIG_DIR, exist_ok=True)
    n = len({r["scenario_id"] for r in rows})
    fig_cost_by_policy(by_policy, order, os.path.join(FIG_DIR, "cost_by_policy.png"))
    _heatmap(by_policy, order, rows, "demand_type",
             "Normalized cost by policy × demand pattern",
             os.path.join(FIG_DIR, "cost_by_demand.png"))
    _heatmap(by_policy, order, rows, "sku_type",
             "Normalized cost by policy × SKU archetype",
             os.path.join(FIG_DIR, "cost_by_sku.png"))
    fig_fillrate_and_vbc(by_policy, order, rows, classical,
                         os.path.join(FIG_DIR, "fillrate_and_vbc.png"))
    print(f"Wrote 4 figures for {n} scenarios to {os.path.relpath(FIG_DIR)}/")


if __name__ == "__main__":
    main()
