"""Virtual Best Classical (VBC) reference.

The honest strong baseline is not any single classical policy — different
policies win in different regimes. VBC takes, per scenario, the *minimum* cost
over the classical policies: "the best a traditional method could do if you
always picked the right one." The paper's central test is then whether the
hybrid agent beats VBC, i.e. adds value *beyond* selecting/tuning the right
classical policy.
"""

from __future__ import annotations

from typing import Dict, Iterable, List


def virtual_best_classical(
    rows: List[Dict], classical: Iterable[str], group_key: str = "scenario_id"
) -> Dict[str, dict]:
    """Return {scenario_id: {"cost": best_cost, "policy": winning_policy}} over
    the given classical policy names."""
    classical = set(classical)
    best: Dict[str, dict] = {}
    for r in rows:
        if r["policy"] not in classical:
            continue
        sid = r[group_key]
        cur = best.get(sid)
        if cur is None or r["total_cost"] < cur["cost"]:
            best[sid] = {"cost": r["total_cost"], "policy": r["policy"]}
    return best


def add_vbc_regret(rows: List[Dict], classical: Iterable[str],
                   group_key: str = "scenario_id") -> Dict[str, dict]:
    """Add 'regret_vs_vbc' (cost above per-scenario best classical) and
    'beats_vbc' (bool) to every row. Returns the VBC map for reporting which
    classical policy won each scenario."""
    vbc = virtual_best_classical(rows, classical, group_key)
    for r in rows:
        b = vbc.get(r[group_key])
        if b is None:
            r["regret_vs_vbc"] = 0.0
            r["beats_vbc"] = False
        else:
            r["regret_vs_vbc"] = r["total_cost"] - b["cost"]
            r["beats_vbc"] = r["total_cost"] < b["cost"] - 1e-9
    return vbc
