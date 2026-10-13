"""Диагностика subset simulation: траектория уровней, перезапуски, момент попадания (по запускам)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from e20_search_benchmark import build_problem  # noqa: E402
from hsearch.search.harness import Problem, run_search  # noqa: E402
from hsearch.search.sus import SubsetSim  # noqa: E402


def main(name="edge_se3_m3_p0.001", score="rarity", R=8, budget=10000):
    s, refs, tidx, kidx = build_problem(name, np.random.default_rng(100))
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        prob = Problem(s, refs, tidx, kidx, rng)
        st = SubsetSim(prob, n0=100, score=score)
        res = run_search(prob, st, budget)
        lv = [(a, b, round(c, 3), round(d, 5)) for a, b, c, d in st.level_log]
        print(f"run {r}: found_at={res['found_at']} restarts={st.n_restarts} levels={lv[:12]}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []), *(sys.argv[2:3] or []))
