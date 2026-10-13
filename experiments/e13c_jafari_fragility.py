"""Хрупкость хаотического множества Jafari 2026 (a3 = -3.548): доля копий, остающихся ограниченными
до T = 20000, в зависимости от амплитуды шума (1e-6, 1e-4, 1e-3) и шага (0.005, 0.0025).
Выход: results/e13c_jafari_fragility.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.integrate import F_ESC, nfeat  # noqa: E402


def main():
    s = systems.get("jafari2026_chaos")
    rng = np.random.default_rng(5)
    state = s.known["HA"]
    out = []
    for noise in (1e-6, 1e-4, 1e-3):
        for dt in (0.005, 0.0025):
            n_copies = 16
            x = state + noise * (1 + np.abs(state)) * rng.normal(size=(n_copies, 3))
            alive = np.ones(n_copies, bool)
            t_end = np.full(n_copies, np.nan)
            n_obs = int(200 / dt)
            for w in range(100):
                feats = np.zeros((n_copies, nfeat(3))); fin = np.zeros((n_copies, 3))
                s.batch(x, s.params, dt, 0, n_obs, s.esc_r, int(0.1 / dt), feats, fin)
                esc = feats[:, F_ESC] > 0.5
                t_end[alive & esc] = (w + 1) * 200
                alive &= ~esc
                x = np.where(alive[:, None], fin, state)
                if not alive.any():
                    break
            te = t_end[~np.isnan(t_end)]
            med = float(np.median(te)) if len(te) else None
            out.append({"noise": noise, "dt": dt, "alive_at_20000": int(alive.sum()), "n": n_copies, "median_t_escape": med})
            print(f"noise={noise:g} dt={dt}: живых при T=20000: {alive.sum()}/{n_copies}; медиана времени ухода {med}")
    (ROOT / "results" / "e13c_jafari_fragility.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
