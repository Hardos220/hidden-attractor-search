"""Проверка семейства D2 («краевое состояние»): измеренная доля бассейна против точной формулы
и градиентность времени ухода у границы бассейна. Выход: results/e23_edge_check.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from e20_search_benchmark import build_problem  # noqa: E402
from hsearch.integrate import F_ESC, F_TESC, run_batch  # noqa: E402
from hsearch.identify import match  # noqa: E402
from hsearch.systems.synthetic import analytic_p  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def main():
    out = {"cases": []}
    for name, n in (("edge_se3_m1_p0.01", 20000), ("edge_se3_m3_p0.01", 20000), ("edge_se3_m3_p0.001", 50000),
                    ("edge_se3_m7_p0.001", 50000)):
        rng = np.random.default_rng(23)
        t0 = time.time()
        s, refs, tidx, kidx = build_problem(name, rng)
        ics = s.sample_box(n, rng)
        feats, finals = run_batch(s, ics)
        labels, z = match(s, feats, finals, refs)
        k = int(np.isin(labels, tidx).sum())
        m = s.meta["m"]; rho = s.meta["rho"]
        p_exact = analytic_p(0.1525, m, rho)
        # градиентность: время ухода как функция |w0|/rho для ушедших
        w = np.linalg.norm(ics[:, 3:], axis=1) / rho
        esc = feats[:, F_ESC] > 0.5
        bins = [1.0, 1.001, 1.01, 1.1, 1.5, 3.0, 1e9]
        grad = []
        for a, b in zip(bins[:-1], bins[1:]):
            sel = esc & (w >= a) & (w < b)
            grad.append({"w_over_rho": [a, b], "n": int(sel.sum()),
                         "median_t_esc": float(np.median(feats[sel, F_TESC]) * s.dt) if sel.any() else None})
        rec = {"system": name, "m": m, "rho": rho, "n": n, "p_measured": k / n, "ci95": wilson(k, n),
               "p_formula": p_exact, "refs": [(r.name, r.kind) for r in refs], "t_esc_vs_w": grad,
               "time_s": round(time.time() - t0, 1)}
        out["cases"].append(rec)
        print(f"{name}: rho={rho:.4f} p_formula={p_exact:.5f} p_measured={k / n:.5f} CI={np.round(rec['ci95'], 5)}; "
              f"t_esc по |w0|/rho: " + ", ".join(f"[{g['w_over_rho'][0]}, {g['w_over_rho'][1]:.3g}): {g['median_t_esc']}" for g in grad))
    (ROOT / "results" / "e23_edge_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
