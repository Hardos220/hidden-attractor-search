"""Проверка синтетического семейства уровня D: измеренная доля бассейна против аналитической.

База: se3 (устойчивое равновесие в нуле + скрытый хаотический аттрактор). Для m in (1, 3, 7)
и целевых p in (0.05, 0.01, 0.002) подбирается rho, измеряется p по выборке n точек в B,
проверяется, что найденный режим совпадает с эталоном HA базовой системы (по x-компонентам).
Выход: results/e09_synthetic.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.systems.synthetic import analytic_p, make_embedded, rho_for_p  # noqa: E402
from hsearch.identify import LAB_ESCAPE, LAB_UNKNOWN, calibrate_reference, fp_references_from_equilibria, match  # noqa: E402
from hsearch.integrate import run_batch  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def main():
    rng = np.random.default_rng(9)
    base = "se3"
    e02 = json.loads((ROOT / "results" / "e02_uniform.json").read_text(encoding="utf-8"))
    p_base = e02[base]["hidden"]["HA"]["p"]
    out = {"base": base, "p_base": p_base, "cases": []}
    for m in (1, 3, 7):
        for p_target in (0.05, 0.01, 0.002):
            rho = rho_for_p(p_target, p_base, m)
            if rho > 1.0:
                continue
            s = make_embedded(base, m, rho)
            refs = fp_references_from_equilibria(s, names=["FP"])
            refs.append(calibrate_reference(s, "HA", s.known["HA"], rng))
            n = 20000 if p_target >= 0.01 else 50000
            t0 = time.time()
            ics = s.sample_box(n, rng)
            feats, finals = run_batch(s, ics)
            labels, z = match(s, feats, finals, refs)
            k_ha = int(np.sum(labels == [k for k, r in enumerate(refs) if r.name == "HA"][0]))
            p_meas = k_ha / n
            rec = {"m": m, "rho": rho, "p_target": p_target, "p_analytic": analytic_p(p_base, m, rho),
                   "n": n, "p_measured": p_meas, "ci95": wilson(k_ha, n),
                   "p_escape": float((labels == LAB_ESCAPE).mean()), "p_unknown": float((labels == LAB_UNKNOWN).mean()),
                   "ref_kind": refs[-1].kind, "ref_lam": refs[-1].lam, "time_s": round(time.time() - t0, 1)}
            out["cases"].append(rec)
            print(f"m={m} rho={rho:.4f}: p_target={p_target} p_analytic={rec['p_analytic']:.5f} "
                  f"p_measured={p_meas:.5f} CI={np.round(rec['ci95'], 5)} unknown={rec['p_unknown']:.4f} "
                  f"HA kind={rec['ref_kind']} λ={rec['ref_lam']:.3f} ({rec['time_s']} с)")
    (ROOT / "results" / "e09_synthetic.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
