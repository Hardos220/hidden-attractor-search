"""Сходимость долей бассейнов Чуа (набор I, шар r=3) по шагу интегрирования RK4.

Для dt из (0.01, 0.005, 0.0025, 0.00125) при T_tr = 1000, T_obs = 100 и n = 20000
точек в шаре радиуса 3: доли ∞ / F0 / HA_a / HA_b с 95% интервалами Вилсона.
Выход: results/e04_chua_dt_convergence.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import LAB_ESCAPE, LAB_UNKNOWN, Reference, match  # noqa: E402
from hsearch.integrate import F_LAM, nfeat  # noqa: E402


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def run(s, ics, dt, T_tr, T_obs, refs):
    n = len(ics)
    feats = np.zeros((n, nfeat(3))); finals = np.zeros((n, 3))
    n_tr = int(round(T_tr / dt)); n_obs = int(round(T_obs / dt))
    renorm = max(1, int(round(0.1 / dt)))
    s.batch(ics, s.params, dt, n_tr, n_obs, s.esc_r, renorm, feats, finals)
    labels, z = match(s, feats, finals, refs)
    return feats, finals, labels


def main():
    rng = np.random.default_rng(4)
    s = systems.get("chua_I")
    n = 20000
    ics = s.sample_ball(n, rng, radius=3.0)
    out = {"system": "chua_I", "n": n, "radius": 3.0, "T_tr": 1000.0, "T_obs": 100.0, "runs": {}}
    for dt in (0.01, 0.005, 0.0025, 0.00125):
        # эталоны калибруются при том же шаге: копии состояния на аттракторе
        refs = [Reference("F0", "fp", point=np.zeros(3)), Reference("Fp", "fp", point=s.equilibria[1]),
                Reference("Fm", "fp", point=s.equilibria[2])]
        for name in ("HA_a", "HA_b"):
            st = s.known[name]
            f0, fi0, _ = run(s, st[None, :], dt, 100.0, 100.0, refs)
            cop = fi0[0] + 1e-3 * (1 + np.abs(fi0[0])) * rng.normal(size=(48, 3))
            f, fi, _ = run(s, cop, dt, 100.0, 100.0, refs)
            keep = (f[:, 0] < 0.5) & (f[:, F_LAM] > 0.02)
            r = Reference(name, "chaotic", mu=f[keep].mean(axis=0), sigma=f[keep].std(axis=0),
                          lam=float(np.median(f[keep, F_LAM])), n_cal=int(keep.sum()), example=fi0[0])
            refs.append(r)
        t0 = time.time()
        feats, finals, labels = run(s, ics, dt, 1000.0, 100.0, refs)
        rec = {}
        for lab, nm in [(LAB_ESCAPE, "inf"), (LAB_UNKNOWN, "unknown")] + [(k, r.name) for k, r in enumerate(refs)]:
            k_ = int((labels == lab).sum())
            rec[nm] = {"n": k_, "p": k_ / n, "ci95": wilson(k_, n)}
        rec["lam_HA"] = [round(float(r.lam), 4) for r in refs if r.kind == "chaotic"]
        rec["time_s"] = round(time.time() - t0, 1)
        out["runs"][str(dt)] = rec
        print(f"dt={dt}: " + ", ".join(f"{k}={v['p']:.4f}" for k, v in rec.items() if isinstance(v, dict))
              + f"  λ={rec['lam_HA']}  ({rec['time_s']} с)")
    (ROOT / "results" / "e04_chua_dt_convergence.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
