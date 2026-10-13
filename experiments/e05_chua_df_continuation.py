"""Базовый метод: продолжение по малому параметру от гармонически линеаризованной системы
(метод описывающих функций Леонова–Кузнецова) для цепи Чуа, набор I.

Система: dx/dt = P0 x + q eps phi(r^T x), P0 = P + k q r^T, phi(r) = psi(r) - k r,
psi(r) = (m0 - m1)/2 (|r+1| - |r-1|), q = (-alpha, 0, 0)^T, r = (1, 0, 0)^T.
eps: 0.1, 0.2, ..., 1.0; стартовая точка (a0, a0 (m1+1+k), a0 (alpha (m1+k) - omega0^2)/alpha),
a0 = 5.8576, omega0 = 2.0392, k = 0.2098 (Guan & Xie 2025, пример 6).
Стоимость — 10 интегрирований длиной T = 500 (+ окно наблюдения).
Выход: results/e05_chua_df_continuation.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import calibrate_reference, fp_references_from_equilibria, match  # noqa: E402
from hsearch.integrate import F_LAM, make_flow_kernels, nfeat, run_batch  # noqa: E402


@njit(cache=False)
def chua_df_rhs(x, p, out):
    # p = [alpha, beta, gamma, m0, m1, k, eps]
    a, b, g, m0, m1, k, eps = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    r = x[0]
    psi = 0.5 * (m0 - m1) * (abs(r + 1.0) - abs(r - 1.0))
    phi = psi - k * r
    # P0 x: P = [[-a(m1+1), a, 0], [1, -1, 1], [0, -b, -g]], P0 = P + k q r^T, q = (-a, 0, 0)
    out[0] = -a * (m1 + 1.0 + k) * x[0] + a * x[1] - a * eps * phi
    out[1] = x[0] - x[1] + x[2]
    out[2] = -b * x[1] - g * x[2]


def main():
    rng = np.random.default_rng(5)
    s = systems.get("chua_I")
    a, b, g, m0, m1 = s.params
    omega0, k, a0 = 2.0392, 0.2098, 5.8576
    batch, traj = make_flow_kernels(chua_df_rhs, 3)
    x = np.array([a0, a0 * (m1 + 1.0 + k), a0 * (a * (m1 + k) - omega0 ** 2) / a])
    print("стартовая точка:", np.round(x, 4), " (в обзоре: 5.8576, 0.3694, -8.3686)")
    refs = fp_references_from_equilibria(s, names=["F0", "Fp", "Fm"])
    for name in ("HA_a", "HA_b"):
        refs.append(calibrate_reference(s, name, s.known[name], rng))
    T_step, dt = 500.0, s.dt
    n_tr = int(T_step / dt); n_obs = int(100.0 / dt)
    log = []
    for j, eps in enumerate(np.linspace(0.1, 1.0, 10)):
        p = np.array([a, b, g, m0, m1, k, eps])
        feats = np.zeros((1, nfeat(3))); finals = np.zeros((1, 3))
        batch(x[None, :], p, dt, n_tr, n_obs, s.esc_r, s.renorm_every, feats, finals)
        rec = {"eps": round(float(eps), 2), "start": x.tolist(), "end": finals[0].tolist(),
               "escaped": bool(feats[0, 0] > 0.5), "lam": round(float(feats[0, F_LAM]), 4),
               "rmax": round(float(feats[0, 4]), 3)}
        if eps >= 1.0 - 1e-12:
            labels, z = match(s, feats, finals, refs)
            rec["label"] = refs[labels[0]].name if labels[0] >= 0 else int(labels[0])
            rec["z"] = round(float(z[0]), 2)
        log.append(rec)
        print(f"eps={eps:.1f}: ушла={rec['escaped']}, λ={rec['lam']:+.4f}, rmax={rec['rmax']}, конец={np.round(finals[0], 4)}"
              + (f", метка={rec['label']} (z={rec['z']})" if 'label' in rec else ""))
        x = finals[0].copy()
        if rec["escaped"]:
            break
    out = {"system": "chua_I", "omega0": omega0, "k": k, "a0": a0, "T_step": T_step, "steps": log,
           "n_integrations": len(log), "found": log[-1].get("label")}
    (ROOT / "results" / "e05_chua_df_continuation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("результат:", out["found"], "за", out["n_integrations"], "интегрирований")


if __name__ == "__main__":
    main()
