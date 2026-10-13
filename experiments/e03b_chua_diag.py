"""Диагностика расхождения e03/e04: что происходит с копиями на хаотическом множестве Чуа при dt=0.01.
Печатает по окнам T=100: λ, флаг FP, std_max, |mean| для 6 копий, до T=3000; затем то же при dt=0.0025.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.integrate import F_ESC, F_FP, F_LAM, feature_slices, nfeat, run_batch  # noqa: E402


def main():
    rng = np.random.default_rng(3)
    s = systems.get("chua_I")
    feats, finals = run_batch(s, s.known["HA_a"][None, :], n_tr=20000, n_obs=10000)
    state = finals[0]
    sl = feature_slices(3)
    for dt in (0.01, 0.0025):
        n_obs = int(round(100.0 / dt))
        renorm = int(round(0.1 / dt))
        x = state + 1e-3 * (1 + np.abs(state)) * rng.normal(size=(6, 3))
        print(f"=== dt={dt}")
        rows = []
        for w in range(30):
            f = np.zeros((6, nfeat(3))); fi = np.zeros((6, 3))
            s.batch(x, s.params, dt, 0, n_obs, s.esc_r, renorm, f, fi)
            rows.append([f"{f[j, F_LAM]:+.3f}" for j in range(6)])
            if w % 5 == 4:
                print(f"T={(w + 1) * 100:5d}  λ:", " ".join(f"{f[j, F_LAM]:+.3f}" for j in range(6)),
                      " std_max:", " ".join(f"{f[j, sl['std']].max():.2f}" for j in range(6)),
                      " |mean|:", " ".join(f"{np.linalg.norm(f[j, sl['mean']]):.2f}" for j in range(6)),
                      " fp:", "".join(str(int(f[j, F_FP])) for j in range(6)))
            x = fi
        lam = np.array([[float(v) for v in r] for r in rows])
        print("доля окон с λ<0.02 по копиям:", np.round((lam < 0.02).mean(axis=0), 2))


if __name__ == "__main__":
    main()
