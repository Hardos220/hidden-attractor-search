"""Проверка геометрических эвристик на системах с известными ответами:
- wangchen1: PP = (0, (1 ± sqrt(1-12a))/6, 0), a = -0.05  -> (0, 0.3775, 0), (0, -0.0442, 0)
- ne4:       PP = (0, 10/31, 0)
- se3:       PP отсутствуют
- chua_I:    нульклины и соединяющие кривые существуют; просто считаем число точек
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.baselines.geometric import connecting_curve_samples, nullcline_samples, perpetual_points  # noqa: E402


def main():
    rng = np.random.default_rng(6)
    for name in ("wangchen1", "ne4", "se3", "le1", "wangchen2"):
        s = systems.get(name)
        t0 = time.time()
        pp = perpetual_points(s, rng, n_starts=300)
        print(f"{name}: перпетуальные точки ({time.time() - t0:.1f} с):", np.round(pp, 4).tolist())
    s = systems.get("chua_I")
    t0 = time.time()
    for i in range(3):
        pts = nullcline_samples(s, rng, i, n=300)
        print(f"chua_I: нульклина x{i}'=0: {len(pts)} точек, пример {np.round(pts[0], 3) if len(pts) else None}")
    print(f"  ({time.time() - t0:.1f} с)")
    t0 = time.time()
    cc = connecting_curve_samples(s, rng, n=200, n_starts=1500)
    print(f"chua_I: соединяющие кривые: {len(cc)} точек ({time.time() - t0:.1f} с); пример:", np.round(cc[:3], 3).tolist())
    s2 = systems.get("ne4")
    cc2 = connecting_curve_samples(s2, rng, n=200, n_starts=1500)
    print(f"ne4: соединяющие кривые: {len(cc2)} точек; пример:", np.round(cc2[:3], 3).tolist())


if __name__ == "__main__":
    main()
