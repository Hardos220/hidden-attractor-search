"""Дымовой тест инфраструктуры: компиляция ядер, калибровка эталонов Чуа, выборка 2000 точек."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import build_reference_set, label_fractions  # noqa: E402
from hsearch.integrate import run_batch, feature_names  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    s = systems.get("chua_I")
    print("равновесия:", [np.round(e, 4) for e in s.equilibria])
    t0 = time.time()
    feats, finals = run_batch(s, np.array([s.known["HA_a"], s.known["HA_b"], s.known["F0"] + 1e-3]))
    print(f"компиляция + 3 траектории: {time.time() - t0:.1f} с")
    names = feature_names(s.dim)
    for row in feats:
        print({n: round(float(v), 4) for n, v in zip(names, row)})
    t0 = time.time()
    ics = s.sample_box(2000, rng)
    feats, finals = run_batch(s, ics)
    dt = time.time() - t0
    steps = 2000 * (s.n_tr + s.n_obs)
    print(f"2000 траекторий: {dt:.1f} с, {steps / dt / 1e6:.1f} млн шагов/с")
    t0 = time.time()
    refs, ics, feats, finals, labels, hist = build_reference_set(s, rng, n_sample=5000)
    print(f"эталоны: {[(r.name, r.kind, round(r.lam, 3)) for r in refs]}  ({time.time() - t0:.1f} с)")
    print("доли:", {k: round(v, 4) for k, v in label_fractions(labels, refs).items()})


if __name__ == "__main__":
    main()
