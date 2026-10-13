"""Воспроизведение табл. 1 обзора Guan & Xie (2025) для цепи Чуа (набор I)
и бассейнов на плоскости z = 0 (Wagemakers 2025; Guan & Xie 2025, рис. 1c).

Выход: results/e01_chua_table1.json, results/e01_chua_basin_z0.png
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import (LAB_ESCAPE, LAB_UNKNOWN, calibrate_reference,  # noqa: E402
                              fp_references_from_equilibria, match)
from hsearch.integrate import run_batch  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def main():
    rng = np.random.default_rng(1)
    s = systems.get("chua_I")
    refs = fp_references_from_equilibria(s, names=["F0", "Fp", "Fm"])
    for name in ("HA_a", "HA_b"):
        refs.append(calibrate_reference(s, name, s.known[name], rng))
    print("эталоны:", [(r.name, r.kind, round(r.lam, 3)) for r in refs])
    out = {"system": "chua_I", "params": s.params.tolist(), "dt": s.dt, "T_tr": s.T_tr, "T_obs": s.T_obs,
           "review_table1": {"r3": {"inf": 0.0605, "F0": 0.7488, "HA_blue": 0.0949, "HA_cyan": 0.0958},
                             "r5": {"inf": 0.2340, "F0": 0.4990, "HA_blue": 0.1352, "HA_cyan": 0.1318}},
           "ours": {}}
    for radius in (3.0, 5.0):
        for n in (10000, 100000):
            t0 = time.time()
            ics = s.sample_ball(n, rng, radius=radius)
            feats, finals = run_batch(s, ics)
            labels, z = match(s, feats, finals, refs)
            rec = {}
            for lab, nm in [(LAB_ESCAPE, "inf"), (LAB_UNKNOWN, "unknown")] + [(k, r.name) for k, r in enumerate(refs)]:
                k_ = int((labels == lab).sum())
                rec[nm] = {"n": k_, "p": k_ / n, "ci95": wilson(k_, n)}
            rec["time_s"] = round(time.time() - t0, 1)
            out["ours"][f"r{int(radius)}_n{n}"] = rec
            print(f"r={radius} n={n}: " + ", ".join(f"{k}={v['p']:.4f}" for k, v in rec.items() if k != "time_s")
                  + f"  ({rec['time_s']} с)")
    # бассейн на плоскости z = 0
    nx, ny = 600, 360
    xs = np.linspace(-10, 10, nx)
    ys = np.linspace(-4, 4, ny)
    X, Y = np.meshgrid(xs, ys)
    ics = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    t0 = time.time()
    feats, finals = run_batch(s, ics)
    labels, z = match(s, feats, finals, refs)
    print(f"бассейн {nx}x{ny}: {time.time() - t0:.1f} с; доли: " +
          ", ".join(f"{nm}={float((labels == lab).mean()):.3f}" for lab, nm in
                    [(LAB_ESCAPE, 'inf'), (LAB_UNKNOWN, 'unk')] + [(k, r.name) for k, r in enumerate(refs)]))
    img = labels.reshape(ny, nx)
    np.save(ROOT / "results" / "e01_chua_basin_z0_labels.npy", img)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors = {LAB_ESCAPE: "#bbbbbb", LAB_UNKNOWN: "#000000"}
    palette = {"F0": "#f28e2b", "Fp": "#e15759", "Fm": "#b07aa1", "HA_a": "#1f77b4", "HA_b": "#17becf"}
    keys = [LAB_ESCAPE, LAB_UNKNOWN] + list(range(len(refs)))
    cols = [colors.get(k, palette.get(refs[k].name, "#59a14f")) if k >= 0 else colors[k] for k in keys]
    lut = {k: i for i, k in enumerate(keys)}
    idx = np.vectorize(lut.get)(img)
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
    ax.imshow(idx, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap=ListedColormap(cols),
              vmin=-0.5, vmax=len(keys) - 0.5, aspect="auto", interpolation="nearest")
    ax.set_xlabel("x(0)"); ax.set_ylabel("y(0)")
    ax.set_title("Чуа, набор I: бассейны на плоскости z(0)=0 (серый — ∞, оранжевый — F0, синий/голубой — скрытые)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "e01_chua_basin_z0.png")
    out["basin_z0"] = {"nx": nx, "ny": ny, "x": [-10, 10], "y": [-4, 4],
                       "fractions": {nm: float((labels == lab).mean()) for lab, nm in
                                     [(LAB_ESCAPE, 'inf'), (LAB_UNKNOWN, 'unknown')] + [(k, r.name) for k, r in enumerate(refs)]}}
    (ROOT / "results" / "e01_chua_table1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("сохранено results/e01_chua_table1.json, results/e01_chua_basin_z0.png")


if __name__ == "__main__":
    main()
