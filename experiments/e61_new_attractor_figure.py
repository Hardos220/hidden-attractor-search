"""Рисунок статьи: скрытый хаотический аттрактор двух связанных цепей Чуа II, рождённый связью (2026-10-13).

(a) цепь 1 в проекции (x, z) и цикл LC отдельной цепи; (b) цепь 2 и скрытые аттракторы HA_a, HA_b отдельной цепи;
(c) старший показатель Ляпунова режима в зависимости от силы связи (скан e58: рождение при eps ≈ 0.045, слияние с
синхронным циклом при eps ≈ 0.5). Пишет results/figures/fig31_new_attractor.{png,pdf} (RU) и
paper/chaos/figures/fig31_new_attractor.{png,pdf} (EN).

    rmq run --cpu 4 --ram 2 --time 10m -- python experiments/e61_new_attractor_figure.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import e50_batch_runner as runner  # noqa: E402
from hsearch.viz import INK2, MUTED, SERIES, save, setup  # noqa: E402

T = {"en": {"lc": "LC of the single circuit", "c1": "circuit 1", "ha": "HA$_a$, HA$_b$ of the single circuit",
            "c2": "circuit 2", "a": "(a) circuit 1", "b": "(b) circuit 2", "c": "(c) largest Lyapunov exponent",
            "eps": "coupling $\\epsilon$", "lam": "$\\lambda$", "per": "periodic", "cha": "chaotic"},
     "ru": {"lc": "цикл LC одной цепи", "c1": "цепь 1", "ha": "HA$_a$, HA$_b$ одной цепи",
            "c2": "цепь 2", "a": "(а) цепь 1", "b": "(б) цепь 2", "c": "(в) старший показатель Ляпунова",
            "eps": "сила связи $\\epsilon$", "lam": "$\\lambda$", "per": "периодический", "cha": "хаотический"}}


def main():
    import numba
    numba.set_num_threads(4)
    setup()
    var = {"coupled": "chua_II", "k": 2, "eps": 0.1, "coord": 0, "ha": ["HA_a", "HA_b"], "products": True, "alleq": True}
    s, refs, tidx, kidx, _ = runner.get_problem(var, None)
    r = next(q for q in refs if q.name == "P_LC-HA_a")
    tr = s.traj(np.asarray(r.example, float), n=120000)[40000:]
    b = runner.get_problem({"system": "chua_II"}, None)
    ref_b = {q.name: q for q in b[1]}
    single = {nm: b[0].traj(np.asarray(ref_b[nm].example, float), n=80000)[20000:] for nm in ("LC", "HA_a", "HA_b")}
    scan = []
    for tag in ("_down", "_up"):
        scan += json.loads((ROOT / "results" / f"e58_eps_scan{tag}.json").read_text(encoding="utf-8"))["points"]
    seen, pts = set(), []
    for q in sorted(scan, key=lambda q: q["eps"]):
        if q["eps"] not in seen:
            seen.add(q["eps"]); pts.append(q)
    for lang, out in (("en", ROOT / "paper" / "chaos" / "figures"), ("ru", ROOT / "results" / "figures")):
        t = T[lang]
        sc = 1.0 if lang == "en" else 1.45
        fig, ax = plt.subplots(1, 3, figsize=(6.69 * sc, 2.25 * sc))
        fs = 7 if lang == "en" else 9
        a = ax[0]
        a.plot(single["LC"][:, 0], single["LC"][:, 2], "-", color=MUTED, lw=0.7, label=t["lc"])
        a.plot(tr[:, 0], tr[:, 2], "-", color=SERIES[0], lw=0.35, alpha=0.85, label=t["c1"])
        a.set_xlabel("$x_1$", fontsize=fs); a.set_ylabel("$z_1$", fontsize=fs); a.set_title(t["a"], fontsize=fs + 1)
        a.legend(fontsize=fs - 1, loc="upper right")
        a = ax[1]
        for nm, c in (("HA_a", MUTED), ("HA_b", "#bbbbb6")):
            a.plot(single[nm][:, 0], single[nm][:, 2], "-", color=c, lw=0.35, label=t["ha"] if nm == "HA_a" else None)
        a.plot(tr[:, 3], tr[:, 5], "-", color=SERIES[1], lw=0.25, alpha=0.85, label=t["c2"])
        a.set_xlabel("$x_2$", fontsize=fs); a.set_ylabel("$z_2$", fontsize=fs); a.set_title(t["b"], fontsize=fs + 1)
        a.legend(fontsize=fs - 1, loc="upper right")
        a = ax[2]
        xs, lam, kind = [], [], []
        for q in pts:
            g = q["regimes"].get("P_LC-HA_a") or {}
            if g.get("status") == "ok" and g.get("lam") is not None:
                xs.append(q["eps"]); lam.append(g["lam"]); kind.append(g["kind"])
        xs, lam = np.array(xs), np.array(lam)
        cha = np.array([k == "chaotic" for k in kind])
        a.plot(xs, lam, "-", color=INK2, lw=0.6)
        a.plot(xs[cha], lam[cha], "o", color=SERIES[0], ms=2.5, label=t["cha"])
        a.plot(xs[~cha], lam[~cha], "s", color=SERIES[1], ms=2.5, label=t["per"])
        a.axhline(0, color=MUTED, lw=0.6, ls="--")
        a.set_xlabel(t["eps"], fontsize=fs); a.set_ylabel(t["lam"], fontsize=fs); a.set_title(t["c"], fontsize=fs + 1)
        a.legend(fontsize=fs - 1, loc="upper left")
        for a in ax:
            a.tick_params(labelsize=fs - 1)
        fig.tight_layout()
        save(fig, out / "fig31_new_attractor")
    print("ok")


if __name__ == "__main__":
    main()
