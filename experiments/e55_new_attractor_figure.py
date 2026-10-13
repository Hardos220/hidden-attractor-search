"""Рабочий рисунок (не для статьи): скрытый хаотический аттрактор двух связанных цепей Чуа II, рождённый связью.

Одна цепь на большом цикле LC, другая — в хаотическом режиме, которого нет у отдельной цепи. Панели: (а) цепь 1
в проекции (x, z) и цикл LC отдельной цепи; (б) цепь 2 и скрытые аттракторы HA_a, HA_b отдельной цепи;
(в) временные ряды x1, x2; (г) старший показатель Ляпунова режима в зависимости от силы связи (перепись e55).
Пишет results/figures/e55_new_attractor.png.
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


def main():
    import numba
    numba.set_num_threads(4)
    setup()
    eps = 0.1
    var = {"coupled": "chua_II", "k": 2, "eps": eps, "coord": 0, "ha": ["HA_a", "HA_b"], "products": True, "alleq": True}
    s, refs, tidx, kidx, _ = runner.get_problem(var, None)
    r = next(q for q in refs if q.name == "P_LC-HA_a")
    tr = s.traj(np.asarray(r.example, float), n=120000)[40000:]
    base = runner.get_problem({"system": "chua_II"}, None)
    b = base[0]
    ref_b = {q.name: q for q in base[1]}
    single = {nm: b.traj(np.asarray(ref_b[nm].example, float), n=80000)[20000:] for nm in ("LC", "HA_a", "HA_b")}
    fig, ax = plt.subplots(2, 2, figsize=(9.5, 7.4))
    a = ax[0, 0]
    a.plot(single["LC"][:, 0], single["LC"][:, 2], "-", color=MUTED, lw=0.6, label="цикл LC отдельной цепи")
    a.plot(tr[:, 0], tr[:, 2], "-", color=SERIES[0], lw=0.4, alpha=0.8, label="цепь 1 на новом аттракторе")
    a.set_xlabel("x₁"); a.set_ylabel("z₁"); a.legend(fontsize=7, loc="upper right"); a.set_title("(а) цепь 1", fontsize=10)
    a = ax[0, 1]
    for nm, c in (("HA_a", MUTED), ("HA_b", "#bbbbb6")):
        a.plot(single[nm][:, 0], single[nm][:, 2], "-", color=c, lw=0.4, label=f"{nm} отдельной цепи")
    a.plot(tr[:, 3], tr[:, 5], "-", color=SERIES[1], lw=0.3, alpha=0.8, label="цепь 2 на новом аттракторе")
    a.set_xlabel("x₂"); a.set_ylabel("z₂"); a.legend(fontsize=7, loc="upper right"); a.set_title("(б) цепь 2", fontsize=10)
    a = ax[1, 0]
    t = np.arange(len(tr)) * s.dt
    m = t < 150
    a.plot(t[m], tr[m, 0], color=SERIES[0], lw=0.8, label="x₁")
    a.plot(t[m], tr[m, 3], color=SERIES[1], lw=0.8, label="x₂")
    a.set_xlabel("t"); a.legend(fontsize=7); a.set_title("(в) временные ряды", fontsize=10)
    a = ax[1, 1]
    cen = json.loads((ROOT / "results" / "e55_coupled_chua_census_v2.json").read_text(encoding="utf-8"))
    xs, lam, kind = [], [], []
    for e, rr in sorted(cen.items(), key=lambda kv: float(kv[0])):
        g = next((q for q in rr["regimes"] if q["name"] == "P_LC-HA_a"), None)
        if g:
            xs.append(float(e)); lam.append(g["lam"]); kind.append(g["kind"])
    a.plot(xs, lam, "o-", color=SERIES[0])
    for x_, l_, k_ in zip(xs, lam, kind):
        a.annotate("хаос" if k_ == "chaotic" else "цикл", (x_, l_), textcoords="offset points", xytext=(4, 4), fontsize=7,
                   color=INK2)
    a.axhline(0, color=MUTED, lw=0.8, ls="--")
    a.set_xscale("log"); a.set_xlabel("сила связи ε"); a.set_ylabel("старший показатель Ляпунова")
    a.set_title("(г) рождение хаоса при усилении связи", fontsize=10)
    fig.suptitle(f"Две цепи Чуа II, связь ε = {eps} по x: скрытый аттрактор «цикл + хаос»", fontsize=11)
    fig.tight_layout()
    save(fig, ROOT / "results" / "figures" / "e55_new_attractor")
    print("ok")


if __name__ == "__main__":
    main()
