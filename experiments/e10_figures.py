"""Фигуры этапа 1 (из результатов e01–e09) -> results/figures/*.png|pdf

1. chua_basin_z0        — бассейны Чуа I на плоскости z=0;
2. chua_survival        — кривые выживания хаотического множества при dt=0.01 и 0.0025;
3. chua_dt_convergence  — доли бассейнов против шага интегрирования;
4. table1_comparison    — доли в шарах: обзор / Python / Attractors.jl;
5. uniform_costs        — N_0.9 равномерного перебора по системам (лог-шкала);
6. synthetic_family     — измеренная доля против аналитической (уровень D);
7. classic_vs_uniform   — стоимость классических эвристик против перебора;
8. attractor_gallery    — проекции скрытых аттракторов уровня A.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import viz  # noqa: E402
from hsearch.viz import ESCAPE_GREY, INK, INK2, MUTED, NICE_SYS, SERIES, color_for, nice_label, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fig_chua_basin():
    d = load("e01_chua_table1.json")
    lab_path = RES / "e01_chua_basin_z0_labels.npy"
    if d is None or not lab_path.exists():
        return
    img = np.load(lab_path)
    b = d["basin_z0"]
    names = ["F0", "Fp", "Fm", "HA_a", "HA_b"]
    keys = [-1, -2] + list(range(len(names)))
    cols = [ESCAPE_GREY, INK] + [color_for(n) for n in names]
    lut = {k: i for i, k in enumerate(keys)}
    idx = np.vectorize(lambda v: lut.get(int(v), 1))(img)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.imshow(idx, origin="lower", extent=[b["x"][0], b["x"][1], b["y"][0], b["y"][1]],
              cmap=ListedColormap(cols), vmin=-0.5, vmax=len(keys) - 0.5, aspect="auto", interpolation="nearest")
    ax.grid(False)
    ax.set_xlabel("x(0)"); ax.set_ylabel("y(0)")
    ax.set_title("Цепь Чуа, набор I: бассейны притяжения на плоскости z(0) = 0")
    from matplotlib.patches import Patch
    handles = [Patch(color=ESCAPE_GREY, label="уход на бесконечность"), Patch(color=color_for("F0"), label="равновесие F0"),
               Patch(color=color_for("HA_a"), label="скрытый аттрактор A"), Patch(color=color_for("HA_b"), label="скрытый аттрактор B")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9, frameon=True)
    save(fig, FIG / "fig01_chua_basin_z0")


def fig_chua_survival():
    d = load("e03_chua_lifetime.json")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, key in enumerate(("survival_dt0.01", "survival_dt0.0025")):
        r = d.get(key)
        if not r:
            continue
        ax.plot(r["T"], r["alive"], color=SERIES[i], label=f"RK4, dt = {r['dt']}, N = {r['n_copies']}")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("t"); ax.set_ylabel("доля копий, остающихся хаотическими")
    ax.set_ylim(0, 1.05)
    ax.set_title("Чуа I: «выживание» на скрытом хаотическом множестве (шум 1e-3)")
    ax.text(0.02, 0.08, "DOP853, rtol = 1e-11: хаос сохраняется до t = 4000", transform=ax.transAxes, fontsize=8, color=INK2)
    ax.legend(loc="center right", fontsize=8)
    save(fig, FIG / "fig02_chua_survival")


def fig_chua_dt():
    d = load("e04_chua_dt_convergence.json")
    if d is None:
        return
    runs = d["runs"]
    dts = sorted((float(k) for k in runs), reverse=True)
    series = [("inf", "уход на бесконечность"), ("F0", "равновесие F0"), ("HA_a", "скрытый аттрактор A"),
              ("HA_b", "скрытый аттрактор B"), ("unknown", "не сопоставлено")]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (key, lab) in enumerate(series):
        y = [runs[str(dt)][key]["p"] for dt in dts]
        lo = [runs[str(dt)][key]["ci95"][0] for dt in dts]
        hi = [runs[str(dt)][key]["ci95"][1] for dt in dts]
        c = color_for(key) if key != "unknown" else INK
        ax.plot(dts, y, marker="o", color=c, label=lab)
        ax.fill_between(dts, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("шаг интегрирования dt (RK4)"); ax.set_ylabel("доля начальных условий (шар r = 3)")
    ax.set_title("Чуа I: сходимость долей бассейнов по шагу (T_tr = 1000, n = 20000)")
    ax.legend(fontsize=8, ncol=2)
    save(fig, FIG / "fig03_chua_dt_convergence")


def fig_table1():
    d = load("e01_chua_table1.json"); j = load("julia_chua_attractors.json"); e4 = load("e04_chua_dt_convergence.json")
    if d is None:
        return
    cats = ["∞", "F0", "HA (1)", "HA (2)"]
    def ours(key):
        r = d["ours"].get(key)
        return [r["inf"]["p"], r["F0"]["p"], r["HA_a"]["p"], r["HA_b"]["p"]] if r else None
    def julia(key):
        if j is None or key not in j:
            return None
        f = j[key]["fractions"]; a = j[key]["attractors"]
        # сопоставление меток: единственная точка -> F0; -1 -> уход; остальные -> HA
        fp = [k for k, v in a.items() if v["n"] <= 3]
        ha = [k for k in a if k not in fp]
        return [f.get("-1", 0.0), sum(f.get(k, 0.0) for k in fp)] + [f.get(k, 0.0) for k in ha[:2]]
    review = {"r3": [0.0605, 0.7488, 0.0949, 0.0958], "r5": [0.2340, 0.4990, 0.1352, 0.1318]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (key, title) in zip(axes, [("r3", "шар радиуса 3"), ("r5", "шар радиуса 5")]):
        groups = [("обзор Guan & Xie (2025)", review[key]), ("Python, RK4 dt = 0.0025", ours(f"{key}_n100000") or ours(f"{key}_n10000")),
                  ("Attractors.jl, Vern9", julia(key))]
        groups = [(n, v) for n, v in groups if v is not None]
        w = 0.8 / len(groups)
        x = np.arange(len(cats))
        for gi, (name, vals) in enumerate(groups):
            ax.bar(x + (gi - (len(groups) - 1) / 2) * w, vals, width=w * 0.92, color=SERIES[gi], label=name)
            for xi, v in zip(x, vals):
                ax.text(xi + (gi - (len(groups) - 1) / 2) * w, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        rotation=90, fontsize=6.5, color=INK2)
        ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.92)
    axes[0].set_ylabel("доля начальных условий")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Чуа I: доли бассейнов при равномерной выборке в шаре — три источника", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, FIG / "fig04_table1_comparison")


def fig_uniform():
    d = load("e02_uniform.json")
    if d is None:
        return
    rows = []
    for name, rec in d.items():
        for h, v in rec["hidden"].items():
            if v["p"] > 0:
                rows.append((nice_label(name, h), v["p"], v["N_0.9"]))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.5))
    y = np.arange(len(rows))
    ax.barh(y, [r[2] for r in rows], color=SERIES[0], height=0.62)
    for yi, r in zip(y, rows):
        ax.text(r[2] * 1.15, yi, f"p = {r[1]:.3g}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("N_0.9 — число интегрирований для вероятности обнаружения 0.9 (равномерный перебор)")
    ax.set_title("Базовая стоимость равномерного перебора")
    ax.grid(axis="y", visible=False)
    save(fig, FIG / "fig05_uniform_costs")


def fig_synthetic():
    d = load("e09_synthetic.json")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ms = sorted(set(c["m"] for c in d["cases"]))
    for i, m in enumerate(ms):
        cs = [c for c in d["cases"] if c["m"] == m]
        x = [c["p_analytic"] for c in cs]; y = [c["p_measured"] for c in cs]
        err = np.array([[c["p_measured"] - c["ci95"][0], c["ci95"][1] - c["p_measured"]] for c in cs]).T
        ax.errorbar(x, y, yerr=err, fmt="o", color=SERIES[i], label=f"m = {m} (n = {m + 3})", capsize=3)
    lim = [min(c["p_analytic"] for c in d["cases"]) / 2, max(c["p_measured"] for c in d["cases"]) * 2]
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", label="p_изм = p_аналит")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("аналитическая доля бассейна p"); ax.set_ylabel("измеренная доля (перебор, 95% ДИ)")
    ax.set_title(f"Синтетическое семейство D1: вложение {d['base'].upper()} в ℝ³⁺ᵐ")
    ax.legend(fontsize=8)
    save(fig, FIG / "fig06_synthetic_family")


def fig_classic():
    d = load("e08_classic.json")
    rec_t = load("e08b_recurrence.json") or {}
    if d is None:
        return
    rows = []
    for name, rec in d.items():
        for t in rec["targets"]:
            pu = rec["p_uniform"].get(t) or 0.0
            nu = np.ceil(np.log(0.1) / np.log1p(-pu)) if pu > 0 else np.nan
            best = max((v for k, v in rec["CVS"].items() if isinstance(v, dict)), key=lambda v: v["p"])
            rows.append((nice_label(name, t), nu, best["N_0.9"], rec["CC"]["N_0.9"],
                         rec["PP"]["hits"], rec["PP"]["n_points"]))
    rows.sort(key=lambda r: -(r[1] if np.isfinite(r[1]) else 1e9))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 2.2))
    y = np.arange(len(rows)); w = 0.26
    labels = ["равномерный перебор", "нульклины (CVS, лучшая)", "соединяющие кривые (CC)"]
    allvals = []
    for i in range(3):
        vals = np.array([r[1 + i] if (r[1 + i] is not None and np.isfinite(r[1 + i])) else np.nan for r in rows], float)
        allvals += [v for v in vals if np.isfinite(v)]
        ax.barh(y + (1 - i) * w, vals, height=w * 0.9, color=SERIES[i], label=labels[i])
        for yi, v in zip(y, vals):
            if not np.isfinite(v):
                ax.text(1.05, yi + (1 - i) * w, "не найдено", fontsize=7, va="center", color=INK2)
    xmax = max(allvals) if allvals else 10
    for yi, r in zip(y, rows):
        txt = f"PP: {r[4]}/{r[5]}" if r[5] > 0 else "PP: нет точек"
        ax.text(xmax * 1.6, yi, txt, fontsize=8, va="center", color=INK2)
    ax.set_xlim(1, xmax * 6)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("число интегрирований до обнаружения с вероятностью 0.9")
    ax.set_title("Классические эвристики против перебора (уровень A); справа — перпетуальные точки: попаданий/всего")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    save(fig, FIG / "fig07_classic_vs_uniform")


def fig_gallery():
    from hsearch import systems
    names = [("chua_I", "HA_a", 0, 2), ("ne4", "HA", 0, 2), ("wangchen1", "HA", 0, 2), ("wei_ne2", "HA", 0, 2),
             ("wangchen2", "HA", 0, 2), ("se3", "HA", 0, 2), ("se11", "HA", 0, 2), ("le1", "HA", 0, 2),
             ("rabinovich_100", "HA", 0, 2)]
    fig, axes = plt.subplots(3, 3, figsize=(9, 8.5))
    for ax, (name, key, i, j) in zip(axes.ravel(), names):
        s = systems.get(name)
        n = s.n_tr + min(s.n_obs, int(200.0 / s.dt))
        tr = s.traj(s.known[key], n=n)
        tr = tr[s.n_tr:]
        tr = tr[~np.isnan(tr).any(axis=1)]
        ax.plot(tr[:, i], tr[:, j], color=SERIES[0], lw=0.35)
        ax.set_title(NICE_SYS.get(name, name), fontsize=9); ax.set_xlabel("xyz"[i], fontsize=8); ax.set_ylabel("xyz"[j], fontsize=8)
        ax.tick_params(labelsize=7); ax.grid(False)
    fig.suptitle("Скрытые аттракторы эталонного набора (уровень A), проекции на плоскость (x, z)", fontsize=11)
    fig.tight_layout()
    save(fig, FIG / "fig08_attractor_gallery")


def main():
    viz.setup()
    for f in (fig_chua_basin, fig_chua_survival, fig_chua_dt, fig_table1, fig_uniform, fig_synthetic, fig_classic, fig_gallery):
        try:
            f()
            print("ok:", f.__name__)
        except Exception as e:  # noqa: BLE001
            print("skip:", f.__name__, "->", repr(e))


if __name__ == "__main__":
    main()
