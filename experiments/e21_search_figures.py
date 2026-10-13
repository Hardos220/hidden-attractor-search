"""Фигуры этапа 2: кривые P(N) по стратегиям для каждой системы из results/e20_search_*.json
и сводная диаграмма коэффициента выигрыша G_0.9 = N_0.9(перебор)/N_0.9(стратегия).
Выход: results/figures/fig2X_*.png|pdf
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import viz  # noqa: E402
from hsearch.viz import INK2, MUTED, SERIES, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
LABELS = {"uniform": "равномерный перебор", "ams": "адаптивное расщепление (AMS)", "novelty_me": "новизна / MAP-Elites",
          "active_boundary": "активное обучение границ", "de_zelinka": "DE Зелинки (оракул размера)",
          "ams_cvs": "AMS + старт с нульклин", "novelty_me_cvs": "MAP-Elites + старт с нульклин"}
ORDER = ["uniform", "ams", "novelty_me", "active_boundary", "de_zelinka", "ams_cvs", "novelty_me_cvs"]


def load_all():
    data = {}
    for p in sorted(RES.glob("e20_search_*.json")):
        if "_smoke" in p.name:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        data[d["_meta"]["system"]] = d
    return data


def fig_pn(data):
    names = list(data)
    if not names:
        return
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        d = data[name]
        for i, sname in enumerate(ORDER):
            if sname not in d:
                continue
            r = d[sname]
            ax.step(r["pn_grid"], r["pn"], where="post", color=SERIES[i], label=LABELS.get(sname, sname))
        ax.axhline(0.9, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_ylim(0, 1.02)
        ax.set_title(f"{name}: цели {', '.join(d['_meta']['targets'])}", fontsize=9)
        ax.set_xlabel("N (интегрирований)", fontsize=8); ax.set_ylabel("P(N)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Вероятность обнаружения цели как функция бюджета (R запусков на стратегию)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    save(fig, FIG / "fig21_pn_curves")


def fig_gain(data):
    rows = []
    for name, d in data.items():
        nu = d.get("uniform", {}).get("N_0.9")
        if not nu:
            continue
        for sname in ORDER:
            if sname in d and sname != "uniform" and d[sname].get("N_0.9"):
                rows.append((name, sname, nu / d[sname]["N_0.9"], d[sname]["success_rate"]))
    if not rows:
        return
    names = sorted(set(r[0] for r in rows))
    strategies = [s for s in ORDER if any(r[1] == s for r in rows)]
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 3, 4))
    w = 0.8 / len(strategies)
    x = np.arange(len(names))
    for i, sname in enumerate(strategies):
        vals = [next((r[2] for r in rows if r[0] == n and r[1] == sname), np.nan) for n in names]
        ax.bar(x + (i - (len(strategies) - 1) / 2) * w, vals, width=w * 0.9, color=SERIES[ORDER.index(sname)],
               label=LABELS.get(sname, sname))
    ax.axhline(1.0, color=MUTED, lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("G_0.9 = N_0.9(перебор) / N_0.9(стратегия)")
    ax.set_title("Коэффициент выигрыша стратегий относительно перебора")
    ax.legend(fontsize=8)
    save(fig, FIG / "fig22_gain")


def main():
    viz.setup()
    data = load_all()
    fig_pn(data)
    fig_gain(data)
    print("системы:", list(data))


if __name__ == "__main__":
    main()
