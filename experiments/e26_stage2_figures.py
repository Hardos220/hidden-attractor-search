"""Рисунки и таблицы этапа 2 по results/e20_search_*.json.

fig21_pn_levelAB      — P(N) для редких целей уровней A/B;
fig22_gain_levelAB    — G_0.9 с 95 % бутстрап-интервалами;
fig23_scaling_p       — N_0.9 против p (D2 и D1), подгонка показателя s;
fig24_scaling_n       — N_0.9 против размерности n при p = 1e-4 (D2, D1);
fig25_ablation        — карта G по вариантам портфеля и системам;
results/e26_stage2_table.md — сводная таблица.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import viz  # noqa: E402
from hsearch.viz import INK, INK2, MUTED, SERIES, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"

LABEL = {
    "uniform": "равномерный перебор", "de_zelinka": "DE Зелинки (оракул размера)", "ams": "AMS (1-е поколение)",
    "novelty_me": "MAP-Elites (1-е поколение)", "sus_tesc": "расщепление: время ухода", "sus_rarity": "расщепление: редкость",
    "sus": "расщепление: длит. переходного процесса", "portfolio_no_sus": "активное обучение + 20 % равн.",
    "portfolio_no_ab": "расщепление + 20 % равн.", "portfolio_eps0": "портфель без равн. доли",
    "portfolio": "портфель (предлагаемый метод)", "portfolio_cvs": "портфель + нульклины",
}
COLOR = {"uniform": MUTED, "de_zelinka": SERIES[3], "ams": SERIES[4], "novelty_me": SERIES[5], "sus_tesc": SERIES[6],
         "sus_rarity": SERIES[7], "sus": SERIES[1], "portfolio_no_sus": SERIES[2], "portfolio_no_ab": SERIES[1],
         "portfolio_eps0": SERIES[4], "portfolio": SERIES[0], "portfolio_cvs": SERIES[5]}
LEVEL_AB = ["chua_I", "chua_II", "se11", "le1", "jafari2026_lc", "huynh2019", "jafari2026_chaos"]
TITLE = {"chua_I": "Чуа I", "chua_II": "Чуа II", "se11": "SE11", "le1": "LE1", "jafari2026_lc": "Jafari 2026 (цикл)",
         "huynh2019": "отображение Хуиня", "jafari2026_chaos": "Jafari 2026 (хаос)"}


def load_e50(variant):
    """Сводки расчёта e50 (results/e50/<батч>/summary/<вариант>.json) по всем батчам; если стратегия есть
    в нескольких батчах, берётся серия с большим R. None, если e50 этого варианта нет."""
    merged = {}
    for sp in sorted((RES / "e50").glob(f"*/summary/{variant}.json")):
        d = json.loads(sp.read_text(encoding="utf-8"))
        for k, v in d.items():
            if k == "_meta":
                merged.setdefault("_meta", v)
            elif isinstance(v, dict) and "found_at" in v and len(v["found_at"]) > len(merged.get(k, {}).get("found_at", [])):
                merged[k] = v
    return merged if len(merged) > 1 else None


def load(system, prefer_r50=True):
    """Результаты бенчмарка: по умолчанию расчёт e50 (R = 100 для уровней A/B, синтетики и абляции);
    без него — прежний e20 с повтором R = 50 (недостающие стратегии дополняются из прогона с R = 20).
    HSEARCH_DATA=e20 — принудительно прежние данные."""
    import os
    if os.environ.get("HSEARCH_DATA", "e50") == "e50":
        d = load_e50(system)
        if d:
            return d
    p = RES / f"e20_search_{system}.json"
    p50 = RES / f"e20_search_{system}_R50.json"
    if prefer_r50 and p50.exists():
        d50 = json.loads(p50.read_text(encoding="utf-8"))
        d50 = {k: v for k, v in d50.items() if k == "_meta" or "found_at" in v}
        if p.exists():
            d20 = json.loads(p.read_text(encoding="utf-8"))
            for k, v in d20.items():
                if k != "_meta" and k not in d50 and "found_at" in v:
                    v = dict(v); v["_R"] = 20
                    d50[k] = v
        return d50 if len(d50) > 1 else None
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    d = {k: v for k, v in d.items() if k == "_meta" or "found_at" in v}
    return d if len(d) > 1 else None


def q09(fa):
    a = np.array([np.inf if f is None else f for f in fa], float)
    return float(np.quantile(a, 0.9))


def gain_ci(fa_u, fa_s, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.array([np.inf if f is None else f for f in fa_u], float)
    b = np.array([np.inf if f is None else f for f in fa_s], float)
    g = []
    for _ in range(n_boot):
        qa = np.quantile(rng.choice(a, len(a)), 0.9)
        qb = np.quantile(rng.choice(b, len(b)), 0.9)
        g.append(qa / qb if np.isfinite(qb) and qb > 0 else 0.0)
    qa, qb = q09(fa_u), q09(fa_s)
    point = qa / qb if np.isfinite(qb) and qb > 0 else 0.0
    lo, hi = np.percentile(g, [2.5, 97.5])
    return point, float(lo), float(hi)


def fig_pn():
    sysl = [s for s in LEVEL_AB if load(s)]
    if not sysl:
        return
    show = ["uniform", "de_zelinka", "sus", "portfolio_no_sus", "portfolio"]
    ncol = 3
    nrow = int(np.ceil(len(sysl) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow), squeeze=False)
    for ax, name in zip(axes.ravel(), sysl):
        d = load(name)
        for sname in show:
            if sname in d:
                r = d[sname]
                ax.step(r["pn_grid"], r["pn"], where="post", color=COLOR[sname], label=LABEL[sname],
                        lw=2.4 if sname == "portfolio" else 1.6)
        ax.axhline(0.9, color=MUTED, lw=0.8, ls=":")
        ax.set_xscale("log"); ax.set_ylim(0, 1.03)
        tg = {"chua_I": "оба хаотических", "chua_II": "оба хаотических", "jafari2026_lc": "цикл"}.get(name, "скрытый аттрактор")
        ax.set_title(f"{TITLE.get(name, name)}: {tg}", fontsize=9)
        ax.set_xlabel("N, интегрирований", fontsize=8); ax.set_ylabel("P(N)", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(sysl):]:
        ax.axis("off")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Вероятность обнаружения скрытого аттрактора как функция бюджета (R = 100 запусков)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    save(fig, FIG / "fig21_pn_levelAB")


def fig_gain_and_table():
    rows = []
    strategies = ["de_zelinka", "ams", "novelty_me", "sus_tesc", "sus_rarity", "sus", "portfolio_no_sus",
                  "portfolio_no_ab", "portfolio_eps0", "portfolio", "portfolio_cvs"]
    table = ["| Система | p | стратегия | успех | N_0.9 | G_0.9 [95 % ДИ] |", "|---|---|---|---|---|---|"]
    for name in LEVEL_AB:
        d = load(name)
        if not d or "uniform" not in d:
            continue
        fa_u = d["uniform"]["found_at"]
        nu = q09(fa_u)
        table.append(f"| {TITLE.get(name, name)} | — | равномерный перебор | {d['uniform']['success_rate']:.2f} | {nu:.0f} | 1 |")
        for sname in strategies:
            if sname not in d:
                continue
            g, lo, hi = gain_ci(fa_u, d[sname]["found_at"])
            rows.append((name, sname, g, lo, hi))
            n9 = q09(d[sname]["found_at"])
            table.append(f"| | | {LABEL[sname]} | {d[sname]['success_rate']:.2f} | {n9:.0f} | {g:.2f} [{lo:.2f}, {hi:.2f}] |")
    (RES / "e26_stage2_table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    if not rows:
        return
    show = ["de_zelinka", "sus", "portfolio_no_sus", "portfolio", "portfolio_cvs"]
    names = [n for n in LEVEL_AB if any(r[0] == n for r in rows)]
    fig, ax = plt.subplots(figsize=(1.5 * len(names) + 3, 4.2))
    w = 0.8 / len(show)
    x = np.arange(len(names))
    for i, sname in enumerate(show):
        vals = [next((r for r in rows if r[0] == n and r[1] == sname), None) for n in names]
        g = np.array([v[2] if v else np.nan for v in vals])
        lo = np.array([v[3] if v else np.nan for v in vals])
        hi = np.array([v[4] if v else np.nan for v in vals])
        xx = x + (i - (len(show) - 1) / 2) * w
        ax.bar(xx, g, width=w * 0.9, color=COLOR[sname], label=LABEL[sname])
        ax.errorbar(xx, g, yerr=[np.clip(g - lo, 0, None), np.clip(hi - g, 0, None)], fmt="none", ecolor=INK2, lw=0.8, capsize=2)
    ax.axhline(1.0, color=INK, lw=1, ls="--")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([TITLE.get(n, n) for n in names], fontsize=8)
    ax.set_ylabel("G_0.9 = N_0.9(перебор) / N_0.9(метод)")
    ax.set_title("Коэффициент выигрыша на редких целях уровней A/B (95 % бутстрап-интервалы)")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, FIG / "fig22_gain_levelAB")


def _sweep(prefix, m, ps):
    out = {}
    for p in ps:
        name = f"{prefix}_se3_m{m}_p{p:g}"
        d = load(name)
        if d:
            out[p] = d
    return out


def slope_ci(series, n_boot=2000, seed=1):
    """series: список (p, found_at). Наклон log N_0.9 против log(1/p) и 95 % бутстрап-интервал
    (по запускам внутри каждого p)."""
    rng = np.random.default_rng(seed)
    ps = np.array([p for p, _ in series])
    arrs = [np.array([np.inf if f is None else f for f in fa], float) for _, fa in series]
    def fit(vals):
        ok = np.isfinite(vals)
        if ok.sum() < 2:
            return np.nan
        return np.polyfit(np.log(1 / ps[ok]), np.log(vals[ok]), 1)[0]
    point = fit(np.array([np.quantile(a, 0.9) for a in arrs]))
    boots = [fit(np.array([np.quantile(rng.choice(a, len(a)), 0.9) for a in arrs])) for _ in range(n_boot)]
    boots = np.array([b for b in boots if np.isfinite(b)])
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan))
    return float(point), float(lo), float(hi)


def fig_scaling_p():
    ps = [1e-2, 1e-3, 1e-4, 1e-5]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    fits = {}
    for ax, prefix, title, show in ((axes[0], "edge", "D2: «краевое состояние» (градиентный сигнал)",
                                     ["uniform", "sus_tesc", "sus_rarity", "sus", "portfolio_no_sus", "portfolio"]),
                                    (axes[1], "emb", "D1: «ворота» (сигнала нет)", ["uniform", "sus", "portfolio_no_sus", "portfolio"])):
        data = _sweep(prefix, 3, ps)
        pp = np.logspace(-5.3, -1.7, 50)
        ax.plot(pp, np.log(10) / pp, color=INK, lw=1, ls="--", label="ln 10 / p (перебор, теория)")
        for sname in show:
            xs, ys, series = [], [], []
            for p, d in sorted(data.items()):
                if sname in d:
                    v = q09(d[sname]["found_at"])
                    series.append((p, d[sname]["found_at"]))
                    if np.isfinite(v):
                        xs.append(p); ys.append(v)
            lab = LABEL[sname]
            if len(series) >= 3:
                s_fit, lo, hi = slope_ci(series)
                fits[(prefix, sname)] = {"s": s_fit, "ci95": [lo, hi], "n_points": len(series)}
                lab = f"{lab}: s = {s_fit:.2f} [{lo:.2f}, {hi:.2f}]"
            if len(xs) >= 1:
                ax.plot(xs, ys, "o-", color=COLOR[sname], label=lab, lw=2.2 if sname == "portfolio" else 1.4)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("доля бассейна цели p"); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("N_0.9 (интегрирований)")
    fig.suptitle("Стоимость обнаружения против меры бассейна (n = 6, R = 100); s — показатель в N_0.9 ∝ p^(−s)", fontsize=10.5)
    fig.tight_layout()
    save(fig, FIG / "fig23_scaling_p")
    (RES / "e26_scaling_fits.json").write_text(json.dumps({f"{a}:{b}": v for (a, b), v in fits.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: round(v["s"], 2) for k, v in fits.items()}


def fig_scaling_n():
    fig, axes = plt.subplots(1, 1, figsize=(6.2, 4.0), squeeze=False)
    axes = axes[0]
    for ax, prefix, ms, show, title in ((axes[0], "edge", [1, 3, 5, 7, 9, 11], ["uniform", "sus", "portfolio_no_sus", "portfolio"], "D2"),):
        for sname in show:
            xs, ys = [], []
            for m in ms:
                d = load(f"{prefix}_se3_m{m}_p0.0001")
                if d and sname in d:
                    v = q09(d[sname]["found_at"])
                    if np.isfinite(v):
                        xs.append(m + 3); ys.append(v)
            if xs:
                ax.plot(xs, ys, "o-", color=COLOR[sname], label=LABEL[sname], lw=2.2 if sname == "portfolio" else 1.4)
        ax.axhline(np.log(10) / 1e-4, color=INK, lw=1, ls="--", label="ln 10 / p")
        ax.set_yscale("log"); ax.set_xlabel("размерность n"); ax.set_title(f"{title}, p = 10⁻⁴", fontsize=10)
    axes[0].set_ylabel("N_0.9"); axes[0].legend(fontsize=7)
    fig.suptitle("Зависимость стоимости от размерности фазового пространства (D2)", fontsize=11)
    fig.tight_layout()
    save(fig, FIG / "fig24_scaling_n")


def fig_ablation():
    variants = ["sus", "portfolio_no_ab", "portfolio_no_sus", "portfolio_eps0", "portfolio", "portfolio_cvs"]
    names = LEVEL_AB + ["edge_se3_m3_p0.001", "edge_se3_m3_p0.0001", "emb_se3_m3_p0.001"]
    M = np.full((len(names), len(variants)), np.nan)
    for i, name in enumerate(names):
        d = load(name)
        if not d or "uniform" not in d:
            continue
        nu = q09(d["uniform"]["found_at"])
        for j, v in enumerate(variants):
            if v in d:
                nv = q09(d[v]["found_at"])
                M[i, j] = nu / nv if np.isfinite(nv) and nv > 0 else 0.0
    keep = ~np.all(np.isnan(M), axis=1)
    if not keep.any():
        return
    M = M[keep]; names = [n for n, k in zip(names, keep) if k]
    fig, ax = plt.subplots(figsize=(1.3 * len(variants) + 3, 0.45 * len(names) + 1.8))
    L = np.log10(np.clip(M, 0.05, 1000))
    im = ax.imshow(L, cmap="RdBu", vmin=-1.3, vmax=1.3, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(L[i, j]) > 0.85 else INK)
    ax.set_xticks(range(len(variants))); ax.set_xticklabels([LABEL[v] for v in variants], rotation=25, ha="right", fontsize=8)
    pretty = {"edge_se3_m3_p0.001": "D2, p = 10⁻³", "edge_se3_m3_p0.0001": "D2, p = 10⁻⁴", "emb_se3_m3_p0.001": "D1, p = 10⁻³"}
    ax.set_yticks(range(len(names))); ax.set_yticklabels([TITLE.get(n, pretty.get(n, n)) for n in names], fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("log10 G_0.9")
    ax.set_title("Абляция: вклад компонентов портфеля (G_0.9 относительно перебора)")
    save(fig, FIG / "fig25_ablation")


def fig_signal():
    """Многомасштабная диагностика канала B (e22): медианное расстояние нецелевых точек верхнего
    множества координаты расщепления до цели против доли q, и наблюдаемый выигрыш расщепления."""
    systems = [("chua_II", "Чуа II"), ("le1", "LE1"), ("se11", "SE11"), ("edge_se3_m3_p0.001", "D2, p = 10⁻³"),
               ("huynh2019", "Хуинь"), ("jafari2026_lc", "цикл Jafari"), ("emb_se3_m3_p0.001", "D1, p = 10⁻³")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1.4, 1]})
    gains = []
    for i, (name, lab) in enumerate(systems):
        p = RES / f"e22_signal_{name}.json"
        d = load(name)
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        cur = [c for c in rec.get("level_curve", []) if c["median_dist_rel"] is not None]
        if not cur:
            continue
        g = None
        if d and "uniform" in d and "sus" in d:
            nu, ns = q09(d["uniform"]["found_at"]), q09(d["sus"]["found_at"])
            g = nu / ns if np.isfinite(ns) and ns > 0 else None
        gains.append((lab, g))
        axes[0].plot([c["q"] for c in cur], [c["median_dist_rel"] for c in cur], "o-", color=SERIES[i % 8],
                     label=f"{lab} (G = {g:.1f})" if g else lab)
    axes[0].set_xscale("log"); axes[0].invert_xaxis(); axes[0].axhline(1.0, color=MUTED, lw=0.8, ls=":")
    axes[0].set_xlabel("доля q верхнего множества координаты"); axes[0].set_ylabel("медиана расстояния до цели (отн.)")
    axes[0].set_title("Сходятся ли «долгие» переходные процессы к цели?", fontsize=10)
    axes[0].legend(fontsize=7)
    if gains:
        labs = [g[0] for g in gains]; vals = [g[1] if g[1] else np.nan for g in gains]
        axes[1].barh(range(len(labs)), vals, color=[SERIES[i % 8] for i in range(len(labs))])
        axes[1].axvline(1.0, color=INK, lw=1, ls="--")
        axes[1].set_yticks(range(len(labs))); axes[1].set_yticklabels(labs, fontsize=8)
        axes[1].set_xscale("log"); axes[1].set_xlabel("G_0.9 расщепления"); axes[1].grid(axis="y", visible=False)
        axes[1].set_title("Наблюдаемый выигрыш", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "fig26_signal_curves")


def main():
    viz.setup()
    for f in (fig_pn, fig_gain_and_table, fig_scaling_p, fig_scaling_n, fig_ablation, fig_signal):
        try:
            r = f()
            print("ok:", f.__name__, "" if r is None else r)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("skip:", f.__name__, repr(e))


if __name__ == "__main__":
    main()
