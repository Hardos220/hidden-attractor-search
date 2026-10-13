"""Таблицы и новые рисунки статьи по расчёту e50 (Р1–Р6) — для русской и английской версий.

    python experiments/e53_paper_results.py            # таблицы results/e53_tables.{json,md} и рисунки ru + en

Рисунки (ru — results/figures, en — paper/chaos/figures):
  fig27_enlarged_regions — доля бассейна в k·B и выигрыш на реальных системах (Р1);
  fig28_coupled_copies   — стоимость обнаружения на k копиях SE11 против размерности (Р4);
  fig29_d2_small_p       — выигрыш на D2 до p = 1e-7 и диагностика предела разрешения расщепления (Р3).
Рисунки 21–25 строит experiments/e26_stage2_figures.py (по тем же данным e50).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import e26_stage2_figures as e26  # noqa: E402
from hsearch import viz  # noqa: E402
from hsearch.viz import INK, INK2, MUTED, SERIES, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

RES = ROOT / "results"
FIG_RU = RES / "figures"
FIG_EN = ROOT / "paper" / "chaos" / "figures"
N_BOOT = 2000
SYS5 = ["se11", "le1", "chua_I", "chua_II", "jafari2026_lc"]
KS = {"se11": [1, 2, 4, 8, 16], "le1": [1, 2, 4, 8], "chua_I": [1, 2, 4, 8], "chua_II": [1, 2, 4], "jafari2026_lc": [1, 2, 4]}
T = {
    "ru": {"se11": "SE11", "le1": "LE1", "chua_I": "Чуа I", "chua_II": "Чуа II", "jafari2026_lc": "цикл Jafari",
           "uniform": "равномерный перебор", "sus": "расщепление", "portfolio": "портфель", "edge_bisect": "бисекция к границам",
           "sus_smin": "расщепление, порог шага 5·10⁻⁴ или без него", "sus_nostall": "расщепление без порога по шагу",
           "sus_coord": "расщепление + перевыборка координат", "portfolio_coord": "портфель + перевыборка координат",
           "sus_dt": "расщепление, меньший шаг Δt", "portfolio_dt": "портфель, меньший шаг Δt",
           "portfolio_smin": "портфель, порог шага 5·10⁻⁴",
           "k": "увеличение области k", "p": "доля бассейна цели p", "G": "G₀.₉ = N₀.₉(перебор) / N₀.₉(метод)",
           "k3": "p₀ / k³ (бассейн внутри B)", "guar": "гарантия ε = 0.2", "even": "G = 1",
           "t27a": "(a) Доля бассейна в k·B", "t27b": "(b) Выигрыш портфеля (сплошная) и расщепления (пунктир)",
           "s27": "Реальные системы в увеличенных областях k·B (R = 50; k = 1 — R = 100)",
           "n": "размерность n", "N": "N₀.₉, интегрирований", "coupled": "связь ε = 0.01", "uncoupled": "без связи",
           "theory": "перебор: ln 10 / p", "s28": "k копий SE11: стоимость обнаружения скрытого режима «все копии на HA»",
           "eqG": "G₀.₉ (p ≤ 10⁻⁶: по доле успехов в бюджете)", "s29": "Семейство D2 (n = 6): выигрыш до p = 10⁻⁷ и диагностика обрыва",
           "budget": "в пределах бюджета 2·10⁵"},
    "en": {"se11": "SE11", "le1": "LE1", "chua_I": "Chua I", "chua_II": "Chua II", "jafari2026_lc": "Jafari cycle",
           "uniform": "uniform sampling", "sus": "splitting", "portfolio": "portfolio", "edge_bisect": "boundary bisection",
           "sus_smin": "splitting, step floor 5×10⁻⁴ or none", "sus_nostall": "splitting, no step floor",
           "sus_coord": "splitting + coordinate redraws", "portfolio_coord": "portfolio + coordinate redraws",
           "sus_dt": "splitting, smaller step Δt", "portfolio_dt": "portfolio, smaller step Δt",
           "portfolio_smin": "portfolio, step floor 5×10⁻⁴",
           "k": "region scale factor k", "p": "target basin measure p", "G": "G₀.₉ = N₀.₉(uniform) / N₀.₉(method)",
           "k3": "p₀ / k³ (basin inside B)", "guar": "guarantee ε = 0.2", "even": "G = 1",
           "t27a": "(a) Basin measure in k·B", "t27b": "(b) Gain of the portfolio (solid) and splitting (dashed)",
           "s27": "Real systems in enlarged regions k·B (R = 50; k = 1: R = 100)",
           "n": "dimension n", "N": "N₀.₉, integrations", "coupled": "coupling ε = 0.01", "uncoupled": "uncoupled",
           "theory": "uniform: ln 10 / p", "s28": "k copies of SE11: cost of detecting the hidden regime “all copies on HA”",
           "eqG": "G₀.₉ (p ≤ 10⁻⁶: from the success fraction)",
           "s29": "Family D2 (n = 6): gain down to p = 10⁻⁷ and the step floor of splitting", "budget": "within the budget 2×10⁵"},
}
COL = {"se11": SERIES[0], "le1": SERIES[1], "chua_I": SERIES[2], "chua_II": SERIES[3], "jafari2026_lc": SERIES[4],
       "uniform": MUTED, "sus": SERIES[1], "portfolio": SERIES[0], "edge_bisect": SERIES[3],
       "sus_smin": SERIES[6], "sus_nostall": SERIES[5], "portfolio_smin": SERIES[7], "sus_coord": SERIES[2], "portfolio_coord": SERIES[5]}


# ============================================================================ данные
def var_name(system, k):
    return system if k == 1 else f"{system}_box{k}"


def fa(d, s):
    return d[s]["found_at"] if d and s in d else None


def gain(fu, fs):
    if fu is None or fs is None:
        return None
    g, lo, hi = e26.gain_ci(fu, fs, n_boot=N_BOOT)
    return {"G": g, "lo": lo, "hi": hi, "N09": e26.q09(fs), "success": float(np.mean([x is not None for x in fs])), "R": len(fs)}


def p_hat(fu):
    k = sum(x is not None for x in fu)
    ev = sum(x for x in fu if x is not None)
    return k / ev if ev else float("nan")


def success_gain(fs, p, budget):
    """Эквивалентный выигрыш по доле успехов в бюджете: -ln(1 - f) / (p B) (перебор: f = 1 - exp(-p B))."""
    f = float(np.mean([x is not None for x in fs]))
    return (-math.log(1 - f) / (p * budget) if f < 1 else float("inf")), f


def p1_reference():
    """Доля бассейна HA у SE11 в исходной области — по выборке e02 (больше точек, чем у запусков перебора)."""
    e02 = json.loads((RES / "e02_uniform.json").read_text(encoding="utf-8"))
    return float(e02["se11"]["hidden"]["HA"]["p"])


def tables():
    out = {}
    # Т1: уровни A/B и Цзян, R = 100
    t1 = {}
    for v in e26.LEVEL_AB + ["jiang_nfe", "jiang_nff", "jiang_nfg", "jiang_sfa", "jiang_sfb"]:
        d = e26.load(v)
        if not d:
            continue
        fu = fa(d, "uniform")
        t1[v] = {"p": p_hat(fu), "N09u": e26.q09(fu), "R": len(fu),
                 **{s: gain(fu, fa(d, s)) for s in ("de_zelinka", "sus", "portfolio_no_sus", "portfolio", "portfolio_cvs") if s in d}}
    out["levelAB"] = t1
    # Т2: D2 (n = 6) по p
    t2 = {}
    for p in ("0.01", "0.001", "0.0001", "1e-05", "1e-06", "1e-07"):
        v = f"edge_se3_m3_p{p}"
        d = e26.load(v)
        if not d:
            continue
        pf = float(p)
        fu = fa(d, "uniform")
        row = {"N09u": e26.q09(fu) if fu else None, "theory": math.log(10) / pf, "R_u": len(fu) if fu else 0}
        for s in ("sus", "sus_tesc", "sus_rarity", "portfolio_no_sus", "portfolio_no_ab", "portfolio", "sus_smin", "sus_nostall",
                  "portfolio_smin", "sus_coord", "portfolio_coord"):
            fs = fa(d, s)
            if fs is None:
                continue
            bud = d[s]["budget"]
            rec = gain(fu, fs) if fu else None
            geq, f = success_gain(fs, pf, bud)
            row[s] = {**(rec or {}), "G_eq": geq, "success": f, "budget": bud, "R": len(fs)}
        t2[p] = row
    out["d2_p"] = t2
    # Т2б: D2 с меньшим шагом интегрирования (B10): p не зависит от шага, эталон перебора — ln 10 / p
    t2b = {}
    for p, f in (("1e-06", 0.25), ("1e-07", 0.25), ("1e-07", 0.0625)):
        d = e26.load(f"edge_se3_m3_p{p}_dt{f:g}")
        if not d:
            continue
        pf = float(p)
        row = {"p": pf, "dt_scale": f, "theory": math.log(10) / pf}
        for s in ("sus", "portfolio"):
            fs = fa(d, s)
            if fs is None:
                continue
            n9 = e26.q09(fs)
            geq, fr = success_gain(fs, pf, d[s]["budget"])
            row[s] = {"N09": n9, "G": row["theory"] / n9 if math.isfinite(n9) else 0.0, "G_eq": geq, "success": fr, "R": len(fs),
                      "median": float(np.median([x for x in fs if x is not None])) if any(x is not None for x in fs) else None}
        t2b[f"{p}:dt{f:g}"] = row
    out["d2_dt"] = t2b
    # Т3: реальные системы в k·B (Р1)
    t3 = {}
    for sname in SYS5:
        for k in KS[sname]:
            d = e26.load(var_name(sname, k))
            if not d:
                continue
            fu = fa(d, "uniform")
            t3[f"{sname}:k{k}"] = {"system": sname, "k": k, "p": p_hat(fu), "N09u": e26.q09(fu), "R": len(fu),
                                   "sus": gain(fu, fa(d, "sus")), "portfolio": gain(fu, fa(d, "portfolio"))}
    out["enlarged"] = t3
    # Т4: связанные копии SE11 (Р4)
    p1 = p1_reference()
    t4 = {}
    for n, v, eps in ((3, "se11", None), (6, "coupled_se11_k2_eps0.01_c1", 0.01), (9, "coupled_se11_k3_eps0.01_c1", 0.01),
                      (9, "coupled_se11_k3_eps0_c1", 0.0), (12, "coupled_se11_k4_eps0_c1", 0.0)):
        d = e26.load(v)
        if not d:
            continue
        fu = fa(d, "uniform")
        k = n // 3
        row = {"n": n, "eps": eps, "p_product": p1 ** k}
        if fu:
            row.update(p=p_hat(fu), N09u=e26.q09(fu), R_u=len(fu))
            ref = fu
        else:
            # перебор недостижим: эталон — геометрическое распределение с p = p1^k (точно при eps = 0)
            row.update(p=p1 ** k, N09u=math.log(10) / p1 ** k, R_u=0, N09u_theory=True)
            ref = None
        for s in ("sus", "portfolio", "edge_bisect"):
            fs = fa(d, s)
            if fs is None:
                continue
            if ref is not None:
                row[s] = gain(ref, fs)
            else:
                n9 = e26.q09(fs)
                row[s] = {"N09": n9, "G": row["N09u"] / n9 if math.isfinite(n9) else 0.0, "R": len(fs),
                          "success": float(np.mean([x is not None for x in fs]))}
        t4[v] = row
    out["coupled"] = t4
    # Т5: бисекция к границам (Р6), Т6: горизонт (Р5), Т7: робастность и абляция
    t5 = {}
    for v in ("chua_I", "chua_II", "se11", "le1", "jafari2026_lc", "huynh2019", "jafari2026_chaos",
              "edge_se3_m3_p0.001", "edge_se3_m3_p0.0001", "emb_se3_m3_p0.001"):
        d = e26.load(v)
        fu = fa(d, "uniform")
        t5[v] = {s: gain(fu, fa(d, s)) for s in ("portfolio", "edge_bisect", "edge_bisect_u20") if d and s in d}
    out["bisection"] = t5
    t6 = {}
    for v in ("chua_II", "se11", "le1", "edge_se3_m3_p0.0001"):
        for h in (None, 0.5, 2):
            d = e26.load(v if h is None else f"{v}_h{h:g}")
            fu = fa(d, "uniform")
            t6[f"{v}:h{1 if h is None else h:g}"] = {"N09u": e26.q09(fu), "R": len(fu),
                                                    **{s: gain(fu, fa(d, s)) for s in ("sus", "portfolio") if s in d}}
    out["horizon"] = t6
    t7 = {}
    for v in ("chua_II", "le1", "edge_se3_m3_p0.0001", "chua_I", "se11", "jafari2026_lc", "huynh2019"):
        d = e26.load(v)
        fu = fa(d, "uniform")
        t7[v] = {s: gain(fu, fa(d, s)) for s in ("portfolio", "portfolio_eps01", "portfolio_eps03", "portfolio_p01",
                                                   "portfolio_p03", "portfolio_sig03", "portfolio_n100", "sus_tesc",
                                                   "sus_rarity", "portfolio_no_ab", "portfolio_eps0", "sus") if d and s in d}
    out["robust_ablation"] = t7
    t8 = {}
    for v in ("edge_se3_m1_p0.0001", "chua_I", "chua_II", "se11", "le1", "coupled_se11_k2_eps0.01_c1"):
        d = e26.load(v)
        fu = fa(d, "uniform")
        t8[v] = {s: gain(fu, fa(d, s)) for s in ("sus", "portfolio", "sus_coord", "portfolio_coord") if d and s in d}
    out["coord_control"] = t8
    fits = RES / "e26_scaling_fits.json"
    out["slopes"] = json.loads(fits.read_text(encoding="utf-8")) if fits.exists() else {}
    out["p1_se11"] = p1
    return out


def md(out):
    f = lambda x: "—" if x is None else ("∞" if not math.isfinite(x) else (f"{x:.2f}" if abs(x) < 100 else f"{x:.3g}"))  # noqa: E731
    g = lambda r: "—" if not r else f"{f(r['G'])} [{f(r.get('lo'))}–{f(r.get('hi'))}]"  # noqa: E731
    L = ["# Итоги e50 для статьи (e53)", "", "## Уровни A/B, R = 100", "", "| система | p | N0.9 перебора | DE | расщепление | AL+20% | портфель | +нульклины |", "|---|---|---|---|---|---|---|---|"]
    for v, r in out["levelAB"].items():
        L.append(f"| {v} | {r['p']:.2g} | {f(r['N09u'])} | " + " | ".join(g(r.get(s)) for s in ("de_zelinka", "sus", "portfolio_no_sus", "portfolio", "portfolio_cvs")) + " |")
    L += ["", "## D2 (n = 6)", "", "| p | N0.9 перебора (теория) | стратегия: N0.9 (G) или доля успехов (экв. G) |", "|---|---|---|"]
    for p, r in out["d2_p"].items():
        cells = []
        for s in ("sus", "portfolio", "sus_smin", "sus_nostall", "portfolio_smin"):
            if s in r:
                x = r[s]
                cells.append(f"{s}: " + (f"{f(x['N09'])} ({f(x['G'])}×)" if x.get("N09") is not None and math.isfinite(x["N09"])
                                         else f"успех {100 * x['success']:.0f}% (экв. G {f(x['G_eq'])})"))
        L.append(f"| {p} | {f(r['N09u'])} ({f(r['theory'])}) | " + "; ".join(cells) + " |")
    L += ["", "## Реальные системы в k·B", "", "| система | k | p | N0.9 перебора | расщепление | портфель |", "|---|---|---|---|---|---|"]
    for r in out["enlarged"].values():
        L.append(f"| {r['system']} | {r['k']} | {r['p']:.2g} | {f(r['N09u'])} | {g(r['sus'])} | {g(r['portfolio'])} |")
    L += ["", "## Связанные копии SE11", "", "| вариант | n | p (p1^k) | N0.9 перебора | расщепление | портфель | бисекция |", "|---|---|---|---|---|---|---|"]
    for v, r in out["coupled"].items():
        L.append(f"| {v} | {r['n']} | {r['p']:.2g} ({r['p_product']:.2g}) | {f(r['N09u'])}{' (теория)' if r.get('N09u_theory') else ''} | "
                 + " | ".join(g(r.get(s)) for s in ("sus", "portfolio", "edge_bisect")) + " |")
    L += ["", "## Бисекция к границам", ""] + [f"- {v}: " + ", ".join(f"{s} {g(x)}" for s, x in r.items()) for v, r in out["bisection"].items()]
    L += ["", "## Горизонт", ""] + [f"- {v}: N0.9u {f(r['N09u'])}, " + ", ".join(f"{s} {g(r.get(s))}" for s in ("sus", "portfolio")) for v, r in out["horizon"].items()]
    L += ["", "## Робастность и абляция", ""] + [f"- {v}: " + ", ".join(f"{s} {g(x)}" for s, x in r.items()) for v, r in out["robust_ablation"].items()]
    L += ["", "## Перевыборка координат: контроль", ""] + [f"- {v}: " + ", ".join(f"{s} {g(x)}" for s, x in r.items()) for v, r in out.get("coord_control", {}).items()]
    L += ["", "## Наклоны s (N0.9 ∝ p^-s, D2/D1 p = 1e-2…1e-5)", ""] + [f"- {k}: {v['s']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]" for k, v in out["slopes"].items()]
    return "\n".join(L) + "\n"


# ============================================================================ рисунки
def _wh(lang, w_en, h_en, scale_ru=1.45):
    return (w_en, h_en) if lang == "en" else (w_en * scale_ru, h_en * scale_ru)


def fig27(out, lang):
    t = T[lang]
    fig, axes = plt.subplots(1, 2, figsize=_wh(lang, 6.69, 2.7))
    ax = axes[0]
    for sname in SYS5:
        rows = sorted((r for r in out["enlarged"].values() if r["system"] == sname), key=lambda r: r["k"])
        if not rows:
            continue
        ax.plot([r["k"] for r in rows], [r["p"] for r in rows], "o-", color=COL[sname], label=t[sname], lw=1.5, ms=4)
        p0 = rows[0]["p"]
        kk = np.array([1, rows[-1]["k"]])
        ax.plot(kk, p0 / kk ** 3, ":", color=COL[sname], lw=0.9)
    ax.plot([], [], ":", color=INK2, label=t["k3"])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlabel(t["k"]); ax.set_ylabel(t["p"]); ax.set_title(t["t27a"], fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    ax = axes[1]
    for sname in SYS5:
        rows = sorted((r for r in out["enlarged"].values() if r["system"] == sname), key=lambda r: r["k"])
        for s, ls in (("portfolio", "-"), ("sus", "--")):
            pts = [(r["p"], r[s]["G"], r[s]["lo"], r[s]["hi"]) for r in rows if r.get(s) and r[s]["G"] > 0]
            if not pts:
                continue
            x, y, lo, hi = map(np.array, zip(*pts))
            ax.plot(x, y, ls, marker="o" if s == "portfolio" else None, color=COL[sname], lw=1.6 if s == "portfolio" else 1.0, ms=4,
                    label=t[sname] if s == "portfolio" else None)
            if s == "portfolio":
                ax.fill_between(x, lo, hi, color=COL[sname], alpha=0.12, lw=0)
    ax.axhline(1.0, color=INK, lw=0.9, ls="--"); ax.axhline(0.2, color=MUTED, lw=0.9, ls=":")
    ax.text(ax.get_xlim()[0] if False else 3e-2, 1.08, t["even"], fontsize=6.5, color=INK2)
    ax.text(3e-2, 0.215, t["guar"], fontsize=6.5, color=INK2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(t["p"]); ax.set_ylabel(t["G"]); ax.set_title(t["t27b"], fontsize=9)
    if lang == "ru":
        fig.suptitle(t["s27"], fontsize=10.5)
    fig.tight_layout()
    return fig


def fig28(out, lang):
    t = T[lang]
    fig, ax = plt.subplots(figsize=_wh(lang, 3.37, 2.9))
    c = out["coupled"]
    rows_c = [c[v] for v in ("se11", "coupled_se11_k2_eps0.01_c1", "coupled_se11_k3_eps0.01_c1") if v in c]
    rows_u = [c[v] for v in ("coupled_se11_k3_eps0_c1", "coupled_se11_k4_eps0_c1") if v in c]
    nn = np.array([3, 6, 9, 12])
    p1 = out["p1_se11"]
    ax.plot(nn, math.log(10) / p1 ** (nn / 3), "--", color=INK, lw=0.9, label=t["theory"] + " (p = p₁ⁿᐟ³)")
    for s in ("uniform", "sus", "portfolio", "edge_bisect"):
        for rows, filled in ((rows_c, True), (rows_u, False)):
            pts = []
            for r in rows:
                if s == "uniform":
                    if not r.get("N09u_theory"):
                        pts.append((r["n"], r["N09u"]))
                elif r.get(s) and math.isfinite(r[s]["N09"]):
                    pts.append((r["n"], r[s]["N09"]))
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, "o-" if filled else "s", color=COL[s], mfc=COL[s] if filled else "none", lw=1.6 if filled else 0,
                    ms=5, label=(t[s] if filled else None))
    ax.plot([], [], "o", color=INK2, label=t["coupled"]); ax.plot([], [], "s", color=INK2, mfc="none", label=t["uncoupled"])
    ax.set_yscale("log"); ax.set_xticks([3, 6, 9, 12])
    ax.set_xlabel(t["n"]); ax.set_ylabel(t["N"])
    ax.legend(fontsize=6.3, loc="upper left")
    if lang == "ru":
        ax.set_title(t["s28"], fontsize=9)
    fig.tight_layout()
    return fig


def fig29(out, lang):
    t = T[lang]
    fig, ax = plt.subplots(figsize=_wh(lang, 3.37, 2.8))
    d2 = out["d2_p"]
    for s in ("sus", "portfolio", "sus_smin", "portfolio_smin", "sus_coord", "portfolio_coord"):
        pts = []
        for p, r in d2.items():
            if s not in r:
                continue
            x = r[s]
            n9 = x.get("N09")
            if n9 is not None and math.isfinite(n9) and x.get("G"):
                pts.append((float(p), x["G"], True))
            else:
                pts.append((float(p), x["G_eq"], False))
        if not pts:
            continue
        pts.sort()
        xs, ys, solid = zip(*pts)
        ls = "-" if s in ("sus", "portfolio") else (":" if s.endswith("smin") else "--")
        ax.plot(xs, ys, ls, color=COL[s], lw=1.6 if s == "portfolio" else 1.1, label=t[s])
        ax.scatter([a for a, b, c in pts if c], [b for a, b, c in pts if c], color=COL[s], s=14, zorder=3)
        ax.scatter([a for a, b, c in pts if not c], [b for a, b, c in pts if not c], facecolor="none", edgecolor=COL[s], s=18, zorder=3)
    dd = out.get("d2_dt", {})
    for s, mk in (("sus", "*"), ("portfolio", "P")):
        pts = [(r["p"], r[s]["G"], r["dt_scale"]) for r in dd.values() if s in r and r[s]["G"] > 0]
        if pts:
            ax.scatter([a for a, _, _ in pts], [b for _, b, _ in pts], marker=mk, s=70, color=COL[s], zorder=4, label=t[s + "_dt"])
            if s == "sus":
                for a, b, c in pts:
                    ax.annotate(f"Δt/{1 / c:g}", (a, b), textcoords="offset points", xytext=(6, 2), fontsize=6.5, color=INK2)
    ax.axhline(1.0, color=INK, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_xlabel(t["p"]); ax.set_ylabel(t["eqG"], fontsize=7)
    ax.legend(fontsize=6, loc="upper left")
    if lang == "ru":
        ax.set_title(t["s29"], fontsize=9)
    fig.tight_layout()
    return fig


# ============================================================================ фрагменты таблиц для статьи
NAMES = {"en": {"chua_I": "Chua I", "chua_II": "Chua II", "se11": "SE11", "le1": "LE1", "jafari2026_lc": "Jafari 2026, cycle",
                "huynh2019": "Huynh 2019", "jafari2026_chaos": "Jafari 2026, chaos"},
         "ru": {"chua_I": "Чуа I", "chua_II": "Чуа II", "se11": "SE11", "le1": "LE1", "jafari2026_lc": "Jafari 2026, цикл",
                "huynh2019": "Хуинь 2019", "jafari2026_chaos": "Jafari 2026, хаос"}}
P_TAB = {"chua_I": "0.035", "chua_II": "0.0027", "se11": "0.0079", "le1": "0.0016", "jafari2026_lc": "0.009",
         "huynh2019": "0.0091", "jafari2026_chaos": ("$(2$--$4)\\times10^{-4}$", "(2–4)·10⁻⁴")}


def num(x):
    if x is None:
        return "---"
    if not math.isfinite(x):
        return "$\\infty$"
    if x >= 100:
        return f"{x:.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def cell(r, lang, bold=True):
    """G с интервалом: LaTeX — \\ci{lo}{hi}, HTML — [lo, hi]; жирным, если нижняя граница выше 1.
    Если цель нашли меньше 90 % запусков (N_0.9 = ∞) — «∞ (доля успехов)»."""
    if not r:
        return "---" if lang == "en" else "—"
    if r.get("success") is not None and r["success"] < 0.9:
        pc = f"{100 * r['success']:.0f}"
        return f"$\\infty$ ({pc}\\%)" if lang == "en" else f"∞ ({pc} %)"
    g = num(r["G"])
    if r.get("lo") is None:
        return g if lang == "en" else g.replace("$\\infty$", "∞")
    b = bold and r["lo"] > 1
    if lang == "en":
        return (f"\\textbf{{{g}}}" if b else g) + f" \\ci{{{num(r['lo'])}}}{{{num(r['hi'])}}}"
    s = (f"<b>{g}</b>" if b else g) + f" [{num(r['lo'])}, {num(r['hi'])}]"
    return s.replace("$\\infty$", "∞")


def row(cells, lang, cls=""):
    if lang == "en":
        return " & ".join(cells) + " \\\\"
    tds = "".join(f'<td class="num">{c}</td>' if i else f"<td>{c}</td>" for i, c in enumerate(cells))
    return f"<tr{f' class={chr(34)}{cls}{chr(34)}' if cls else ''}>{tds}</tr>"


def snippets(out):
    d = RES / "e53_snippets"
    d.mkdir(exist_ok=True)
    for lang, ext in (("en", "tex"), ("ru", "html")):
        nm = NAMES[lang]
        # Т1: уровни A/B
        L = []
        for v in e26.LEVEL_AB:
            r = out["levelAB"].get(v)
            if not r:
                continue
            p = P_TAB[v] if isinstance(P_TAB[v], str) else P_TAB[v][0 if lang == "en" else 1]
            L.append(row([nm[v], p, f"{r['N09u']:.0f}"] + [cell(r.get(s), lang) for s in
                                                          ("de_zelinka", "sus", "portfolio_no_sus", "portfolio", "portfolio_cvs")], lang))
        jn = [out["levelAB"].get(v) for v in ("jiang_nfe", "jiang_nff", "jiang_nfg")]
        js = [out["levelAB"].get(v) for v in ("jiang_sfa", "jiang_sfb")]
        sep = "/" if lang == "en" else " / "
        def grp(rows, s):
            return sep.join(num(r[s]["G"]) if r and r.get(s) else "—" for r in rows)
        if all(jn):
            L.append(row(["Jiang NF I e/f/g" if lang == "en" else "Цзян NF I e / f / g", "0.028--0.039" if lang == "en" else "0.028–0.039",
                          sep.join(f"{r['N09u']:.0f}" for r in jn)] + [grp(jn, s) for s in ("de_zelinka", "sus", "portfolio_no_sus", "portfolio")]
                         + ["---" if lang == "en" else "—"], lang, "grp"))
        if all(js):
            L.append(row(["Jiang SF I a/b" if lang == "en" else "Цзян SF I a / b", "0.065", sep.join(f"{r['N09u']:.0f}" for r in js)]
                         + [grp(js, s) for s in ("de_zelinka", "sus", "portfolio_no_sus", "portfolio")] + ["---" if lang == "en" else "—"], lang))
        (d / f"tab_gains.{ext}").write_text("\n".join(L) + "\n", encoding="utf-8")
        # Т: увеличенные области
        L = []
        for r in out["enlarged"].values():
            L.append(row([nm[r["system"]], str(r["k"]), f"{r['p']:.1e}".replace("e-0", "e-") if lang == "en" else f"{r['p']:.1e}",
                          f"{r['N09u']:.0f}", cell(r["sus"], lang), cell(r["portfolio"], lang)], lang))
        (d / f"tab_enlarged.{ext}").write_text("\n".join(L) + "\n", encoding="utf-8")
        # Т: связанные копии
        L = []
        for v, r in out["coupled"].items():
            eps = "---" if r["eps"] is None else (f"{r['eps']:g}")
            Nu = f"{r['N09u']:.3g}" + (" (theory)" if lang == "en" and r.get("N09u_theory") else " (теория)" if r.get("N09u_theory") else "")
            L.append(row([str(r["n"]), eps, f"{r['p']:.2g}", Nu] + [cell(r.get(s), lang) for s in ("sus", "portfolio", "edge_bisect")], lang))
        (d / f"tab_coupled.{ext}").write_text("\n".join(L) + "\n", encoding="utf-8")
        # Т: робастность (R = 50; база — R = 100)
        L = []
        for v, title in (("chua_II", nm["chua_II"]), ("le1", nm["le1"]), ("edge_se3_m3_p0.0001", "D2, $n=6$, $p=10^{-4}$" if lang == "en" else "D2, \\(n=6\\), \\(p=10^{-4}\\)")):
            r = out["robust_ablation"][v]
            L.append(row([title] + [cell(r.get(s), lang, bold=False) for s in ("portfolio", "portfolio_eps01", "portfolio_eps03", "portfolio_p01",
                                                                              "portfolio_p03", "portfolio_sig03", "portfolio_n100")], lang))
        (d / f"tab_robust.{ext}").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("фрагменты таблиц: results/e53_snippets/")


def main():
    viz.setup()
    out = tables()
    (RES / "e53_tables.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    (RES / "e53_tables.md").write_text(md(out), encoding="utf-8")
    print("таблицы: results/e53_tables.json, results/e53_tables.md")
    snippets(out)
    FIG_EN.mkdir(parents=True, exist_ok=True)
    for lang, d in (("ru", FIG_RU), ("en", FIG_EN)):
        for name, fn in (("fig27_enlarged_regions", fig27), ("fig28_coupled_copies", fig28), ("fig29_d2_small_p", fig29)):
            save(fn(out, lang), d / name)
            print("рисунок:", lang, name)


if __name__ == "__main__":
    main()
