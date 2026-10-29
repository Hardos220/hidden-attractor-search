"""Product targets: information bound and ideal splitting against the coupled-copies data (2026-10-29).

Theory (article, Theorem 2.1, Corollary 2.3, Proposition 4.2): for k uncoupled copies of a unit whose target basin
(measure p1) lies in the nonescaping set Q1 (measure q), an algorithm whose only information besides the target
label is the escape step cannot beat uniform sampling by more than ln(1/delta)/((1-delta) q^k); ideal multilevel
splitting along the escape step costs n0 + n0 (1-p0) K_k + ln(1/delta) (q/p1)^k with K_k = ceil(k ln(1/q)/ln(1/p0)).
The script measures q for SE11 from the uniform census (results/e02_uniform.json), evaluates both curves, and compares
them with the measured gains of splitting and of the portfolio on coupled copies (results/e53_tables.json "coupled";
bootstrap intervals for n = 12 from the raw runs in results/e50/B8). It fits the base of the geometric growth over
k = 2..4 and the cost exponent along p = p1^k, and draws the figure of the article.

    python experiments/e68_product_theory.py
    python experiments/e68_product_theory.py --figure    # only the figure, from results/e68_product_theory.json

Writes results/e68_product_theory.{json,md} and paper/siads/figures/fig_product_gain.{pdf,png}.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"
DELTA = 0.1
N0_INIT, P0 = 50, 0.2           # splitting of the article: reference sample 50, selection fraction 0.2
N_BOOT = 4000


def q09(a):
    a = np.asarray(a, float)
    return float(np.quantile(a, 0.9)) if np.mean(np.isfinite(a)) >= 0.9 else math.inf


def boot_gain(found, n0, rng):
    a = np.array([math.inf if x is None else float(x) for x in found], float)
    g = n0 / q09(a)
    gb = [n0 / q09(rng.choice(a, len(a))) for _ in range(N_BOOT)]
    return g, float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))


def main():
    rng = np.random.default_rng(68)
    cen = json.loads((RES / "e02_uniform.json").read_text(encoding="utf-8"))["se11"]
    n = cen["n_sample"]
    fr = cen["fractions"]
    q = 1.0 - fr["escape"]
    q_se = math.sqrt(q * (1 - q) / n)
    p1 = fr["HA"]
    c_bound = math.log(1 / DELTA) / (1 - DELTA)

    tab = json.loads((RES / "e53_tables.json").read_text(encoding="utf-8"))["coupled"]
    rows = []
    keys = [("se11", 1, None), ("coupled_se11_k2_eps0.01_c1", 2, 0.01), ("coupled_se11_k3_eps0.01_c1", 3, 0.01),
            ("coupled_se11_k3_eps0_c1", 3, 0.0), ("coupled_se11_k4_eps0_c1", 4, 0.0)]
    for key, k, eps in keys:
        t = tab[key]
        n0u = t["N09u"]
        rec = {"key": key, "k": k, "n": 3 * k, "eps": eps, "p": t["p"], "N0": n0u}
        for strat in ("sus", "portfolio"):
            s = t[strat]
            if "lo" in s:
                g, lo, hi = s["G"], s["lo"], s["hi"]
            else:
                f = RES / "e50" / "B8" / f"{key}__{strat}.jsonl"
                runs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
                runs = [r for r in runs if "error" not in r]
                g, lo, hi = boot_gain([r.get("found_at") for r in runs], n0u, rng)
            rec[strat] = {"G": g, "lo": lo, "hi": hi, "N09": n0u / g}
        K = math.ceil(k * math.log(1 / q) / math.log(1 / P0))
        n_ideal = N0_INIT + N0_INIT * (1 - P0) * K + math.log(1 / DELTA) * (q / p1) ** k
        rec["K_ideal"] = K
        rec["N_ideal"] = n_ideal
        rec["G_ideal"] = n0u / n_ideal
        rec["G_bound"] = c_bound * q ** (-k)
        rec["eff_portfolio_vs_ideal"] = rec["portfolio"]["G"] / rec["G_ideal"]
        rec["eff_portfolio_vs_bound"] = rec["portfolio"]["G"] / rec["G_bound"]
        rows.append(rec)

    # geometric base over k = 2..4 (k = 3 uncoupled, so that k = 3, 4 share eps = 0)
    def base(strat):
        r2 = next(r for r in rows if r["k"] == 2)
        r4 = next(r for r in rows if r["k"] == 4)
        b = math.sqrt(r4[strat]["G"] / r2[strat]["G"])
        blo = math.sqrt(r4[strat]["lo"] / r2[strat]["hi"])
        bhi = math.sqrt(r4[strat]["hi"] / r2[strat]["lo"])
        return b, blo, bhi

    sel = [r for r in rows if r["key"] != "coupled_se11_k3_eps0.01_c1"]

    def slope(strat):
        x = np.log([1 / r["p"] for r in sel])
        y = np.log([r[strat]["N09"] for r in sel])
        return float(np.polyfit(x, y, 1)[0])

    out = {"q": q, "q_se": q_se, "p1": p1, "n_census": n, "inv_q": 1 / q, "bound_const": c_bound,
           "s_ideal": 1 - math.log(1 / q) / math.log(1 / p1), "rows": rows,
           "base": {"portfolio": base("portfolio"), "sus": base("sus"), "ideal": 1 / q},
           "slope": {"portfolio": slope("portfolio"), "sus": slope("sus")}}
    (RES / "e68_product_theory.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    L = [f"# e68: product targets (SE11 copies)", "",
         f"q = mu(nonescaping set of SE11) = {q:.4f} +- {q_se:.4f} (n = {n}); 1/q = {1/q:.1f}; p1 = {p1}",
         f"s_ideal along p = p1^k: {out['s_ideal']:.3f}; measured: portfolio {out['slope']['portfolio']:.3f}, "
         f"splitting {out['slope']['sus']:.3f}", "",
         "| k | eps | p | N0 | G splitting | G portfolio | ideal splitting N (K) | G ideal | bound 2.56/q^k | portfolio/ideal |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['k']} | {r['eps']} | {r['p']:.3g} | {r['N0']:.3g} | {r['sus']['G']:.3g} [{r['sus']['lo']:.3g}, "
                 f"{r['sus']['hi']:.3g}] | {r['portfolio']['G']:.3g} [{r['portfolio']['lo']:.3g}, {r['portfolio']['hi']:.3g}] "
                 f"| {r['N_ideal']:.0f} ({r['K_ideal']}) | {r['G_ideal']:.3g} | {r['G_bound']:.3g} | {r['eff_portfolio_vs_ideal']:.2f} |")
    for s in ("portfolio", "sus"):
        b = out["base"][s]
        L.append(f"base of geometric growth, {s}: {b[0]:.1f} [{b[1]:.1f}, {b[2]:.1f}] (ideal 1/q = {1/q:.1f})")
    (RES / "e68_product_theory.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    figure(out)


def figure(out):
    """Figure of the article at the SIAM column size (included at 0.62 of the text width, 3.18 in): white
    background, framed legend in the empty upper left, so that the legend markers are not read as data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hsearch import viz
    viz.setup()
    viz.setup_en()
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    rows = out["rows"]
    q, p1 = out["q"], out["p1"]
    width = 3.18
    fig, ax = plt.subplots(figsize=(width, 2.6))
    kk = np.linspace(1, 4, 50)
    ax.semilogy(kk, out["bound_const"] * q ** (-kk), color="0.2", ls="-", lw=1.2,
                label=r"bound $2.56\,q^{-k}$")
    ki = np.arange(1, 5)
    gi = []
    for k in ki:
        K = math.ceil(k * math.log(1 / q) / math.log(1 / P0))
        n_id = N0_INIT + N0_INIT * (1 - P0) * K + math.log(1 / DELTA) * (q / p1) ** k
        gi.append((math.log(1 / DELTA) / p1 ** k) / n_id)
    ax.semilogy(ki, gi, color="0.45", ls="--", lw=1.2, marker="none", label="ideal splitting")
    for strat, mk, col, lab, dx in (("portfolio", "o", "#1f5fa8", "portfolio", -0.1), ("sus", "s", "#c2571a", "splitting", 0.1)):
        for r in rows:
            g = r[strat]["G"]
            err = [[g - r[strat]["lo"]], [r[strat]["hi"] - g]]
            filled = r["eps"] in (None, 0.0)
            de = {0.01: -0.04, 0.0: 0.04}.get(r["eps"], 0.0) if r["k"] == 3 else 0.0   # coupled and uncoupled k = 3 side by side
            ax.errorbar(r["k"] + dx + de, g, yerr=err, fmt=mk, color=col, mfc=col if filled else "white", ms=4, capsize=2,
                        lw=0.9, label=lab if r["k"] == 1 else None)
    ax.axhline(1, color="0.6", lw=0.8, ls=":")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.75, 4.25)
    ax.set_ylim(0.6, 3e7)
    ax.set_xlabel("number of coupled copies $k$ (dimension $3k$)")
    ax.set_ylabel(r"gain $G_{0.9}$")
    ax.legend(loc="upper left", frameon=True, framealpha=1.0, edgecolor="#d0d0d0", fancybox=False,
              borderaxespad=0.4, labelspacing=0.3, handlelength=1.8)
    fig.tight_layout()
    pad = 2 * float(plt.rcParams["savefig.pad_inches"])
    for _ in range(6):                   # the tight bounding box must fit the column, so that LaTeX does not scale
        fig.canvas.draw()
        w = fig.get_tightbbox(fig.canvas.get_renderer()).width + pad
        if abs(w - width) <= 0.01:
            break
        fig.set_size_inches(fig.get_size_inches()[0] - (w - width), fig.get_size_inches()[1])
        fig.tight_layout()
    d = ROOT / "paper" / "siads" / "figures"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / "fig_product_gain.pdf")
    fig.savefig(d / "fig_product_gain.png")


if __name__ == "__main__":
    if "--figure" in sys.argv:           # redraw from the saved results only
        figure(json.loads((RES / "e68_product_theory.json").read_text(encoding="utf-8")))
    else:
        main()
