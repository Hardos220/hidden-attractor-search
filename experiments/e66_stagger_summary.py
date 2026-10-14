"""Stagger-and-step against the portfolio: summary of e65 (Chua II, SE11, LE1, D2 p = 1e-4; e20 runs with tag _stagger)
and batch B23 (failure problems, D2 scaling in p, D1, three coupled copies of SE11).

    python experiments/e66_stagger_summary.py

G = N0_0.9 / N_0.9 relative to the uniform reference of the article (fixed per problem, Tables II-IV); the interval is
the 95% bootstrap interval over the stagger runs (2000 resamples; a resample with less than 90% success gives G = 0).
The scaling exponent s of stagger on D2 is the slope of ln N_0.9 on ln(1/p) over p = 1e-2 ... 1e-5 with its bootstrap
interval. Writes results/e66_stagger_summary.{json,md} and results/e53_snippets/tab_stagger.tex (rows of the appendix
table of the article).
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_BOOT = 2000

# problem: (label, uniform N0 of the article, G of splitting, G of the portfolio, source of the runs)
PROBLEMS = [
    ("chua_I", "Chua I", 66, 1.20, 1.24, "B23"),
    ("chua_II", "Chua II", 818, 5.05, 7.36, "e65"),
    ("se11", "SE11", 317, 2.53, 3.10, "e65"),
    ("le1", "LE1", 1617, 8.29, 6.24, "e65"),
    ("jafari2026_lc", "Jafari 2026, cycle", 219, 0.29, 0.43, "B23"),
    ("huynh2019", "Huynh 2019", 221, 1.36, 1.40, "B23"),
    ("jafari2026_chaos", "Jafari 2026, chaos", 5265, 0.22, 0.42, "B23"),
    ("emb_se3_m3_p0.001", "D1, $p=10^{-3}$", 1932, 0.69, 1.19, "B23"),
    ("edge_se3_m3_p0.01", "D2, $p=10^{-2}$", 277, 2.29, 2.27, "B23"),
    ("edge_se3_m3_p0.001", "D2, $p=10^{-3}$", 1932, 7.37, 5.87, "B23"),
    ("edge_se3_m3_p0.0001", "D2, $p=10^{-4}$", 17416, 24.4, 17.0, "e65"),
    ("edge_se3_m3_p1e-05", "D2, $p=10^{-5}$", 189765, 93.1, 72.0, "B23"),
    ("coupled_se11_k3_eps0.01_c1", "3 coupled SE11", 7.6e6, 682, 1435, "B23"),
]
D2 = [("edge_se3_m3_p0.01", 1e-2), ("edge_se3_m3_p0.001", 1e-3), ("edge_se3_m3_p0.0001", 1e-4), ("edge_se3_m3_p1e-05", 1e-5)]


def found(name, src):
    if src == "e65":
        d = json.loads((RES / f"e20_search_{name}_stagger.json").read_text(encoding="utf-8"))
        return [None if x is None else float(x) for x in d["stagger"]["found_at"]], d["_meta"]["budget"]
    f = RES / "e50" / "B23" / f"{name}__stagger.jsonl"
    if not f.exists():
        return [], None
    runs = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    runs = [r for r in runs if "error" not in r]
    return [None if r.get("found_at") is None else float(r["found_at"]) for r in runs], (runs[0].get("budget") if runs else None)


def q09(a):
    a = np.asarray(a, float)
    return float(np.quantile(a, 0.9)) if np.mean(np.isfinite(a)) >= 0.9 else math.inf


def gain_ci(fa, n0, rng):
    a = np.array([math.inf if x is None else x for x in fa], float)
    n9 = q09(a)
    g = n0 / n9 if math.isfinite(n9) else 0.0
    gb = []
    for _ in range(N_BOOT):
        qb = q09(rng.choice(a, len(a)))
        gb.append(n0 / qb if math.isfinite(qb) else 0.0)
    return g, float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5)), n9, float(np.mean(np.isfinite(a)))


def fmt(v):
    if v == 0:
        return "0"
    if v >= 1000:
        return f"{v:.0f}" if v < 1e4 else f"{v:.2g}".replace("e+0", "e").replace("e+", "e")
    if v >= 10:
        return f"{v:.0f}" if v >= 100 else f"{v:.1f}"
    return f"{v:.2f}"


def tex_num(v):
    s = fmt(v)
    if "e" in s:
        m, e = s.split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return s


def main():
    rng = np.random.default_rng(66)
    out, rows_md, rows_tex = {}, [], []
    for name, lab, n0, g_sus, g_port, src in PROBLEMS:
        fa, budget = found(name, src)
        if not fa:
            rows_md.append(f"| {lab} | — | — | — | — | — | {g_sus} | {g_port} |")
            continue
        g, lo, hi, n9, succ = gain_ci(fa, n0, rng)
        out[name] = {"label": lab, "R": len(fa), "budget": budget, "success": succ, "N09": n9, "N0": n0, "G": g,
                     "lo": lo, "hi": hi, "G_splitting": g_sus, "G_portfolio": g_port, "source": src}
        n9s = "inf" if not math.isfinite(n9) else f"{n9:.0f}"
        rows_md.append(f"| {lab} | {len(fa)} | {budget} | {succ:.2f} | {n9s} | {g:.2f} [{lo:.2f}, {hi:.2f}] | {g_sus} | {g_port} |")
        cell = (f"\\textbf{{{tex_num(g)}}}" if lo > 1 else tex_num(g)) + f" \\ci{{{fmt(lo)}}}{{{fmt(hi)}}}"
        if succ < 0.9:
            cell = f"--- ({100 * succ:.0f}\\%)"
        rows_tex.append(f"{lab} & {len(fa)} & {cell} & {tex_num(g_sus)} & {tex_num(g_port)} \\\\")
    # scaling exponent of stagger on D2
    arrs, ps = [], []
    for name, p in D2:
        fa, _ = found(name, "e65" if name == "edge_se3_m3_p0.0001" else "B23")
        if fa:
            arrs.append(np.array([math.inf if x is None else x for x in fa], float))
            ps.append(p)
    s_info = None
    if len(arrs) == len(D2) and all(math.isfinite(q09(a)) for a in arrs):
        x = np.log(1 / np.array(ps))

        def slope(qs):
            return float(np.polyfit(x, np.log(qs), 1)[0])
        s0 = slope([q09(a) for a in arrs])
        sb = []
        for _ in range(N_BOOT):
            qs = [q09(rng.choice(a, len(a))) for a in arrs]
            if all(math.isfinite(v) for v in qs):
                sb.append(slope(qs))
        s_info = {"s": s0, "lo": float(np.percentile(sb, 2.5)), "hi": float(np.percentile(sb, 97.5)), "points": len(ps)}
        out["_scaling_D2"] = s_info
    (RES / "e66_stagger_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    md = ["# e66: stagger-and-step против портфеля", "",
          "G = N0_0.9 (перебор статьи) / N_0.9 stagger; интервал — бутстрап по запускам stagger (2000 перевыборок).", "",
          "| задача | R | бюджет | успех | N_0.9 | G stagger [95 %] | G расщепления | G портфеля |", "|---|---|---|---|---|---|---|---|"]
    md += rows_md
    if s_info:
        md += ["", f"Показатель скейлинга stagger на D2 (p = 1e-2…1e-5): s = {s_info['s']:.2f} [{s_info['lo']:.2f}, {s_info['hi']:.2f}]"]
    (RES / "e66_stagger_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RES / "e53_snippets" / "tab_stagger.tex").write_text("\n".join(rows_tex) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
