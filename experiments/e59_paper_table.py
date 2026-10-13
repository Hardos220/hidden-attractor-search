"""Таблица статьи: время до попадания у перебора с разными способами оценки траектории (полный счёт, двухэтапная оценка,
рекуррентный метод Attractors.jl) и у портфеля, один поток (e59 + B21). Пишет results/e53_snippets/tab_time.{tex,html}.

Для перебора T_0.9 = N_0.9 × средняя стоимость оценки (N_0.9 — из измеренных запусков перебора); для портфеля — измеренное
время запусков на одном потоке (B21, R = 20). Attractors.jl — лучшая из настроек (Tsit5, consecutive_recurrences = 100).
"""
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E59 = ROOT / "results" / "e59"
ROWS = [("chua_II", "Chua II", "Чуа II", ("B13", "chua_II__uniform__scr"), "chua_II__portfolio_fast__scr", 2.6e-3),
        ("coupled_chua_II_eps0.1", "2 Chua II, $\\epsilon=0.1$", "2 цепи Чуа II, ε = 0.1",
         ("B19", "coupled_chua_II_k2_eps0.1_c0_prod_eq__uniform__scr"), "coupled_chua_II_k2_eps0.1_c0_prod_eq__portfolio_fast__scr", 2.3e-5),
        ("se11", "SE11", "SE11", ("B13", "se11__uniform__scr"), "se11__portfolio_fast__scr", 7.7e-3),
        ("le1", "LE1", "LE1", ("B13", "le1__uniform__scr"), "le1__portfolio_fast__scr", 1.3e-3)]


def runs(bid, jid):
    p = ROOT / "results" / "e50" / bid / f"{jid}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and "error" not in l]


def q09(v):
    return float(np.quantile(np.array(v, float), 0.9))


def fmt_t(x):
    if x < 10:
        return f"{x:.2g}"
    if x < 1000:
        return f"{x:.0f}"
    return f"{x:.2g}".replace("e+0", "e").replace("e+", "e")


def fmt_g(x):
    return f"{x:.2g}" if x < 1 else f"{x:.1f}"


def fmt_c(x):
    return f"{x:.3g}" if x < 10 else f"{x:.1f}"


def main():
    py = json.loads((E59 / "cost_python.json").read_text(encoding="utf-8"))
    jl = json.loads((E59 / "cost_julia.json").read_text(encoding="utf-8"))
    tex, html = [], []
    for key, en, ru, (ub, uj), pj, p in ROWS:
        u = runs(ub, uj)
        n09 = q09([r["found_at"] or math.inf for r in u])
        pf = runs("B21", pj)
        n09p = q09([r["found_at"] or math.inf for r in pf])
        t09p = q09([r["time_s"] * r["found_at"] / r["n_eval"] if r["found_at"] else math.inf for r in pf])
        c_full = py[key]["full"]["ms_per_ic"]
        c_scr = py[key]["screened"]["ms_per_ic"]
        c_jl = min(v["ms_per_ic"] for v in jl[key].values())
        t_full, t_scr, t_jl = n09 * c_full / 1e3, n09 * c_scr / 1e3, n09 * c_jl / 1e3
        g_scr, g_jl = t_scr / t09p, t_jl / t09p
        pe = f"${p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}$"
        pr = f"{p:.1e}".replace("e-0", "·10⁻").replace("e-", "·10⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")).replace("¹.", "1.")
        pr = f"{p:.1e}"
        m, e = pr.split("e")
        pr = f"{m}·10{str(int(e)).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
        bold = lambda x, c: (f"\\textbf{{{x}}}" if c else x)
        tex.append(f"{en} & {pe} & {n09:.0f} & {fmt_c(c_full)} & {fmt_c(c_scr)} & {fmt_c(c_jl)} & {fmt_t(t_full)} & {fmt_t(t_scr)} & {fmt_t(t_jl)} & "
                   f"{n09p:.0f} & {fmt_t(t09p)} & {bold(fmt_g(g_scr), g_scr > 1)} & {bold(fmt_g(g_jl), g_jl > 1)} \\\\")
        cells = [ru, pr, f"{n09:.0f}", fmt_c(c_full), fmt_c(c_scr), fmt_c(c_jl), fmt_t(t_full), fmt_t(t_scr), fmt_t(t_jl),
                 f"{n09p:.0f}", fmt_t(t09p), fmt_g(g_scr), fmt_g(g_jl)]
        html.append("<tr><td>" + cells[0] + "</td>" + "".join(
            f'<td class="num">{("<b>" + c + "</b>") if (i >= 10 and float(c) > 1) else c}</td>' for i, c in enumerate(cells[1:], 1)) + "</tr>")
    sn = ROOT / "results" / "e53_snippets"
    (sn / "tab_time.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (sn / "tab_time.html").write_text("\n".join(html) + "\n", encoding="utf-8")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
