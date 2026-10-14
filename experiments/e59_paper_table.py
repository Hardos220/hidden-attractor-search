"""Таблица статьи: время до попадания у перебора с разными способами оценки траектории (полный счёт, двухэтапная оценка,
рекуррентный метод Attractors.jl) и у портфеля, один поток (e59 + B21). Пишет results/e53_snippets/tab_time.{tex,html}
и results/e59/paper_table.json.

Для перебора T_0.9 = N0_0.9 × средняя стоимость оценки, где N0_0.9 — фиксированная точка отсчёта статьи (e54_cost_figure.REF_N0:
Чуа II 818, SE11 317, LE1 1617; две цепи Чуа II при eps = 0.1 — измеренное по запускам перебора B19, R = 20); для портфеля —
измеренное время запусков на одном потоке (B21, R = 20). Attractors.jl — лучшая из настроек (Tsit5, consecutive_recurrences = 100).
Выигрыши по времени даются с интервалами 95 % (бутстрап по запускам портфеля, 2000 перевыборок; для измеренного N0 — и по
запускам перебора); жирным — только при нижней границе выше 1.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

from e54_cost_figure import N_BOOT, REF_N0, pfmt, q09, sci  # noqa: E402

E59 = ROOT / "results" / "e59"
ROWS = [("chua_II", "Chua II", "Чуа II", ("B13", "chua_II__uniform__scr"), "chua_II__portfolio_fast__scr", 2.6e-3, "chua_II"),
        ("coupled_chua_II_eps0.1", "2 Chua II, $\\epsilon=0.1$", "2 цепи Чуа II, ε = 0.1",
         ("B19", "coupled_chua_II_k2_eps0.1_c0_prod_eq__uniform__scr"), "coupled_chua_II_k2_eps0.1_c0_prod_eq__portfolio_fast__scr", 2.3e-5,
         "coupled_chua_II_k2_eps0.1_c0_prod_eq"),
        ("se11", "SE11", "SE11", ("B13", "se11__uniform__scr"), "se11__portfolio_fast__scr", 7.7e-3, "se11"),
        ("le1", "LE1", "LE1", ("B13", "le1__uniform__scr"), "le1__portfolio_fast__scr", 1.3e-3, "le1")]


def runs(bid, jid):
    p = ROOT / "results" / "e50" / bid / f"{jid}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip() and "error" not in l]


def fmt_t(x, lang):
    """Время, с: 2 значащие до 10, целое до 1000, дальше a×10^k (никогда не 7.6e3)."""
    if x < 10:
        return f"{x:.2g}"
    if x < 1000:
        return f"{x:.0f}"
    return sci(x, lang, 2)


def fmt_n(x, lang):
    """Число интегрирований: целое до 10^4, дальше a×10^k с тремя значащими цифрами (1.05×10^5)."""
    return f"{x:.0f}" if x < 1e4 else sci(x, lang, 3)


def fmt_g(x):
    return f"{x:.2g}" if x < 1 else f"{x:.1f}"


def fmt_c(x):
    return f"{x:.3g}" if x < 10 else f"{x:.1f}"


def gcell(g, lo, hi, lang):
    s = fmt_g(g)
    if lang == "en":
        return (f"\\textbf{{{s}}}" if lo > 1 else s) + f" \\ci{{{fmt_g(lo)}}}{{{fmt_g(hi)}}}"
    return (f"<b>{s}</b>" if lo > 1 else s) + f" [{fmt_g(lo)}, {fmt_g(hi)}]"


def main():
    py = json.loads((E59 / "cost_python.json").read_text(encoding="utf-8"))
    jl = json.loads((E59 / "cost_julia.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    tex, html, out = [], [], []
    for key, en, ru, (ub, uj), pj, p, ref in ROWS:
        n0, src = REF_N0[ref]
        fa_u = None
        if n0 == "measured":
            fa_u = np.array([r["found_at"] or math.inf for r in runs(ub, uj)], float)
            n0 = q09(fa_u)
        pf = runs("B21", pj)
        fa = np.array([r["found_at"] or math.inf for r in pf], float)
        tt = np.array([r["time_s"] * r["found_at"] / r["n_eval"] if r["found_at"] else math.inf for r in pf], float)
        n09p, t09p = q09(fa), q09(tt)
        c_full = py[key]["full"]["ms_per_ic"]
        c_scr = py[key]["screened"]["ms_per_ic"]
        c_jl = min(v["ms_per_ic"] for v in jl[key].values())
        t_full, t_scr, t_jl = n0 * c_full / 1e3, n0 * c_scr / 1e3, n0 * c_jl / 1e3
        g_scr, g_jl = t_scr / t09p, t_jl / t09p
        boots = np.zeros((N_BOOT, 2))
        for b in range(N_BOOT):
            idx = rng.integers(0, len(pf), len(pf))
            n0b = n0 if fa_u is None else q09(fa_u[rng.integers(0, len(fa_u), len(fa_u))])
            tb = q09(tt[idx])
            boots[b] = (n0b * c_scr / 1e3 / tb, n0b * c_jl / 1e3 / tb) if np.isfinite(tb) and tb > 0 else (0.0, 0.0)
        (lo_s, lo_j), (hi_s, hi_j) = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
        # the uniform times T0.9 = N0 x cost per evaluation are not printed in the English table (column width; the
        # caption gives the rule); the Russian html keeps them
        tex.append(f"{en} & {pfmt(p, 'en')} & {fmt_n(n0, 'en')} & {fmt_c(c_full)} & {fmt_c(c_scr)} & {fmt_c(c_jl)} & "
                   f"{n09p:.0f} & {fmt_t(t09p, 'en')} & {gcell(g_scr, lo_s, hi_s, 'en')} & "
                   f"{gcell(g_jl, lo_j, hi_j, 'en')} \\\\")
        cells = [ru, pfmt(p, "ru"), fmt_n(n0, "ru"), fmt_c(c_full), fmt_c(c_scr), fmt_c(c_jl), fmt_t(t_full, "ru"), fmt_t(t_scr, "ru"),
                 fmt_t(t_jl, "ru"), f"{n09p:.0f}", fmt_t(t09p, "ru"), gcell(g_scr, lo_s, hi_s, "ru"), gcell(g_jl, lo_j, hi_j, "ru")]
        html.append("<tr><td>" + cells[0] + "</td>" + "".join(f'<td class="num">{c}</td>' for c in cells[1:]) + "</tr>")
        out.append({"key": key, "p": p, "N0_ref": float(n0), "N0_source": src, "ms_full": c_full, "ms_scr": c_scr, "ms_jl": c_jl,
                    "T_full": t_full, "T_scr": t_scr, "T_jl": t_jl, "N09_portfolio": n09p, "T09_portfolio": t09p,
                    "G_T_scr": g_scr, "G_T_scr_ci": [float(lo_s), float(hi_s)], "G_T_jl": g_jl, "G_T_jl_ci": [float(lo_j), float(hi_j)],
                    "R_portfolio": len(pf), "n_boot": N_BOOT})
    sn = ROOT / "results" / "e53_snippets"
    (sn / "tab_time.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    (sn / "tab_time.html").write_text("\n".join(html) + "\n", encoding="utf-8")
    (E59 / "paper_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(tex))


if __name__ == "__main__":
    main()
