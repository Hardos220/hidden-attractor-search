"""Рисунок и таблица статьи о стоимости поиска (батчи B13, B14, B17, B18): выигрыш портфеля по числу интегрирований
(G_N), по шагам интегратора (G_S) и по времени (G_T) относительно перебора с той же двухэтапной оценкой.

Итоговая конфигурация метода — портфель с быстрым kNN и двухэтапной оценкой (portfolio_fast__scr); у связанных систем —
с эталонами-произведениями (_prod). Если итоговая конфигурация на задаче ещё не измерена, берётся portfolio__scr (на
задачах с дорогим интегрированием накладные расходы kNN не больше 4 %, и числа оценок совпадают).

    python experiments/e54_cost_figure.py
пишет results/figures/fig30_cost_gains.{png,pdf} (RU), paper/chaos/figures/fig30_cost_gains.{png,pdf} (EN),
results/e53_snippets/tab_cost.{tex,html} и results/e54_cost_final.json.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hsearch.viz import INK, INK2, MUTED, SERIES, save, setup  # noqa: E402

BATCHES = ["B13", "B14", "B17", "B18", "B19", "B20"]
# (вариант задачи, подпись EN, подпись RU, размерность, p)
PROBLEMS = [
    (["chua_I"], "Chua I", "Чуа I", 3),
    (["se11"], "SE11", "SE11", 3),
    (["le1"], "LE1", "LE1", 3),
    (["edge_se3_m3_p0.0001"], "D2, $p=10^{-4}$", "D2, p = 10⁻⁴", 6),
    (["edge_se3_m3_p1e-05"], "D2, $p=10^{-5}$", "D2, p = 10⁻⁵", 6),
    (["coupled_se11_k2_eps0.01_c1_prod", "coupled_se11_k2_eps0.01_c1"], "2 coupled SE11", "2 связанные SE11", 6),
    (["coupled_se11_k3_eps0.01_c1_prod", "coupled_se11_k3_eps0.01_c1"], "3 coupled SE11", "3 связанные SE11", 9),
    (["chua_II"], "Chua II", "Чуа II", 3),
    (["coupled_chua_II_k2_eps0.1_c0_prod_eq"], "2 Chua II, $\epsilon=0.1$", "2 цепи Чуа II, ε = 0.1", 6),
    (["coupled_chua_II_k2_eps0_c0_prod", "coupled_chua_II_k2_eps0_c0"], "2 Chua II, $\epsilon=0$", "2 × Чуа II, ε = 0", 6),
    (["coupled_chua_II_k3_eps0_c0"], "3 Chua II, $\epsilon=0$", "3 × Чуа II, ε = 0", 9),
]
LABELS = ["portfolio_cap__scr", "portfolio_fast__scr", "portfolio__scr"]
P_REF = {"chua_I": 3.3e-2, "chua_II": 2.6e-3, "se11": 7.7e-3, "le1": 1.3e-3, "edge_se3_m3_p0.0001": 1e-4,
         "edge_se3_m3_p1e-05": 1e-5, "coupled_se11_k2_eps0.01_c1": 5.5e-5, "coupled_se11_k3_eps0.01_c1": 3.5e-7,
         "coupled_chua_II_k2_eps0_c0": (0.00142 + 0.00126) ** 2, "coupled_chua_II_k3_eps0_c0": (0.00142 + 0.00126) ** 3,
         "coupled_chua_II_k2_eps0.1_c0_prod_eq": 2.3e-5}


def rows():
    subprocess.run([sys.executable, str(ROOT / "experiments" / "e54_cost_table.py")] + BATCHES, check=True,
                   capture_output=True)
    return json.loads((ROOT / "results" / f"e54_cost_{'_'.join(BATCHES)}.json").read_text(encoding="utf-8"))


# Время сравнимо только при одинаковой загрузке машины: B13, B14 и B17 шли пятью процессами по четыре потока одни,
# B18 — одним процессом рядом с B17. Поэтому для B18 перебор берётся только из того же батча.
FULL_LOAD = {"B13", "B14", "B17", "B19", "B20"}


def comparable(b_m, b_u):
    return b_m == b_u or (b_m in FULL_LOAD and b_u in FULL_LOAD)


def pick(all_rows):
    out = []
    for variants, en, ru, n in PROBLEMS:
        hit = u = None
        for v in variants:
            for lab in LABELS:
                for c in sorted([r for r in all_rows if r["system"] == v and r["label"] == lab], key=lambda r: -r["R"]):
                    us = [r for r in all_rows if r["system"] == v and r["label"] == "uniform__scr"
                          and comparable(c["batch"], r["batch"])]
                    if us:
                        hit = c
                        u = sorted(us, key=lambda r: (r["batch"] != c["batch"], -r["R"]))[0]
                        break
                if hit:
                    break
            if hit:
                break
        if not hit:
            continue
        g = {k: (u[q] / hit[q] if hit[q] and math.isfinite(hit[q]) else None)
             for k, q in (("G_N", "N09"), ("G_S", "S09"), ("G_T", "T09"))}
        level = 0.9
        if hit["success"] <= 0.9 and u.get("uniform_from_p"):
            # часть запусков исчерпала бюджет: выигрыш на уровне доли успеха (эмпирическая обратная функция
            # распределения по запускам; перебор — ln(1/(1-q))/p оценок при измеренной стоимости одной оценки)
            level = math.floor(hit["success"] * 10) / 10
            recs = [json.loads(x) for x in (ROOT / "results" / "e50" / hit["batch"] / f"{hit['system']}__{hit['label']}.jsonl")
                    .read_text(encoding="utf-8").splitlines() if x.strip()]
            fin = sorted(((r["found_at"], r["steps"] * r["found_at"] / r["n_eval"], r["time_s"] * r["found_at"] / r["n_eval"])
                          for r in recs if r["found_at"]), key=lambda t: t[0])
            i = math.ceil(level * len(recs)) - 1
            nu = math.log(1.0 / (1.0 - level)) / u["uniform_from_p"]
            g = {"G_N": nu / fin[i][0], "G_S": nu * u["steps_per_eval"] / fin[i][1],
                 "G_T": nu * u["ms_per_eval"] / 1e3 / fin[i][2]}
        base = hit["system"].replace("_prod", "")
        p_ref = P_REF.get(hit["system"], P_REF.get(base, u.get("uniform_from_p")))
        out.append({"en": en, "ru": ru, "n": n, "system": hit["system"], "label": hit["label"], "batch": hit["batch"],
                    "batch_u": u["batch"], "R": hit["R"], "p": p_ref, "success": hit["success"],
                    "N09": hit["N09"], "N09u": u["N09"], "S_u": u["steps_per_eval"],
                    "S_m": hit["steps_per_eval"], "ms_u": u["ms_per_eval"], "ms_m": hit["ms_per_eval"],
                    "T09": hit["T09"], "T09u": u["T09"], **g, "level": level, "overhead": hit["overhead_frac"],
                    "stopped_u": u.get("stopped_frac"), "stopped_m": hit.get("stopped_frac")})
    return out


T = {"en": {"x": "gain relative to uniform sampling", "N": "integrations $G_N$",
            "S": "integrator steps $G_S$", "T": "wall time $G_T$"},
     "ru": {"x": "выигрыш относительно перебора", "N": "интегрирования \\(G_N\\)",
            "S": "шаги интегратора \\(G_S\\)", "T": "время \\(G_T\\)"}}


def figure(sel, lang):
    setup()
    w, h = (3.37, 3.1) if lang == "en" else (5.0, 4.4)
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(sel))[::-1]
    col = {"G_N": MUTED, "G_S": SERIES[1], "G_T": SERIES[0]}
    mk = {"G_N": "o", "G_S": "s", "G_T": "D"}
    lab = {"G_N": T[lang]["N"].replace("\\(", "$").replace("\\)", "$"),
           "G_S": T[lang]["S"].replace("\\(", "$").replace("\\)", "$"),
           "G_T": T[lang]["T"].replace("\\(", "$").replace("\\)", "$")}
    for yi, r in zip(y, sel):
        vals = [r[k] for k in ("G_N", "G_S", "G_T") if r[k]]
        ax.plot([min(vals), max(vals)], [yi, yi], "-", color=INK2, lw=0.8, alpha=0.6, zorder=1)
        for k in ("G_N", "G_S", "G_T"):
            if r[k]:
                ax.plot(r[k], yi, mk[k], color=col[k], ms=5 if k != "G_T" else 5.5, mec="white", mew=0.6, zorder=3,
                        label=lab[k] if yi == y[0] else None)
    ax.axvline(1.0, color=INK, lw=0.8, ls="--", zorder=0)
    # граница групп: выше — системы, где большинство траекторий уходит на бесконечность; ниже — ограниченные
    nb = sum(1 for r in sel if r["system"].startswith(("chua_II", "coupled_chua")))
    if 0 < nb < len(sel):
        ax.axhline(nb - 0.5, color=MUTED, lw=0.6, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([r[lang] + ("*" if r["level"] < 0.9 else "") for r in sel], fontsize=7.5 if lang == "en" else 9)
    ax.set_xlabel(T[lang]["x"], fontsize=7.5 if lang == "en" else 9)
    ax.tick_params(axis="x", labelsize=7.5 if lang == "en" else 9)
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=6.5 if lang == "en" else 8, loc="upper right", handletextpad=0.3, borderaxespad=0.2)
    fig.tight_layout()
    return fig


def fmt(x, lang, nd=3):
    if x is None or not math.isfinite(x):
        return "---" if lang == "en" else "—"
    if x >= 1e4:
        e = int(math.floor(math.log10(x)))
        m = x / 10 ** e
        return f"${m:.1f}\\times10^{{{e}}}$" if lang == "en" else f"{m:.1f}·10{str(e).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
    s = f"{x:.{nd}g}"
    if "e" in s:
        s = f"{x:.0f}"
    return s


def gfmt(x, lang):
    """Выигрыш: две значащие цифры (0.27, 3.8, 22, 120, 2600), как в остальных таблицах статьи."""
    if x is None or not math.isfinite(x):
        return "---" if lang == "en" else "—"
    if x >= 100:
        e = int(math.floor(math.log10(x)))
        v = int(round(x / 10 ** (e - 1)) * 10 ** (e - 1))
        return f"{v:,}".replace(",", "\\," if lang == "en" else " ") if v >= 10000 else str(v)
    if x >= 10:
        return f"{x:.0f}"
    if x >= 1:
        return f"{x:.1f}"
    if x >= 0.1:
        return f"{x:.2f}"
    return f"{x:.2g}"


def pfmt(p, lang):
    e = int(math.floor(math.log10(p)))
    m = p / 10 ** e
    if lang == "en":
        return f"${m:.1f}\\times10^{{{e}}}$"
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{m:.1f}·10{str(e).translate(sup)}"


def tables(sel):
    en, ru = [], []
    for r in sel:
        cells = [r["en"] + ("$^{*}$" if r["level"] < 0.9 else ""), str(r["n"]), pfmt(r["p"], "en"), fmt(r["S_u"], "en", 2), fmt(r["S_m"], "en", 2),
                 gfmt(r["G_N"], "en"), gfmt(r["G_S"], "en"),
                 "\\textbf{" + gfmt(r["G_T"], "en") + "}" if r["G_T"] and r["G_T"] > 1 else gfmt(r["G_T"], "en")]
        en.append(" & ".join(cells) + " \\\\")
        cells = [r["ru"] + ("*" if r["level"] < 0.9 else ""), str(r["n"]), pfmt(r["p"], "ru"), fmt(r["S_u"], "ru", 2), fmt(r["S_m"], "ru", 2),
                 gfmt(r["G_N"], "ru"), gfmt(r["G_S"], "ru"), gfmt(r["G_T"], "ru")]
        b = r["G_T"] and r["G_T"] > 1
        ru.append("<tr>" + f"<td>{cells[0]}</td>" + "".join(f'<td class="num">{c}</td>' for c in cells[1:-1])
                  + (f'<td class="num"><b>{cells[-1]}</b></td>' if b else f'<td class="num">{cells[-1]}</td>') + "</tr>")
    sn = ROOT / "results" / "e53_snippets"
    sn.mkdir(exist_ok=True)
    (sn / "tab_cost.tex").write_text("\n".join(en) + "\n", encoding="utf-8")
    (sn / "tab_cost.html").write_text("\n".join(ru) + "\n", encoding="utf-8")


def main():
    sel = pick(rows())
    (ROOT / "results" / "e54_cost_final.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in sel:
        print(f"{r['en']:18s} {r['label']:20s} {r['batch']}/{r['batch_u']} R={r['R']:3d} N09={r['N09']:.4g} Nu={r['N09u']:.4g} "
              f"S_u={r['S_u']:.4g} S_m={r['S_m']:.4g} G_N={r['G_N']:.3g} G_S={r['G_S']:.3g} G_T={r['G_T']:.3g} "
              f"ovh={r['overhead']:.2f}")
    tables(sel)
    for lang, d in (("ru", ROOT / "results" / "figures"), ("en", ROOT / "paper" / "chaos" / "figures")):
        save(figure(sel, lang), d / "fig30_cost_gains")


if __name__ == "__main__":
    main()
