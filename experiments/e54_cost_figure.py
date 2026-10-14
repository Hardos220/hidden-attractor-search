"""Рисунок и таблица статьи о стоимости поиска (батчи B13, B14, B17, B18, B19, B20): выигрыш портфеля по числу
интегрирований (G_N), по шагам интегратора (G_S) и по времени (G_T) относительно перебора с той же двухэтапной оценкой.

Итоговая конфигурация метода — портфель с быстрым kNN и двухэтапной оценкой (portfolio_fast__scr); у связанных систем —
с эталонами-произведениями (_prod). Если итоговая конфигурация на задаче ещё не измерена, берётся portfolio__scr (на
задачах с дорогим интегрированием накладные расходы kNN не больше 4 %, и числа оценок совпадают).

Точка отсчёта (2026-10-13): для каждой задачи стоимость перебора N0_0.9 — фиксированное значение REF_N0 (таблицы статьи:
перебор с R = 50–100; для двух цепей Чуа II при eps = 0.1 — измеренное в B19, R = 20; для независимых копий Чуа II —
ln 10 / p с p = 0.0027^k), а его шаги и время — N0 × измеренные средние шаги и время на одну оценку перебора того же
батча. Интервалы 95 % — бутстрап по запускам портфеля (2000 перевыборок; для измеренного N0 — и по запускам перебора).

    python experiments/e54_cost_figure.py
пишет results/figures/fig30_cost_gains.{png,pdf} (RU), paper/chaos/figures/fig30_cost_gains.{png,pdf} (EN),
results/e53_snippets/tab_cost.{tex,html}, results/e54_cost_final.json и results/e54_cost_final.md (старые и новые значения).
"""
import json
import math
import shutil
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
N_BOOT = 2000
# (вариант задачи, подпись EN, подпись RU, размерность)
PROBLEMS = [
    (["chua_I"], "Chua I", "Чуа I", 3),
    (["se11"], "SE11", "SE11", 3),
    (["le1"], "LE1", "LE1", 3),
    (["edge_se3_m3_p0.0001"], "D2, $p=10^{-4}$", "D2, p = 10⁻⁴", 6),
    (["edge_se3_m3_p1e-05"], "D2, $p=10^{-5}$", "D2, p = 10⁻⁵", 6),
    (["coupled_se11_k2_eps0.01_c1_prod", "coupled_se11_k2_eps0.01_c1"], "2 coupled SE11", "2 связанные SE11", 6),
    (["coupled_se11_k3_eps0.01_c1_prod", "coupled_se11_k3_eps0.01_c1"], "3 coupled SE11", "3 связанные SE11", 9),
    (["chua_II"], "Chua II", "Чуа II", 3),
    (["coupled_chua_II_k2_eps0.1_c0_prod_eq"], "2 Chua II, $\\epsilon=0.1$", "2 цепи Чуа II, ε = 0.1", 6),
    (["coupled_chua_II_k2_eps0_c0_prod", "coupled_chua_II_k2_eps0_c0"], "2 Chua II, $\\epsilon=0$", "2 × Чуа II, ε = 0", 6),
    (["coupled_chua_II_k3_eps0_c0"], "3 Chua II, $\\epsilon=0$", "3 × Чуа II, ε = 0", 9),
]
LABELS = ["portfolio_cap__scr", "portfolio_fast__scr", "portfolio__scr"]
P1_CHUA_II = 0.0027                      # доля бассейна скрытых аттракторов Чуа II (таблица II статьи)
P_REF = {"chua_I": 3.3e-2, "chua_II": 2.6e-3, "se11": 7.7e-3, "le1": 1.3e-3, "edge_se3_m3_p0.0001": 1e-4,
         "edge_se3_m3_p1e-05": 1e-5, "coupled_se11_k2_eps0.01_c1": 5.5e-5, "coupled_se11_k3_eps0.01_c1": 3.5e-7,
         "coupled_chua_II_k2_eps0_c0": P1_CHUA_II ** 2, "coupled_chua_II_k3_eps0_c0": P1_CHUA_II ** 3,
         "coupled_chua_II_k2_eps0.1_c0_prod_eq": 2.3e-5}
# фиксированная стоимость перебора N0_0.9 (число интегрирований) и источник каждого значения
REF_N0 = {
    "chua_I": (66.0, "таблица II статьи: перебор, R = 100 (B1a)"),
    "chua_II": (818.0, "таблица II статьи: перебор, R = 100 (B1a)"),
    "se11": (317.0, "таблица II статьи: перебор, R = 100 (B1a)"),
    "le1": (1617.0, "таблица II статьи: перебор, R = 100 (B1a)"),
    "edge_se3_m3_p0.0001": (17416.0, "семейство D2 (масштабирование по p): перебор, R = 100 (B1b: 17416.3)"),
    "edge_se3_m3_p1e-05": (189765.0, "семейство D2 (масштабирование по p): перебор, R = 100 (B1b: 189764.8)"),
    "coupled_se11_k2_eps0.01_c1": (3.8e4, "таблица связанных копий SE11: перебор, R = 50 (B4a: 37751)"),
    "coupled_se11_k3_eps0.01_c1": (7.6e6, "таблица связанных копий SE11: перебор, R = 100 (B8: 7574940)"),
    "coupled_chua_II_k2_eps0.1_c0_prod_eq": ("measured", "измерено: перебор с двухэтапной оценкой, B19, R = 20 (104980); "
                                                        "бутстрап и по запускам перебора"),
    "coupled_chua_II_k2_eps0_c0": (math.log(10.0) / P1_CHUA_II ** 2, "ln 10 / p, p = p1^2, p1 = 0.0027 (таблица II)"),
    "coupled_chua_II_k3_eps0_c0": (math.log(10.0) / P1_CHUA_II ** 3, "ln 10 / p, p = p1^3, p1 = 0.0027 (таблица II)"),
}


def rows():
    subprocess.run([sys.executable, str(ROOT / "experiments" / "e54_cost_table.py")] + BATCHES, check=True,
                   capture_output=True)
    return json.loads((ROOT / "results" / f"e54_cost_{'_'.join(BATCHES)}.json").read_text(encoding="utf-8"))


# Время сравнимо только при одинаковой загрузке машины: B13, B14 и B17 шли пятью процессами по четыре потока одни,
# B18 — одним процессом рядом с B17. Поэтому для B18 перебор берётся только из того же батча.
FULL_LOAD = {"B13", "B14", "B17", "B19", "B20"}


def comparable(b_m, b_u):
    return b_m == b_u or (b_m in FULL_LOAD and b_u in FULL_LOAD)


def ref_key(system):
    for k in (system, system.replace("_prod_eq", "_eq"), system.replace("_prod", "")):
        if k in REF_N0:
            return k
    raise KeyError(system)


def runs_of(batch, system, label):
    p = ROOT / "results" / "e50" / batch / f"{system}__{label}.jsonl"
    recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    recs = [r for r in recs if "error" not in r and "steps" in r]
    fa = np.array([r["found_at"] if r["found_at"] is not None else np.inf for r in recs], float)
    frac = np.where(np.isfinite(fa), fa / np.array([r["n_eval"] for r in recs], float), np.inf)
    st = np.array([r["steps"] for r in recs], float) * frac
    ts = np.array([r["time_s"] for r in recs], float) * frac
    return fa, st, ts


def q09(x):
    """Квантиль 0.9 по запускам (np.quantile, как в e54_cost_table); если он бесконечен, но цель нашли не меньше 90 %
    запусков (R = 10, 9 успехов), — порядковая статистика ceil(0.9 R) (как в прежней версии таблицы)."""
    x = np.asarray(x, float)
    with np.errstate(invalid="ignore"):
        v = float(np.quantile(x, 0.9))
    if not np.isfinite(v):
        k = int(math.ceil(0.9 * len(x)))
        srt = np.sort(x)
        if k <= len(x) and np.isfinite(srt[k - 1]) and np.isfinite(x).mean() >= 0.9:
            v = float(srt[k - 1])
    return v


def gains(n0, fa, st, ts, S_u, ms_u):
    qN, qS, qT = q09(fa), q09(st), q09(ts)
    g = lambda num, den: (num / den) if (den and np.isfinite(den) and den > 0) else 0.0
    return g(n0, qN), g(n0 * S_u, qS), g(n0 * ms_u / 1e3, qT), qN, qS, qT


def bootstrap(n0, fa_u, fa, st, ts, S_u, ms_u, rng):
    R = len(fa)
    out = np.zeros((N_BOOT, 3))
    for b in range(N_BOOT):
        idx = rng.integers(0, R, R)
        n0b = n0 if fa_u is None else q09(fa_u[rng.integers(0, len(fa_u), len(fa_u))])
        out[b] = gains(n0b, fa[idx], st[idx], ts[idx], S_u, ms_u)[:3]
    lo = np.percentile(out, 2.5, axis=0)
    hi = np.percentile(out, 97.5, axis=0)
    return [(float(lo[i]), float(hi[i])) for i in range(3)]


def pick(all_rows):
    out = []
    rng = np.random.default_rng(0)
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
        key = ref_key(hit["system"])
        n0, src = REF_N0[key]
        fa, st, ts = runs_of(hit["batch"], hit["system"], hit["label"])
        fa_u = None
        if n0 == "measured":
            fa_u, _, _ = runs_of(u["batch"], u["system"], u["label"])
            n0 = q09(fa_u)
        S_u, ms_u = u["steps_per_eval"], u["ms_per_eval"]
        G_N, G_S, G_T, qN, qS, qT = gains(n0, fa, st, ts, S_u, ms_u)
        ci = bootstrap(n0, fa_u, fa, st, ts, S_u, ms_u, rng)
        base = hit["system"].replace("_prod", "")
        p_ref = P_REF.get(hit["system"], P_REF.get(base, u.get("uniform_from_p")))
        out.append({"en": en, "ru": ru, "n": n, "system": hit["system"], "label": hit["label"], "batch": hit["batch"],
                    "batch_u": u["batch"], "R": hit["R"], "R_u": u["R"], "p": p_ref, "success": hit["success"],
                    "N0_ref": float(n0), "N0_source": src, "N0_measured_batch": u["N09"],
                    "N09": qN, "S09": qS, "T09": qT, "S_u": S_u, "S_m": hit["steps_per_eval"], "ms_u": ms_u, "ms_m": hit["ms_per_eval"],
                    "S09u": n0 * S_u, "T09u": n0 * ms_u / 1e3,
                    "G_N": G_N, "G_S": G_S, "G_T": G_T, "G_N_ci": list(ci[0]), "G_S_ci": list(ci[1]), "G_T_ci": list(ci[2]),
                    "n_boot": N_BOOT, "level": 0.9, "overhead": hit["overhead_frac"],
                    "stopped_u": u.get("stopped_frac"), "stopped_m": hit.get("stopped_frac")})
    return out


T = {"en": {"x": "gain relative to uniform sampling", "N": "integrations $G_N$",
            "S": "integrator steps $G_S$", "T": "wall time $G_T$", "esc": "escaping", "bnd": "bounded"},
     "ru": {"x": "выигрыш относительно перебора", "N": "интегрирования $G_N$",
            "S": "шаги интегратора $G_S$", "T": "время $G_T$", "esc": "уходящие", "bnd": "ограниченные"}}


def figure(sel, lang):
    setup()
    fs = 8 if lang == "en" else 9
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs,
                         "legend.fontsize": fs, "axes.titlesize": fs})
    w, h = (3.37, 3.5) if lang == "en" else (5.0, 4.6)
    fig, ax = plt.subplots(figsize=(w, h))
    y = np.arange(len(sel))[::-1]
    col = {"G_N": MUTED, "G_S": SERIES[1], "G_T": SERIES[0]}
    lab = {k: T[lang][k[-1]] for k in ("G_N", "G_S", "G_T")}
    for yi, r in zip(y, sel):
        vals = [r[k] for k in ("G_N", "G_S", "G_T") if r[k]]
        ax.plot([min(vals), max(vals)], [yi, yi], "-", color=INK2, lw=0.8, alpha=0.6, zorder=1)
        # время — крупный полый ромб снизу, шаги — квадрат поверх, интегрирования — кружок сверху: маркеры не прячут друг друга
        if r["G_T"]:
            ax.plot(r["G_T"], yi, "D", ms=7, mfc="white", mec=col["G_T"], mew=1.3, zorder=3, label=lab["G_T"] if yi == y[0] else None)
        if r["G_S"]:
            ax.plot(r["G_S"], yi, "s", ms=4.5, color=col["G_S"], mec="white", mew=0.5, zorder=4, label=lab["G_S"] if yi == y[0] else None)
        if r["G_N"]:
            ax.plot(r["G_N"], yi, "o", ms=4, color=col["G_N"], mec="white", mew=0.5, zorder=5, label=lab["G_N"] if yi == y[0] else None)
    ax.axvline(1.0, color=INK, lw=0.8, ls="--", zorder=0)
    # группы: сверху — системы, где большинство траекторий уходит на бесконечность, снизу — ограниченные
    nb = sum(1 for r in sel if r["system"].startswith(("chua_II", "coupled_chua")))
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([r[lang] for r in sel])
    ax.set_xlabel(T[lang]["x"])
    ax.set_ylim(-0.7, len(sel) - 0.3)
    ax.grid(axis="y", visible=False)
    if 0 < nb < len(sel):
        import matplotlib.transforms as mtr
        tr = mtr.blended_transform_factory(ax.transAxes, ax.transData)
        ax.axhspan(-0.5, nb - 0.5, color=SERIES[0], alpha=0.06, zorder=0, lw=0)
        for y0, y1, key in ((nb - 0.5, len(sel) - 0.5, "esc"), (-0.5, nb - 0.5, "bnd")):
            ax.plot([1.015, 1.015], [y0 + 0.15, y1 - 0.15], "-", color=INK2, lw=0.8, transform=tr, clip_on=False)
            ax.text(1.035, 0.5 * (y0 + y1), T[lang][key], transform=tr, rotation=90, va="center", ha="left", fontsize=fs, color=INK2)
    ax.legend(loc="lower left", handletextpad=0.3, borderaxespad=0.2, labelspacing=0.25)
    fig.tight_layout()
    return fig


def sci(x, lang, sig=2):
    """a×10^k: EN — $a\\times10^{k}$, RU — a·10ᵏ; sig — значащих цифр мантиссы."""
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    if round(m, sig - 1) >= 10:
        m /= 10
        e += 1
    ms = f"{m:.{sig - 1}f}"
    if lang == "en":
        return f"${ms}\\times10^{{{e}}}$"
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{ms}·10{str(e).translate(sup)}"


def fmt(x, lang, sig=2):
    """Счётчики: целое до 10^4, дальше a×10^k (никогда не 7.6e3)."""
    if x is None or not math.isfinite(x):
        return "---" if lang == "en" else "—"
    if x >= 1e4:
        return sci(x, lang, sig)
    return f"{x:.0f}" if x >= 100 else f"{x:.{sig}g}"


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
    """p: десятичная запись при p >= 1e-3 (0.033, 0.0026), иначе a×10^-k."""
    if p >= 1e-3:
        return f"{p:.2g}"
    return sci(p, lang, 2)


def gcell(r, key, lang):
    """«значение \\ci{lo}{hi}», жирным только при нижней границе выше 1."""
    g, (lo, hi) = r[key], r[key + "_ci"]
    s = gfmt(g, lang)
    if lang == "en":
        s = (f"\\textbf{{{s}}}" if lo > 1 else s) + f" \\ci{{{gfmt(lo, lang)}}}{{{gfmt(hi, lang)}}}"
    else:
        s = (f"<b>{s}</b>" if lo > 1 else s) + f" [{gfmt(lo, lang)}, {gfmt(hi, lang)}]"
    return s


def tables(sel):
    en, ru = [], []
    for r in sel:
        cells = [r["en"], str(r["n"]), pfmt(r["p"], "en"), fmt(r["S_u"], "en"), fmt(r["S_m"], "en"),
                 gcell(r, "G_N", "en"), gcell(r, "G_S", "en"), gcell(r, "G_T", "en")]
        en.append(" & ".join(cells) + " \\\\")
        cells = [r["ru"], str(r["n"]), pfmt(r["p"], "ru"), fmt(r["S_u"], "ru"), fmt(r["S_m"], "ru"),
                 gcell(r, "G_N", "ru"), gcell(r, "G_S", "ru"), gcell(r, "G_T", "ru")]
        ru.append("<tr>" + f"<td>{cells[0]}</td>" + "".join(f'<td class="num">{c}</td>' for c in cells[1:]) + "</tr>")
    sn = ROOT / "results" / "e53_snippets"
    sn.mkdir(exist_ok=True)
    (sn / "tab_cost.tex").write_text("\n".join(en) + "\n", encoding="utf-8")
    (sn / "tab_cost.html").write_text("\n".join(ru) + "\n", encoding="utf-8")


def changes_md(sel, old):
    old_by = {r["system"]: r for r in old}
    f = lambda v: "—" if v is None or not (isinstance(v, (int, float)) and math.isfinite(v)) else f"{v:.3g}"
    L = ["# e54: стоимость поиска с фиксированной точкой отсчёта и бутстрап-интервалами (2026-10-13)", "",
         "Что изменилось: (i) N0_0.9 перебора — фиксированное значение REF_N0 (источники ниже), а не квантиль по 20 запускам перебора "
         "того же батча; шаги и время перебора — N0 × средние шаги и время на оценку перебора того же батча; (ii) интервалы 95 % — "
         f"бутстрап по запускам портфеля ({N_BOOT} перевыборок; для измеренного N0 двух цепей Чуа II при eps = 0.1 — и по запускам "
         "перебора); жирным в таблице только при нижней границе > 1; (iii) формат чисел: p десятичное при p >= 1e-3, иначе a×10^-k; "
         "большие счётчики — a×10^k; (iv) рисунок fig30: шрифты >= 8 pt, маркеры G_S не прячутся за G_T, группы подписаны "
         "«escaping» / «bounded» вместо пунктира.", "",
         "Квантиль 0.9 — np.quantile по запускам; у трёх копий Чуа II (R = 10, 9 успехов) он бесконечен, и берётся порядковая "
         "статистика ceil(0.9 R) = 9-й запуск (как прежде); в бутстрапе перевыборки с двумя и более неудачами дают G = 0, поэтому "
         "нижняя граница этой строки равна 0.", "",
         "| задача | N0 старое (батч) | N0 новое | источник N0 | G_N старое → новое [95 %] | G_S старое → новое [95 %] | G_T старое → новое [95 %] |",
         "|---|---|---|---|---|---|---|"]
    for r in sel:
        o = old_by.get(r["system"], {})
        L.append(f"| {r['en']} | {f(o.get('N09u'))} ({o.get('batch_u', '—')}) | {f(r['N0_ref'])} | {r['N0_source']} | "
                 f"{f(o.get('G_N'))} → {f(r['G_N'])} [{f(r['G_N_ci'][0])}, {f(r['G_N_ci'][1])}] | "
                 f"{f(o.get('G_S'))} → {f(r['G_S'])} [{f(r['G_S_ci'][0])}, {f(r['G_S_ci'][1])}] | "
                 f"{f(o.get('G_T'))} → {f(r['G_T'])} [{f(r['G_T_ci'][0])}, {f(r['G_T_ci'][1])}] |")
    L += ["", "Числа портфеля (N_0.9, S_0.9, T_0.9 по запускам) не менялись; изменилась только точка отсчёта перебора.", ""]
    (ROOT / "results" / "e54_cost_final.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    cur = ROOT / "results" / "e54_cost_final.json"
    prev = ROOT / "results" / "e54_cost_final_prev.json"
    if cur.exists() and not prev.exists():
        shutil.copy(cur, prev)               # прежние значения (до фиксированной точки отсчёта) — для сравнения в md
    old = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else []
    sel = pick(rows())
    cur.write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in sel:
        print(f"{r['en']:18s} {r['label']:20s} {r['batch']}/{r['batch_u']} R={r['R']:3d} N09={r['N09']:.4g} N0={r['N0_ref']:.4g} "
              f"G_N={r['G_N']:.3g} [{r['G_N_ci'][0]:.3g}, {r['G_N_ci'][1]:.3g}] G_S={r['G_S']:.3g} [{r['G_S_ci'][0]:.3g}, {r['G_S_ci'][1]:.3g}] "
              f"G_T={r['G_T']:.3g} [{r['G_T_ci'][0]:.3g}, {r['G_T_ci'][1]:.3g}]")
    tables(sel)
    changes_md(sel, old)
    for lang, d in (("ru", ROOT / "results" / "figures"), ("en", ROOT / "paper" / "chaos" / "figures")):
        save(figure(sel, lang), d / "fig30_cost_gains")


if __name__ == "__main__":
    main()
