"""Таблица параметров интегрирования эталонных систем для приложения статьи (из реестра систем).
Выход: paper/src/86_systems_table.html
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.viz import NICE_SYS  # noqa: E402

ORDER = ["chua_I", "chua_II", "ne4", "wangchen1", "wei_ne2", "jafari2026_lc", "jafari2026_chaos", "wangchen2", "se3", "se11",
         "le1", "rabinovich_100", "rabinovich_68", "lorenz_transient", "huynh2019", "jiang_nfe", "jiang_nff", "jiang_nfg",
         "jiang_nfi", "jiang_sfa", "jiang_sfb", "jiang_sfc", "jiang_sfd", "jiang_a", "jiang_b", "jiang_c", "henon",
         "li_sprott_4d_a6", "li_sprott_4d_a7", "li_sprott_4d_hyper", "jahanshahi2019_4d"]
EXTRA = {"jiang_nfe": "Цзян NF I e", "jiang_nff": "Цзян NF I f", "jiang_nfg": "Цзян NF I g", "jiang_nfi": "Цзян NF I i",
         "jiang_sfa": "Цзян SF I a", "jiang_sfb": "Цзян SF I b", "jiang_sfc": "Цзян SF I c", "jiang_sfd": "Цзян SF I d",
         "henon": "Хенон (контроль)", "jafari2026_lc": "Jafari 2026, \\(a_3=-1\\)", "jafari2026_chaos": "Jafari 2026, \\(a_3=-3.548\\)"}


def fmt_box(s):
    lo, hi = np.asarray(s.box_lo), np.asarray(s.box_hi)
    parts = []
    i = 0
    while i < len(lo):
        j = i
        while j + 1 < len(lo) and lo[j + 1] == lo[i] and hi[j + 1] == hi[i]:
            j += 1
        seg = f"[{lo[i]:g}, {hi[i]:g}]".replace("-", "−")
        parts.append(seg + (f"<sup>{j - i + 1}</sup>" if j > i else ""))
        i = j + 1
    return "×".join(parts)


def main():
    rows = []
    for name in ORDER:
        try:
            s = systems.get(name)
        except Exception:  # noqa: BLE001
            continue
        title = EXTRA.get(name, NICE_SYS.get(name, name))
        if s.kind == "flow":
            integ = f"RK4, \\(\\Delta t={s.dt:g}\\)"
            ttr, tob = s.n_tr * s.dt, s.n_obs * s.dt
            tt = f"{ttr:g} / {tob:g}"
        else:
            integ = "итерации"
            tt = f"{s.n_tr} / {s.n_obs}"
        rows.append(f"<tr><td>{title}</td><td class=\"num\">{s.dim}</td><td>{integ}</td><td class=\"num\">{tt}</td>"
                    f"<td>{fmt_box(s)}</td><td class=\"num\">{s.esc_r:g}</td></tr>")
    html = ("<table data-tab=\"integration\">\n<caption>Параметры интегрирования и области начальных условий \\(B\\). "
            "Для потоков указаны длины переходного участка и окна наблюдения в единицах времени, для отображений — "
            "в итерациях; \\(R_{esc}\\) — порог ухода</caption>\n"
            "<thead><tr><th>Система</th><th class=\"num\">\\(n\\)</th><th>интегрирование</th>"
            "<th class=\"num\">\\(T_{tr}/T_{obs}\\)</th><th>область \\(B\\)</th><th class=\"num\">\\(R_{esc}\\)</th></tr></thead>\n<tbody>\n"
            + "\n".join(rows) + "\n</tbody>\n</table>\n")
    out = ROOT / "paper" / "src" / "86_systems_table.html"
    out.write_text(html, encoding="utf-8")
    print("ok:", out, len(rows), "строк")


if __name__ == "__main__":
    main()
