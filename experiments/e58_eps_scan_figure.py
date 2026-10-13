"""Рабочий рисунок по скану силы связи двух цепей Чуа II (e58): доли бассейнов и показатели Ляпунова режимов
в зависимости от eps, скрытость, равновесия. Пишет results/figures/e58_eps_scan.png и печатает таблицу.

    python experiments/e58_eps_scan_figure.py [_down _up ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hsearch.viz import INK2, MUTED, SERIES, save, setup  # noqa: E402

NEW = ("P_LC-HA_a", "P_HA_a-LC")                     # аттрактор «цикл + хаос» и его зеркальная копия
RARE = ("P_HA_a-E2", "P_E1-HA_a", "P_E2-HA_b", "HA_aa", "HA_ab", "HA_ba", "HA_bb")   # скрытый хаос хотя бы в одной цепи, другая не на LC


def load(tags):
    pts = []
    for t in tags:
        p = ROOT / "results" / f"e58_eps_scan{t}.json"
        if p.exists():
            pts += json.loads(p.read_text(encoding="utf-8"))["points"]
    pts.sort(key=lambda r: r["eps"])
    # одинаковые eps из двух сканов (стартовая точка) — берём первую
    out, seen = [], set()
    for r in pts:
        if r["eps"] not in seen:
            seen.add(r["eps"]); out.append(r)
    return out


def main():
    tags = sys.argv[1:] or ["_down", "_up"]
    pts = load(tags)
    if not pts:
        print("нет данных"); return
    setup()
    eps = np.array([r["eps"] for r in pts])
    n = pts[0]["n_sample"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
    # (а) доли бассейнов
    a = ax[0, 0]
    names = sorted({nm for r in pts for nm, g in r["regimes"].items() if g.get("status") in ("ok", "discovered")})
    col = {}
    for i, nm in enumerate(names):
        col[nm] = SERIES[i % len(SERIES)]
    for nm in names:
        y = np.array([r["regimes"].get(nm, {}).get("frac", np.nan) if r["regimes"].get(nm, {}).get("status") in ("ok", "discovered") else np.nan for r in pts])
        if np.nanmax(np.nan_to_num(y)) <= 0:
            continue
        a.plot(eps, np.where(y > 0, y, np.nan), "o-", ms=3, lw=1, color=col[nm], label=nm)
    a.axhline(1.0 / n, color=MUTED, lw=0.8, ls=":")
    a.set_yscale("log"); a.set_xlabel("ε"); a.set_ylabel("доля бассейна"); a.set_title("(а) доли бассейнов режимов", fontsize=10)
    a.legend(fontsize=6, ncol=2)
    # (б) показатель Ляпунова нового аттрактора и его природа
    a = ax[0, 1]
    for nm, c in zip(NEW, (SERIES[0], SERIES[1])):
        lam = np.array([r["regimes"].get(nm, {}).get("lam", np.nan) if r["regimes"].get(nm, {}).get("status") == "ok" else np.nan for r in pts], float)
        a.plot(eps, lam, "o-", ms=3, lw=1, color=c, label=nm)
    a.axhline(0, color=MUTED, lw=0.8, ls="--")
    a.set_xlabel("ε"); a.set_ylabel("старший показатель Ляпунова"); a.set_title("(б) аттрактор «цикл + хаос»", fontsize=10)
    a.legend(fontsize=7)
    # (в) неопознанные и скрытость нового аттрактора
    a = ax[1, 0]
    unk = np.array([r["fractions"]["unknown"] for r in pts])
    a.plot(eps, unk, "s-", ms=3, lw=1, color=INK2, label="неопознанные (доля)")
    for nm, c in zip(NEW, (SERIES[0], SERIES[1])):
        h = np.array([max(r["regimes"].get(nm, {}).get("h_eps", {"0.01": np.nan}).values()) if r["regimes"].get(nm, {}).get("status") == "ok" else np.nan for r in pts], float)
        a.plot(eps, h, "^-", ms=3, lw=1, color=c, label=f"h_eps({nm}) — из окрестностей равновесий")
    a.set_yscale("symlog", linthresh=1e-4); a.set_xlabel("ε"); a.set_title("(в) неопознанные траектории и скрытость", fontsize=10)
    a.legend(fontsize=6)
    # (г) что делает каждая цепь на новом аттракторе
    a = ax[1, 1]
    d2 = np.array([r["regimes"].get(NEW[0], {}).get("units", [{}, {}])[1].get("rel_dist", np.nan) if r["regimes"].get(NEW[0], {}).get("status") == "ok" else np.nan for r in pts], float)
    d1 = np.array([r["regimes"].get(NEW[0], {}).get("units", [{}, {}])[0].get("rel_dist", np.nan) if r["regimes"].get(NEW[0], {}).get("status") == "ok" else np.nan for r in pts], float)
    a.plot(eps, d1, "o-", ms=3, lw=1, color=SERIES[0], label="цепь 1: расстояние до режима отдельной цепи")
    a.plot(eps, d2, "o-", ms=3, lw=1, color=SERIES[1], label="цепь 2: то же")
    a.set_xlabel("ε"); a.set_ylabel("отн. расстояние"); a.set_title("(г) отличие цепей от режимов одиночной цепи", fontsize=10)
    a.legend(fontsize=6)
    fig.suptitle("Две цепи Чуа II, связь по x: скан силы связи", fontsize=11)
    fig.tight_layout()
    save(fig, ROOT / "results" / "figures" / "e58_eps_scan")
    # таблица
    print(f"{'eps':>6} {'unk':>8} " + " ".join(f"{nm[:12]:>12}" for nm in names))
    for r in pts:
        row = []
        for nm in names:
            g = r["regimes"].get(nm)
            if not g or g.get("status") not in ("ok", "discovered"):
                row.append(f"{g['status'][:4] if g else '-':>12}")
            else:
                row.append(f"{g['kind'][:2]} {g.get('lam', 0):+.3f} {g.get('frac', 0):.0e}"[:12].rjust(12))
        print(f"{r['eps']:6g} {r['fractions']['unknown']:8.1e} " + " ".join(row))


if __name__ == "__main__":
    main()
