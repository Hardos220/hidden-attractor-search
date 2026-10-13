"""Скан параметра a3 системы Jafari et al. (2026) у границы ограниченности:
обратная бифуркация от a3 = -3.50 к -3.56 с шагом 2e-4, продолжение по конечному состоянию.
Для каждого a3: показатель Ляпунова λ (окно T = 400), флаг ухода, размах. Ищем окно λ > 0.
Выход: results/e12_jafari2026_scan.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.integrate import F_ESC, F_LAM, F_RMAX, nfeat  # noqa: E402


def main():
    s = systems.get("jafari2026_chaos")
    x = np.array([5.7, 1.3, 1.9])
    # сначала выводим на цикл при a3 = -1, затем медленно уменьшаем a3
    dt, n_tr, n_obs = 0.005, 60000, 80000
    rows = []
    a3_values = np.concatenate([np.arange(-1.0, -3.50, -0.005), np.arange(-3.50, -3.5602, -2e-4)])
    for a3 in a3_values:
        p = np.array([6.0, -1.0, a3, -1.0, -1.0])
        feats = np.zeros((1, nfeat(3))); fin = np.zeros((1, 3))
        s.batch(x[None, :], p, dt, n_tr, n_obs, s.esc_r, 20, feats, fin)
        esc = bool(feats[0, F_ESC] > 0.5)
        rows.append({"a3": round(float(a3), 5), "lam": round(float(feats[0, F_LAM]), 5), "esc": esc,
                     "rmax": round(float(feats[0, F_RMAX]), 3), "state": [float(v) for v in fin[0]]})
        if a3 <= -3.5:
            print(f"a3={a3:.4f}: λ={feats[0, F_LAM]:+.4f} esc={esc} rmax={feats[0, F_RMAX]:.2f}")
        if esc:
            print("уход на бесконечность при a3 =", a3)
            break
        x = fin[0]
    chaotic = [r for r in rows if (not r["esc"]) and r["lam"] > 0.01]
    best = max(chaotic, key=lambda r: r["lam"]) if chaotic else None
    out = {"rows": rows, "chaotic_window": [r["a3"] for r in chaotic], "best": best}
    (ROOT / "results" / "e12_jafari2026_scan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("хаотическое окно a3:", [r["a3"] for r in chaotic][:5], "...", "лучший:", best)


if __name__ == "__main__":
    main()
