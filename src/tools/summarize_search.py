"""Сводка результатов бенчмарка стратегий (results/e20_search_*.json) в консоль и в results/e20_summary.md."""
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def fmt(v, nd=0):
    if v is None:
        return "—"
    return f"{v:.{nd}f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    lines = ["| Система | цели | стратегия | успех | медиана | N_0.9 [95% ДИ] | G_0.9 |", "|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(str(ROOT / "results" / "e20_search_*.json"))):
        if "_smoke" in p:
            continue
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        m = d["_meta"]
        print(f"{os.path.basename(p)}  ({time.ctime(os.path.getmtime(p))})  R={m['R']} budget={m['budget']} targets={m['targets']}")
        nu = d.get("uniform", {}).get("N_0.9")
        for k, v in d.items():
            if k == "_meta" or "found_at" not in v:
                continue
            g = (nu / v["N_0.9"]) if (nu and v.get("N_0.9")) else None
            ci = v.get("N_0.9_ci") or [None, None]
            print(f"   {k:16s} success {v['success_rate']:.2f}  median {fmt(v['median'])}  N09 {fmt(v['N_0.9'])} "
                  f"[{fmt(ci[0])}, {fmt(ci[1])}]  G {fmt(g, 2)}")
            lines.append(f"| {m['system']} | {', '.join(m['targets'])} | {k} | {v['success_rate']:.2f} | {fmt(v['median'])} | "
                         f"{fmt(v['N_0.9'])} [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(g, 2)} |")
    (ROOT / "results" / "e20_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
