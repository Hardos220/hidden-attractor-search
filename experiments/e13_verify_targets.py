"""Формальная верификация целевых аттракторов по протоколу (docs/07): для каждой системы и каждой
скрытой цели — время жизни (20 окон T_obs), повторное интегрирование с dt/2, сходимость λ по окнам.
Выход: results/e13_verify_targets.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import calibrate_reference, verify_candidate  # noqa: E402

SYSTEMS = ["chua_I", "chua_II", "ne4", "wangchen1", "wei_ne2", "wangchen2", "se3", "se11", "le1",
           "rabinovich_100", "jafari2026_lc", "jafari2026_chaos", "jiang_a", "jiang_b", "jiang_c", "huynh2019",
           "li_sprott_4d_a6", "jahanshahi2019_4d", "lorenz_transient", "rabinovich_68"]


def main(names=None):
    names = names or SYSTEMS
    out_path = ROOT / "results" / "e13_verify_targets.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    e02 = json.loads((ROOT / "results" / "e02_uniform.json").read_text(encoding="utf-8"))
    for name in names:
        rng = np.random.default_rng(13)
        s = systems.get(name)
        rec = {}
        # цели: известные скрытые + обнаруженные (из e02: example-состояния)
        targets = {k: s.known[k] for k in s.hidden if k in s.known}
        for r in e02.get(name, {}).get("refs", []):
            if r["name"].startswith("new") and r["kind"] in ("chaotic", "periodic") and r.get("example"):
                targets[r["name"]] = np.array(r["example"])
        if name == "lorenz_transient":
            targets = {"TC": s.known["TC"]}
        for tname, ic in targets.items():
            t0 = time.time()
            ref = calibrate_reference(s, tname, np.asarray(ic, float), rng)
            if ref.kind in ("escape", "fp"):
                rec[tname] = {"kind": ref.kind, "note": "цель не калибруется как осциллятор"}
                print(f"{name}/{tname}: {ref.kind}")
                continue
            rep = verify_candidate(s, ref.example, ref, rng, life_factor=20)
            rep.update({"kind": ref.kind, "lam": round(float(ref.lam), 4), "escape_frac_cal": ref.escape_frac,
                        "time_s": round(time.time() - t0, 1)})
            rec[tname] = rep
            print(f"{name}/{tname}: {ref.kind} λ={ref.lam:.4f} survived={rep['survived']} lifetime={rep['lifetime_T']:.0f} "
                  f"halfstep_ok={rep['halfstep_ok']} (z={rep['halfstep_z']:.2f}) λ_окна={[round(v, 3) for v in rep['lam_windows']]}")
        out[name] = rec
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
