"""Разбор «неизвестных» остатков выборки: что это — переходные процессы или новые аттракторы?

Для указанных систем: повторная выборка (n точек), сопоставление с эталонами из известных IC,
кластеризация неизвестных, для медоидов — продолжение интегрирования (верификация времени жизни)
и повторное сопоставление. Выход: results/e07_unknown_<system>.json
Запуск: python experiments/e07_unknown_residues.py wangchen2 le1 rabinovich_100 chua_II
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import (LAB_UNKNOWN, build_reference_set, calibrate_reference,  # noqa: E402
                              feature_slices, match, verify_candidate)
from hsearch.integrate import F_ESC, F_FP, F_LAM, F_RMAX, F_SPEED, run_batch  # noqa: E402


def main(names):
    for name in names:
        rng = np.random.default_rng(7)
        s = systems.get(name)
        n = 20000
        refs, ics, feats, finals, labels, hist = build_reference_set(s, rng, n_sample=n, max_rounds=1, verbose=False)
        unk = np.where(labels == LAB_UNKNOWN)[0]
        sl = feature_slices(s.dim)
        print(f"=== {name}: неизвестных {len(unk)} из {n}; эталоны {[(r.name, r.kind) for r in refs]}")
        out = {"system": name, "n": n, "n_unknown": int(len(unk)), "cases": []}
        if len(unk) == 0:
            continue
        # краткая статистика неизвестных
        lam = feats[unk, F_LAM]
        print(f"  λ: медиана {np.median(lam):+.4f}, доля λ<-0.002: {(lam < -0.002).mean():.2f}, "
              f"доля |λ|<0.02: {(np.abs(lam) < 0.02).mean():.2f}, доля λ>0.02: {(lam > 0.02).mean():.2f}")
        # кластеризация по признакам и верификация медоидов
        from sklearn.cluster import DBSCAN
        X = np.column_stack([feats[unk, F_LAM], feats[unk, F_RMAX], feats[unk, F_SPEED],
                             feats[unk][:, sl["mean"]], feats[unk][:, sl["std"]]])
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        cl = DBSCAN(eps=0.8, min_samples=3).fit_predict(Xs)
        for c in sorted(set(cl)):
            members = unk[cl == c]
            med = members[np.argmin(np.linalg.norm(Xs[cl == c] - Xs[cl == c].mean(0), axis=1))]
            f = feats[med]
            # продолжаем интегрирование из конечного состояния: 10 окон наблюдения
            x = finals[med].copy()
            track = []
            for w in range(10):
                f2, fin2 = run_batch(s, x[None, :], n_tr=0, n_obs=s.n_obs)
                lab2, z2 = match(s, f2, fin2, refs)
                track.append({"w": w, "lam": round(float(f2[0, F_LAM]), 4), "esc": bool(f2[0, F_ESC] > 0.5),
                              "fp": bool(f2[0, F_FP] > 0.5), "rmax": round(float(f2[0, F_RMAX]), 3),
                              "label": refs[lab2[0]].name if lab2[0] >= 0 else int(lab2[0])})
                x = fin2[0]
                if f2[0, F_ESC] > 0.5:
                    break
            case = {"cluster": int(c), "size": int(len(members)), "medoid_ic": ics[med].tolist(),
                    "lam0": round(float(f[F_LAM]), 4), "rmax0": round(float(f[F_RMAX]), 3),
                    "mean0": [round(float(v), 3) for v in f[sl["mean"]]], "track": track}
            out["cases"].append(case)
            print(f"  кластер {c}: размер {len(members)}, λ0={case['lam0']:+.4f}, rmax0={case['rmax0']}, "
                  f"mean={case['mean0']}; продолжение: " + " ".join(f"[{t['label']} λ={t['lam']:+.3f}]" for t in track))
        (ROOT / "results" / f"e07_unknown_{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["wangchen2", "le1", "rabinovich_100"])
