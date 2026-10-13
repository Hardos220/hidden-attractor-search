"""Анализ «сигнала» для модели стоимости: насколько наблюдаемые свойства траекторий, не попавших
в цель, указывают на близость к бассейну цели (каналы A и B модели стоимости).

Для системы (эталоны строятся детерминированно, как в бенчмарке e20): выборка n точек в B,
метки и отпечатки. Затем:
1. Канал A (геометрия): для каждой целевой точки — метки её k = 10 ближайших соседей в
   нормированном пространстве начальных условий: доля целевых («толщина»), состав нецелевых
   («смежность» с уходом / равновесиями / известными аттракторами); пространственная
   корреляция меток для случайных точек; доля «граничных» точек (есть сосед другого класса) и
   доля целевых точек среди граничных (обогащение границы целью).
2. Канал B (переходный процесс): для нецелевых точек y = «ближайшая целевая точка ближе r»
   (r — 5 %-квантиль расстояний); признаки: длительность переходного процесса (координата
   расщепления), время ухода, раннее окно, асимптотические λ и rmax; AUC по каждому признаку
   (без учёта знака) и логистической регрессии по всем.
Выход: results/e22_signal_<system>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from e20_search_benchmark import build_problem  # noqa: E402
from hsearch.identify import LAB_ESCAPE, match  # noqa: E402
from hsearch.integrate import F_ESC, F_LAM, F_RMAX, F_TESC, feature_slices, run_batch  # noqa: E402
from hsearch.search.sus import transient_length  # noqa: E402


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    if y.sum() == 0 or y.sum() == len(y):
        return None
    a = float(roc_auc_score(y, score))
    return max(a, 1 - a)


def main(names, n=20000):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    for name in names:
        rng = np.random.default_rng(22)
        s, refs, tidx, kidx = build_problem(name, rng)
        ics = s.sample_box(n, rng)
        feats, finals = run_batch(s, ics)
        labels, z = match(s, feats, finals, refs)
        y_t = np.isin(labels, tidx)
        rec = {"system": name, "n": n, "targets": [refs[k].name for k in tidx], "p_target": float(y_t.mean())}
        if y_t.sum() < 5:
            print(f"{name}: целевых точек {y_t.sum()} — мало для анализа")
            rec["note"] = "слишком мало целевых точек"
            (ROOT / "results" / f"e22_signal_{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            continue
        cls = np.where(feats[:, F_ESC] > 0.5, LAB_ESCAPE, labels)
        X = (ics - s.box_lo) / (s.box_hi - s.box_lo)
        nn = NearestNeighbors(n_neighbors=11).fit(X)
        dist, idx = nn.kneighbors(X)
        neigh = cls[idx[:, 1:]]
        names_lab = {LAB_ESCAPE: "escape", -2: "unknown", **{k: r.name for k, r in enumerate(refs)}}
        nt = neigh[y_t]
        comp = {}
        for lab in np.unique(nt):
            comp[names_lab.get(int(lab), str(lab))] = float((nt == lab).mean())
        rec["target_neighbourhood"] = comp
        rec["thickness"] = float(np.isin(nt, tidx).mean())
        rec["label_spatial_correlation"] = float((neigh == cls[:, None]).mean())
        edge = (neigh != cls[:, None]).any(axis=1)
        rec["edge_fraction"] = float(edge.mean())
        rec["target_share_on_edge"] = float(y_t[edge].mean()) if edge.any() else None
        rec["edge_enrichment"] = (rec["target_share_on_edge"] / rec["p_target"]) if edge.any() else None
        # канал B
        ntm = ~y_t
        nn_t = NearestNeighbors(n_neighbors=1).fit(X[y_t])
        d_to_t, _ = nn_t.kneighbors(X[ntm])
        d_to_t = d_to_t[:, 0]
        r = np.quantile(d_to_t, 0.05)
        yy = (d_to_t <= r).astype(int)
        sl = feature_slices(s.dim)
        f = feats[ntm]
        n_total = float(s.n_tr + s.n_obs)
        cand = {
            "trans": transient_length(s, f),
            "t_esc": np.where(f[:, F_ESC] > 0.5, f[:, F_TESC] / n_total, 1.0),
            "emean": np.linalg.norm(f[:, sl["emean"]], axis=1),
            "estd": np.linalg.norm(f[:, sl["estd"]], axis=1),
            "lam": np.where(f[:, F_ESC] > 0.5, 0.0, f[:, F_LAM]),
            "rmax": np.where(f[:, F_ESC] > 0.5, 0.0, f[:, F_RMAX]),
        }
        aucs = {k: auc(v, yy) for k, v in cand.items()}
        # многомасштабная проверка канала B: для верхних множеств координаты расщепления
        # S_q = {xi >= (1-q)-квантиль} (по всем точкам, включая целевые) — доля цели в S_q
        # (обогащение E(q) = доля/p) и медиана расстояния нецелевых точек S_q до цели
        xi_all = transient_length(s, feats)
        curve = []
        d_all_med = float(np.median(d_to_t))
        for q in (1.0, 0.2, 0.04, 0.008, 0.0016):
            thr = np.quantile(xi_all, 1 - q) if q < 1 else -np.inf
            S = xi_all >= thr
            n_S = int(S.sum())
            share = float(y_t[S].mean()) if n_S else None
            Snt = S[ntm]
            med = float(np.median(d_to_t[Snt]) / d_all_med) if Snt.sum() else None
            curve.append({"q": q, "n": n_S, "target_share": share,
                          "enrichment": (share / rec["p_target"]) if share is not None else None,
                          "median_dist_rel": med})
        rec["level_curve"] = curve
        # канал A до первого обнаружения: границы между *известными* классами (без цели) —
        # проходят ли они рядом с целью? Доля нецелевых точек с соседом другого нецелевого класса,
        # лежащих ближе r к цели, отнесённая к базовой доле 0.05 (по определению r).
        nt_idx = np.where(ntm)[0]
        neigh_nt = neigh[nt_idx]
        own = cls[nt_idx]
        diff_known = (neigh_nt != own[:, None]) & ~np.isin(neigh_nt, tidx)
        kedge = diff_known.any(axis=1)
        rec["known_edge_fraction"] = float(kedge.mean())
        rec["known_edge_near_target_enrichment"] = (float(yy[kedge].mean()) / 0.05) if kedge.any() else None
        Z = np.nan_to_num(np.column_stack(list(cand.values())))
        Zs = StandardScaler().fit_transform(Z)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Zs, yy)
        rec["proximity_radius"] = float(r)
        rec["auc_single"] = aucs
        rec["auc_logistic_all"] = auc(lr.decision_function(Zs), yy)
        print(f"{name}: p={rec['p_target']:.4g} толщина={rec['thickness']:.2f} корр.меток={rec['label_spatial_correlation']:.3f} "
              f"граница={rec['edge_fraction']:.3f} обогащение границы={rec['edge_enrichment']:.1f} соседи={ {k: round(v, 2) for k, v in comp.items()} }; "
              f"AUC: " + ", ".join(f"{k}={v:.2f}" for k, v in aucs.items() if v) + f"; логит={rec['auc_logistic_all']:.2f}; "
              f"E(q): " + ", ".join(f"q={c['q']:g}: E={c['enrichment']:.1f} d={c['median_dist_rel']:.2f}" for c in curve
                                    if c['enrichment'] is not None and c['median_dist_rel'] is not None)
              + f"; известные границы: доля {rec['known_edge_fraction']:.3f}, близость к цели ×{rec['known_edge_near_target_enrichment']}", flush=True)
        (ROOT / "results" / f"e22_signal_{name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:] or ["chua_II", "se11", "le1", "huynh2019", "jafari2026_lc", "chua_I",
                          "edge_se3_m3_p0.001", "emb_se3_m3_p0.001"])
