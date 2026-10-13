"""Бенчмарк стратегий поиска: для системы и списка стратегий — R запусков с бюджетом B,
момент первого попадания в целевой аттрактор, P(N), N_0.9 с бутстрап-интервалом.

Запуск: python experiments/e20_search_benchmark.py chua_I [--strategies uniform,ams,...] [--R 20] [--budget 2000]
Выход: results/e20_search_<system>.json (дозаписывается по стратегиям)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hsearch import systems  # noqa: E402
from hsearch.identify import build_reference_set  # noqa: E402
from hsearch.search.harness import Problem, n_delta_from_runs, pn_curve, run_search  # noqa: E402
from hsearch.search.strategies import STRATEGIES  # noqa: E402

DEFAULT_TARGETS = {
    "chua_I": ["HA_a", "HA_b"], "chua_II": ["HA_a", "HA_b"], "ne4": ["HA"], "wangchen1": ["HA"], "wei_ne2": ["HA"],
    "wangchen2": ["HA"], "se3": ["HA"], "se11": ["HA"], "le1": ["HA"], "rabinovich_100": ["HA"],
    "jafari2026_lc": ["LC"], "jafari2026_chaos": ["HA"], "jiang_a": ["HA"], "jiang_b": ["HA"], "jiang_c": ["HA"],
    "huynh2019": ["HA"], "li_sprott_4d_a6": ["T"], "jahanshahi2019_4d": ["HA"],
}


def get_system(name):
    """emb_<base>_m<m>_p<p> — семейство D1 («ворота»); edge_<base>_m<m>_p<p> — семейство D2
    («краевое состояние»); доля бассейна p задаётся через радиус rho."""
    if name.startswith("emb_") or name.startswith("edge_"):
        from hsearch.systems.synthetic import make_edge_embedded, make_embedded, rho_for_p
        try:
            return systems.get(name)
        except KeyError:
            pass
        parts = name.split("_")
        base = parts[1]
        m = int(parts[2][1:])
        p_target = float(parts[3][1:])
        e02 = json.loads((ROOT / "results" / "e02_uniform.json").read_text(encoding="utf-8"))
        p_base = e02[base]["hidden"]["HA"]["p"]
        rho = rho_for_p(p_target, p_base, m)
        if name.startswith("emb_"):
            return make_embedded(base, m, rho, name=name)
        return make_edge_embedded(base, m, rho, name=name)
    return systems.get(name)


def build_problem(name, rng, n_sample=5000, targets=None):
    """Эталоны: равновесия + калибровка по всем известным состояниям системы (детерминированно),
    затем обнаружение прочих режимов по выборке (они считаются известными). Цели — по именам."""
    from hsearch.identify import _duplicate_of, calibrate_reference, discover, fp_references_from_equilibria, match
    from hsearch.integrate import run_batch
    s = get_system(name)
    tnames = targets or (["HA"] if (name.startswith("emb_") or name.startswith("edge_")) else None) \
        or DEFAULT_TARGETS.get(name) or list(s.hidden)
    refs = fp_references_from_equilibria(s)
    for nm, ic in s.known.items():
        r = calibrate_reference(s, nm, ic, rng)
        if r.kind == "escape":
            continue
        if r.kind == "fp" and _duplicate_of(s, r, refs) is not None:
            continue          # то же равновесие уже есть в списке
        refs.append(r)
    ics = s.sample_box(n_sample, rng)
    feats, finals = run_batch(s, ics)
    for _ in range(3):
        labels, z = match(s, feats, finals, refs)
        unk = labels == -2
        if unk.sum() < 3:
            break
        new = [r for r in discover(s, feats, finals, unk, rng) if _duplicate_of(s, r, refs) is None]
        if not new:
            break
        for r in new:
            r.name = f"other{sum(1 for q in refs if q.name.startswith('other'))}"
            refs.append(r)
    target_idx = [k for k, r in enumerate(refs) if r.name in tnames]
    if not target_idx:
        raise RuntimeError(f"цели {tnames} не найдены среди эталонов {[r.name for r in refs]}")
    known_idx = [k for k, r in enumerate(refs) if k not in target_idx]
    return s, refs, target_idx, known_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("system")
    ap.add_argument("--strategies", default="uniform,ams,novelty_me,active_boundary")
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--diam", default=None, help="диапазон диаметра для de_zelinka, например 3,11")
    ap.add_argument("--n_init", type=int, default=50, help="стартовая популяция (AMS: n_pop, MAP-Elites/active: n_init)")
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()
    out_path = ROOT / "results" / f"e20_search_{args.system}{args.tag}.json"
    out = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    rng0 = np.random.default_rng(100)
    s, refs, target_idx, known_idx = build_problem(args.system, rng0, targets=args.targets.split(",") if args.targets else None)
    print(f"{args.system}: эталоны {[(r.name, r.kind) for r in refs]}, цели {[refs[k].name for k in target_idx]}")
    import subprocess
    try:
        commit = subprocess.run(["C:/Program Files/Git/cmd/git.exe", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = None
    meta_extra = {}
    if getattr(s, "meta", None):
        meta_extra = dict(s.meta)
    out["_meta"] = {"system": args.system, "targets": [refs[k].name for k in target_idx], "budget": args.budget, "R": args.R,
                    "refs": [(r.name, r.kind) for r in refs], "n_init": args.n_init, "batch": args.batch,
                    "commit": commit, "system_meta": meta_extra, "dim": s.dim}
    for sname in args.strategies.split(","):
        base_name = sname[:-4] if sname.endswith("_cvs") else sname
        cls = STRATEGIES[base_name]
        found = []
        t0 = time.time()
        extra = []
        pool = None
        if sname.endswith("_cvs"):
            # априорное множество: точки на нульклинах (поверхностях критической скорости), общий пул
            from hsearch.baselines.geometric import nullcline_samples
            prng = np.random.default_rng(777)
            per = 3000 // s.dim
            pool = [nullcline_samples(s, prng, i, n=per) for i in range(s.dim)]
            pool = np.vstack([q for q in pool if len(q)]) if any(len(q) for q in pool) else None
        for r in range(args.R):
            rng = np.random.default_rng(1000 + r)
            prob = Problem(s, refs, target_idx, known_idx, rng)
            kw = {"batch": args.batch, "n_init": args.n_init, "n_pop": args.n_init}
            if pool is not None:
                sampler = (lambda n, rg, P=pool: P[rg.integers(0, len(P), size=n)])
                if base_name.startswith("portfolio"):
                    kw["uniform_sampler"] = sampler
                    kw["ab_kw"] = {"cand_sampler": sampler}
                    kw["seed_ics"] = np.vstack([sampler(args.n_init // 2, rng), s.sample_box(args.n_init - args.n_init // 2, rng)])
                else:
                    kw["seed_ics"] = sampler(args.n_init, rng)
            if sname == "de_zelinka":
                kw["n_pop"] = max(10, args.n_init // 2)
                if args.diam:
                    lo, hi = (float(v) for v in args.diam.split(","))
                else:
                    # оракул: диапазон по размеру цели (0.5..1.5 от rmax эталона)
                    rm = max(float(refs[k].mu[4]) for k in target_idx)
                    lo, hi = 0.5 * rm, 1.5 * rm
                kw["diam"] = (lo, hi)
            strat = cls(prob, **kw)
            res = run_search(prob, strat, args.budget)
            found.append(res["found_at"])
            extra.append({"found_at": res["found_at"], "n_discoveries": len(res["discoveries"])})
        q, lo, hi = n_delta_from_runs(found)
        grid, pn = pn_curve(found, args.budget)
        rec = {"found_at": found, "N_0.9": None if not np.isfinite(q) else q, "N_0.9_ci": [lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None],
               "success_rate": float(np.mean([f is not None for f in found])),
               "median": float(np.median([f if f is not None else np.inf for f in found])) if any(f is not None for f in found) else None,
               "pn_grid": grid.tolist(), "pn": pn.tolist(), "time_s": round(time.time() - t0, 1)}
        out[sname] = rec
        print(f"  {sname:16s}: успех {rec['success_rate']:.2f}, медиана {rec['median']}, N_0.9 = {rec['N_0.9']} "
              f"[{rec['N_0.9_ci'][0]}, {rec['N_0.9_ci'][1]}]  ({rec['time_s']} с)")
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
