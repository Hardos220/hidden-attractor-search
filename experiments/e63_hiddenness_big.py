"""Скрытость с большой выборкой (2026-10-13): верхние границы доли бассейна вместо «нулей».

Для двух связанных цепей Чуа II (eps = 0.1; система coupled_chua_II_k2_eps0.1_c0_prod_eq, как в e55) и для одиночных
систем Чуа II и SE11: N стартов на каждый радиус eps_h in (1e-3, 1e-2) L, распределённых по шарам вокруг всех
равновесий (по N / n_eq на равновесие) плюс точки e ± eps_h u вдоль неустойчивых собственных направлений якобиана —
та же конструкция, что в e11/e55. Каждый старт размечается двухэтапной оценкой (harness.screened_batch) по эталонам
задачи поиска (runner.get_problem: равновесия, произведения, HA-сочетания; целевые режимы никогда не останавливаются
на этапе 1 — их подтверждает полный протокол). Для каждого режима и радиуса: n, попадания, доля и верхняя граница
95 % (3/n при нуле попаданий; точная граница Клоппера–Пирсона при попаданиях).

Счёт идёт порциями (CHUNK стартов), каждая порция сразу дописывается в results/e63_hiddenness_big_parts.jsonl, поэтому
прерванная задача rmq (--preempt requeue, файл RMQ_STOP_FILE) продолжается с того же места. Точки порции порождаются
генератором с seed, зависящим от (система, радиус, равновесие, номер порции), и при повторе совпадают.

    rmq submit --cpu 4 --ram 4 --time 2h --preempt requeue --name e63 -- python experiments/e63_hiddenness_big.py
    python experiments/e63_hiddenness_big.py --n 100000 --systems coupled,chua_II,se11   (n — стартов на радиус)
пишет results/e63_hiddenness_big.json и results/e63_hiddenness_big.md
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import e50_batch_runner as runner  # noqa: E402
from hsearch.identify import LAB_ESCAPE, LAB_UNKNOWN, scale  # noqa: E402
from hsearch.search.harness import screened_batch  # noqa: E402

RADII = (1e-3, 1e-2)
PARTS = ROOT / "results" / "e63_hiddenness_big_parts.jsonl"
OUT_JSON = ROOT / "results" / "e63_hiddenness_big.json"
OUT_MD = ROOT / "results" / "e63_hiddenness_big.md"


def set_tag(tag):
    """Суффикс файлов (для проб: --tag _test), чтобы не смешивать порции с основным расчётом."""
    global PARTS, OUT_JSON, OUT_MD
    PARTS = ROOT / "results" / f"e63_hiddenness_big_parts{tag}.jsonl"
    OUT_JSON = ROOT / "results" / f"e63_hiddenness_big{tag}.json"
    OUT_MD = ROOT / "results" / f"e63_hiddenness_big{tag}.md"
VARIANTS = {
    "coupled": {"coupled": "chua_II", "k": 2, "eps": 0.1, "coord": 0, "ha": ["HA_a", "HA_b"], "products": True, "alleq": True},
    "chua_II": {"system": "chua_II"},
    "se11": {"system": "se11"},
}


def progress(done, total, note):
    pf = os.environ.get("RMQ_PROGRESS_FILE")
    if pf:
        try:
            Path(pf).write_text(f"{done}/{total} {note}", encoding="utf-8")
        except OSError:
            pass
    print(f"[{done}/{total}] {note}", flush=True)


def stop_requested():
    p = os.environ.get("RMQ_STOP_FILE")
    return bool(p) and os.path.exists(p)


def jacobian(s, e, h=1e-7):
    J = np.zeros((s.dim, s.dim)); f0 = np.zeros(s.dim); f1 = np.zeros(s.dim)
    s.rhs(e, s.params, f0)
    for i in range(s.dim):
        xp = e.copy(); xp[i] += h
        s.rhs(xp, s.params, f1)
        J[:, i] = (f1 - f0) / h
    return J


def unstable_dirs(s, e):
    w, V = np.linalg.eig(jacobian(s, e))
    out = []
    for k in range(len(w)):
        if np.real(w[k]) > 1e-6:
            u = np.real(V[:, k]); u /= np.linalg.norm(u) + 1e-12
            out.append(u)
    return out


def ball_points(e, e_h, n, rng, d):
    v = rng.normal(size=(n, d))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rr = e_h * rng.random(n) ** (1.0 / d)
    return e + v * rr[:, None]


def cp_upper(k, n, alpha=0.05):
    """Верхняя граница 95 % для доли по k попаданиям из n (Клоппер–Пирсон; при k = 0 — 1 - alpha^(1/n) ≈ 3/n)."""
    if n == 0:
        return None
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def read_parts():
    done = {}
    if PARTS.exists():
        for line in PARTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["key"]] = r
    return done


def append_part(rec):
    with open(PARTS, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def plan_chunks(sysname, n_eq, n_per_radius, chunk):
    """Список порций: (key, radius_idx, eq_idx, chunk_idx, n_points, kind)."""
    per_eq = int(math.ceil(n_per_radius / n_eq))
    out = []
    for ri in range(len(RADII)):
        for ei in range(n_eq):
            out.append((f"{sysname}|r{ri}|e{ei}|unst", ri, ei, -1, 0, "unst"))
            n_ch = int(math.ceil(per_eq / chunk))
            for ci in range(n_ch):
                n_pts = min(chunk, per_eq - ci * chunk)
                out.append((f"{sysname}|r{ri}|e{ei}|c{ci}", ri, ei, ci, n_pts, "ball"))
    return out


def targets_of(sysname, s, refs):
    if sysname == "coupled":
        return [k for k, r in enumerate(refs) if "HA" in r.name]
    return [k for k, r in enumerate(refs) if r.name in s.hidden]


def run_system(sysname, n_per_radius, chunk, done, total_chunks, n_done_before):
    var = VARIANTS[sysname]
    t0 = time.time()
    s, refs, tidx, kidx, _ = runner.get_problem(var, None)
    refs = list(refs)
    tidx = targets_of(sysname, s, refs)
    kidx = [k for k in range(len(refs)) if k not in tidx]
    L = scale(s)
    eqs = [np.asarray(e, float) for e in s.equilibria]
    udirs = [unstable_dirs(s, e) for e in eqs]
    print(f"{sysname}: {s.name}, эталоны {[r.name for r in refs]}, цели {[refs[k].name for k in tidx]}, "
          f"равновесий {len(eqs)}, неустойчивых направлений {[len(u) for u in udirs]}, построено за {time.time() - t0:.0f} с",
          flush=True)
    chunks = plan_chunks(sysname, len(eqs), n_per_radius, chunk)
    n_done = n_done_before
    for key, ri, ei, ci, n_pts, kind in chunks:
        if key in done:
            n_done += 1
            continue
        if stop_requested():
            print("остановка по RMQ_STOP_FILE: порции на диске, продолжение той же командой", flush=True)
            raise SystemExit(3)
        e_h = RADII[ri] * L
        e = eqs[ei]
        if kind == "unst":
            pts = [np.vstack([e + e_h * u, e - e_h * u]) for u in udirs[ei]]
            if not pts:
                append_part({"key": key, "system": sysname, "radius": RADII[ri], "eq": ei, "kind": kind, "n": 0, "counts": {},
                             "unknown": 0, "escape": 0})
                n_done += 1
                continue
            ics = np.vstack(pts)
        else:
            seed = int(np.random.SeedSequence([63, {"coupled": 1, "chua_II": 2, "se11": 3}[sysname], ri, ei, ci]).generate_state(1)[0])
            rng = np.random.default_rng(seed)
            ics = ball_points(e, e_h, n_pts, rng, s.dim)
        t1 = time.time()
        F, X, lab, steps, stop = screened_batch(s, ics, refs, kidx)
        counts = {}
        for k, r in enumerate(refs):
            c = int((lab == k).sum())
            if c:
                counts[r.name] = c
        hits = []
        for i in np.where(np.isin(lab, tidx))[0]:
            hits.append({"ic": [float(q) for q in ics[i]], "label": refs[lab[i]].name, "lam": float(F[i, 2])})
        rec = {"key": key, "system": sysname, "radius": RADII[ri], "eq": ei, "kind": kind, "n": int(len(ics)), "counts": counts,
               "unknown": int((lab == LAB_UNKNOWN).sum()), "escape": int((lab == LAB_ESCAPE).sum()),
               "stopped": int(stop.sum()), "steps": float(steps.sum()), "wall_s": round(time.time() - t1, 2)}
        if hits:
            rec["hits"] = hits[:50]
        append_part(rec)
        done[key] = rec
        n_done += 1
        progress(n_done, total_chunks, f"{sysname} r={RADII[ri]:g} e{ei} {kind}: {len(ics)} стартов, {rec['wall_s']} с, "
                                         f"попаданий в цели {len(hits)}")
    meta = {"system": s.name, "L": L, "dim": s.dim, "equilibria": [[float(q) for q in e] for e in eqs],
            "n_unstable_dirs": [len(u) for u in udirs], "refs": [(r.name, r.kind) for r in refs],
            "targets": [refs[k].name for k in tidx], "n_tr": int(s.n_tr), "n_obs": int(s.n_obs), "dt": float(s.dt)}
    return meta, n_done


def aggregate(metas, done):
    out = {"radii": list(RADII), "note": "n — стартов на радиус (шары вокруг всех равновесий + точки вдоль неустойчивых "
                                        "направлений); upper95 — верхняя граница 95 % доли бассейна: 1 - 0.05^(1/n) ≈ 3/n при "
                                        "нуле попаданий, Клоппер–Пирсон при попаданиях", "systems": {}}
    for sysname, meta in metas.items():
        recs = [r for r in done.values() if r["system"] == sysname]
        per_r = {}
        for ri, rad in enumerate(RADII):
            rr = [r for r in recs if abs(r["radius"] - rad) < 1e-15]
            n = sum(r["n"] for r in rr)
            n_unst = sum(r["n"] for r in rr if r["kind"] == "unst")
            counts = {}
            for r in rr:
                for k, c in r["counts"].items():
                    counts[k] = counts.get(k, 0) + c
            unk = sum(r["unknown"] for r in rr)
            esc = sum(r["escape"] for r in rr)
            hits = [h for r in rr for h in r.get("hits", [])]
            per_r[str(rad)] = {"n": n, "n_ball": n - n_unst, "n_unstable": n_unst, "n_chunks": len(rr), "counts": counts,
                               "unknown": unk, "escape": esc, "upper95_zero": (1.0 - 0.05 ** (1.0 / n)) if n else None,
                               "three_over_n": (3.0 / n) if n else None, "wall_s": round(sum(r.get("wall_s", 0) for r in rr), 1),
                               "stopped_frac": (sum(r.get("stopped", 0) for r in rr) / n) if n else None,
                               "steps_per_start": (sum(r.get("steps", 0) for r in rr) / n) if n else None,
                               "target_hits": hits[:100]}
        regimes = {}
        names = [nm for nm, _ in meta["refs"]]
        for nm in names:
            regimes[nm] = {}
            for rad in RADII:
                pr = per_r[str(rad)]
                k = pr["counts"].get(nm, 0)
                regimes[nm][str(rad)] = {"n": pr["n"], "hits": k, "frac": (k / pr["n"]) if pr["n"] else None,
                                         "upper95": cp_upper(k, pr["n"]), "is_target": nm in meta["targets"]}
        fam = {}
        for rad in RADII:
            pr = per_r[str(rad)]
            k = sum(pr["counts"].get(nm, 0) for nm in meta["targets"])
            fam[str(rad)] = {"n": pr["n"], "hits": k, "frac": (k / pr["n"]) if pr["n"] else None, "upper95": cp_upper(k, pr["n"])}
        out["systems"][sysname] = {**meta, "per_radius": per_r, "regimes": regimes, "targets_total": fam}
    return out


def write_md(out, n_per_radius):
    L = ["# e63: скрытость с большой выборкой — верхние границы вместо нулей", "",
         f"Стартов на радиус: {n_per_radius} (план), распределённых по шарам радиуса eps_h вокруг всех равновесий плюс точки "
         "e ± eps_h u вдоль неустойчивых направлений (как в e11/e55). Разметка — двухэтапная оценка (screened_batch) по эталонам "
         "задачи поиска; целевые режимы на этапе 1 не останавливаются. Верхняя граница 95 % при нуле попаданий: "
         "1 - 0.05^(1/n) ≈ 3/n; при попаданиях — Клоппер–Пирсон.", ""]
    for sysname, S in out["systems"].items():
        L.append(f"## {sysname}: {S['system']} (dim {S['dim']}, L = {S['L']:g}, равновесий {len(S['equilibria'])}, "
                 f"неустойчивых направлений {S['n_unstable_dirs']})")
        L.append("")
        L.append(f"Цели (скрытое семейство): {', '.join(S['targets'])}")
        L.append("")
        L.append("| радиус eps_h / L | n | шары | неуст. напр. | неопознано | уход | остановлено на этапе 1 | шагов/старт | 3/n |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for rad, pr in S["per_radius"].items():
            L.append(f"| {rad} | {pr['n']} | {pr['n_ball']} | {pr['n_unstable']} | {pr['unknown']} | {pr['escape']} | "
                     f"{100 * (pr['stopped_frac'] or 0):.1f} % | {(pr['steps_per_start'] or 0):.3g} | "
                     f"{(pr['three_over_n'] or 0):.2e} |")
        L.append("")
        L.append("| режим | цель | " + " | ".join(f"{rad}: попад. / n | доля | верх. 95 %" for rad in S["per_radius"]) + " |")
        L.append("|---|---|" + "---|---|---|" * len(S["per_radius"]))
        for nm, rr in S["regimes"].items():
            cells = []
            for rad in S["per_radius"]:
                q = rr[rad]
                cells.append(f"{q['hits']} / {q['n']} | {q['frac']:.2e} | {q['upper95']:.2e}" if q["n"] else "— | — | —")
            L.append(f"| {nm} | {'да' if rr[list(S['per_radius'])[0]]['is_target'] else ''} | " + " | ".join(cells) + " |")
        L.append("")
        fam = S["targets_total"]
        L.append("Все цели вместе: " + "; ".join(f"eps_h = {rad} L: {q['hits']} / {q['n']}, верхняя граница 95 % {q['upper95']:.2e}"
                                                 for rad, q in fam.items()))
        L.append("")
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000, help="стартов на радиус (на все равновесия вместе)")
    ap.add_argument("--chunk", type=int, default=4000)
    ap.add_argument("--systems", default="coupled,chua_II,se11")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.tag:
        set_tag(args.tag)
    import numba
    th = int(os.environ.get("RMQ_CPU", "4") or 4)
    numba.set_num_threads(max(1, min(th, numba.config.NUMBA_NUM_THREADS)))
    systems = [q for q in args.systems.split(",") if q]
    done = read_parts()
    metas = {}
    prev_json = json.loads(OUT_JSON.read_text(encoding="utf-8")) if OUT_JSON.exists() else {}
    if args.report_only:
        for sysname in systems:
            if sysname in prev_json.get("systems", {}):
                m = prev_json["systems"][sysname]
                metas[sysname] = {k: m[k] for k in ("system", "L", "dim", "equilibria", "n_unstable_dirs", "refs", "targets", "n_tr", "n_obs", "dt")}
        out = aggregate(metas, done)
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        write_md(out, args.n)
        return
    # общее число порций для прогресса
    n_eq = {"coupled": 9, "chua_II": 3, "se11": 1}
    total = sum(len(plan_chunks(q, n_eq[q], args.n, args.chunk)) for q in systems)
    n_done = 0
    t0 = time.time()
    for sysname in systems:
        meta, n_done = run_system(sysname, args.n, args.chunk, done, total, n_done)
        metas[sysname] = meta
        out = aggregate(metas, done)
        for k, v in prev_json.get("systems", {}).items():
            out["systems"].setdefault(k, v)
        out["n_per_radius"] = args.n
        out["wall_s"] = round(time.time() - t0, 1)
        OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        write_md(out, args.n)
        S = out["systems"][sysname]
        print(f"=== {sysname}: " + "; ".join(f"eps_h={rad}: цели {q['hits']}/{q['n']} (верх. 95% {q['upper95']:.1e})"
                                              for rad, q in S["targets_total"].items()), flush=True)
    progress(total, total, "готово")


if __name__ == "__main__":
    main()
