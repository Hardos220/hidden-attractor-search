"""Запуск батчей e50 (план Р1–Р6) для длительных расчётов на этом ПК — только через очередь rmq.

На ПК работает менеджер ресурсов rmq (C:/Cursor/resources_manager, правила в ~/.claude/CLAUDE.md и `rmq guide`):
тяжёлые расчёты ставятся в его очередь, а не запускаются напрямую. Поэтому команда run без rmq отказывается
считать; ставить батчи надо командой submit, которая вызывает `rmq submit` с заявкой --cpu/--ram/--time.

    python experiments/e50_batch_runner.py list                    # батчи, оценка часов
    python experiments/e50_batch_runner.py submit B0 --dry-run     # какая команда rmq будет поставлена
    python experiments/e50_batch_runner.py submit B0 --cpu 4       # поставить в очередь (печатает номер задачи)
    python experiments/e50_batch_runner.py submit all --chain      # все обязательные батчи одной задачей, по очереди
    python experiments/e50_batch_runner.py status [B1a]            # прогресс и оценка остатка
    python experiments/e50_batch_runner.py summarize B1a           # сводки по уже посчитанному
    (run <батчи> вызывает сама задача rmq; вручную — только с --dry-run)

Заявка по умолчанию --cpu 20: по руководству rmq при свободном ПК заявляют все потоки, а процессор не
захватывается, а делится между агентами. Поэтому батчи ставятся одной задачей (--chain): отдельные задачи с
такой заявкой шли бы одновременно, делили бы те же потоки и выходили бы за заявленное время.

Устройство:
* единица работы — один запуск (задание × номер запуска r, генератор default_rng(1000 + r));
* результат каждого запуска сразу дописывается строкой в results/e50/<батч>/<задание>.jsonl,
  поэтому прерванный расчёт (перезапуск задачи rmq, перезагрузка) продолжается с того же места повторной командой
  run; запуск, упавший с ошибкой, не сохраняется и повторяется при следующем run, а сбой батча цепочки
  не останавливает следующие батчи;
* внутри задачи rmq запуски идут в RMQ_CPU // threads параллельных процессах с threads потоками numba
  (стратегии предлагают пачки по 10–30 точек, и одна пачка не занимает все потоки); приоритет, заморозку
  и перезапуск задачи (--preempt requeue) ведёт менеджер, запускатель их не трогает;
* по завершении задания пишется сводка results/e50/<батч>/summary/<вариант>.json в формате
  e20_search_*.json (found_at, N_0.9 с бутстрап-интервалом, P(N)), а для повторов прежних
  конфигураций — сверка первых запусков со старыми результатами.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import e50_batches as plan  # noqa: E402

OUT_ROOT = ROOT / "results" / "e50"

# ============================================================================ рабочий процесс
_CACHE = {}
_WARM = set()


def worker_init(threads):
    import numba
    numba.set_num_threads(max(1, min(int(threads), numba.config.NUMBA_NUM_THREADS)))


def build_variant(var):
    """Система варианта задания, зарегистрированная в реестре под именем plan.vname(var)."""
    from hsearch import systems
    name = plan.vname(var)
    try:
        return systems.get(name)
    except KeyError:
        pass
    if "coupled" in var:
        from hsearch.systems.coupled import make_coupled
        return make_coupled(var["coupled"], k=var["k"], eps=var.get("eps", 0.0), coord=var.get("coord", 0), name=name,
                            ha_names=var.get("ha"), products=bool(var.get("products")),
                            all_equilibria=bool(var.get("alleq")))
    import e20_search_benchmark as e20
    base = e20.get_system(var["system"])
    if not var.get("box") and not var.get("horizon") and not var.get("dt") and not var.get("par") and not var.get("census"):
        return base
    meta = dict(getattr(base, "meta", {}) or {})
    kw = {"name": name}
    if var.get("census"):
        # известные состояния — из переписи (results/<census>.json, запись системы): состояния-примеры аттракторов,
        # скрытые — по мере скрытости; имена целей задаются в задании (2026-10-13, осциллятор ван дер Поля–Дуффинга)
        cen = json.loads((ROOT / "results" / f"{var['census']}.json").read_text(encoding="utf-8"))[var["system"]]
        known = {g["name"]: np.asarray(g["example"], float) for g in cen["regimes"] if g.get("example") is not None}
        hidden = [g["name"] for g in cen["regimes"] if g.get("hidden") and g.get("example") is not None]
        kw.update(known=known, hidden=hidden)
        meta.update(census=var["census"])
    if var.get("par"):
        # параметр [индекс, значение] (2026-10-13); состояния известных аттракторов — из скана продолжением
        # (results/<known_from>.json, точка с этим значением параметра), равновесия — методом Ньютона
        from hsearch.systems.coupled import _newton_all
        i_p, v = int(var["par"][0]), float(var["par"][1])
        pars = base.params.copy()
        pars[i_p] = v
        scan = json.loads((ROOT / "results" / f"{var['known_from']}.json").read_text(encoding="utf-8"))
        pt = min(scan["points"], key=lambda q: abs(q["value"] - v))
        assert abs(pt["value"] - v) < 1e-9 and pt.get("HA") == "ok", (var, pt["value"])
        known = {nm: np.asarray(st, float) for nm, st in pt["states"].items()}
        eqs = _newton_all(base.rhs, pars, [np.asarray(e, float) for e in base.equilibria], base.dim) if base.equilibria else []
        kw.update(params=pars, known=known, equilibria=eqs)
        meta.update(param_index=i_p, param_value=v)
    if var.get("box"):
        k = float(var["box"])
        c = 0.5 * (base.box_lo + base.box_hi)
        lo, hi = c + k * (base.box_lo - c), c + k * (base.box_hi - c)
        corner = float(np.linalg.norm(np.maximum(np.abs(lo), np.abs(hi))))
        kw.update(box_lo=lo, box_hi=hi, esc_r=max(base.esc_r, 2.0 * corner))
        meta.update(box_scale=k)
    if var.get("horizon"):
        h = float(var["horizon"])
        kw.update(n_tr=int(round(base.n_tr * h)), n_obs=int(round(base.n_obs * h)))
        meta.update(horizon_scale=h)
    if var.get("dt"):
        # шаг интегрирования × f при тех же длительностях окон (T_tr, T_obs) и периоде перенормировки
        f = float(var["dt"])
        kw.update(dt=base.dt * f, n_tr=int(round(kw.get("n_tr", base.n_tr) / f)), n_obs=int(round(kw.get("n_obs", base.n_obs) / f)),
                  renorm_every=max(1, int(round(base.renorm_every / f))))
        meta.update(dt_scale=f)
    s = replace(base, **kw)
    s.meta = meta
    systems.register(s)
    return s


def _targets_for(var, targets):
    if targets:
        return list(targets)
    if "coupled" in var:
        return list(build_variant(var).hidden)          # ["HA"] для прежних вариантов; сочетания для var["ha"]
    if var.get("census"):
        return list(build_variant(var).hidden)
    import e20_search_benchmark as e20
    base = var["system"]
    if base.startswith("emb_") or base.startswith("edge_"):
        return ["HA"]
    return e20.DEFAULT_TARGETS.get(base)


def _drop_product_dups(s, refs, tidx):
    """Эталоны-произведения (P_*), калибровка которых пришла на целевой режим или повторяет другой эталон, —
    отбрасываются (при связи блоков смешанное состояние может не существовать)."""
    from hsearch.identify import match
    keep = []
    for k, r in enumerate(refs):
        if r.name.startswith("P_") and r.kind in ("periodic", "chaotic") and r.mu is not None:
            others = [q for i, q in enumerate(refs) if i != k and (not q.name.startswith("P_") or i < k)]
            f = np.asarray(r.mu, float)[None, :]
            lab, z = match(s, f, np.zeros((1, s.dim)), others)
            if lab[0] >= 0:
                continue
        keep.append(k)
    tnames = {refs[k].name for k in tidx}
    refs = [refs[k] for k in keep]
    tidx = [k for k, r in enumerate(refs) if r.name in tnames]
    kidx = [k for k in range(len(refs)) if k not in tidx]
    return refs, tidx, kidx


def get_problem(var, targets):
    key = (json.dumps(var, sort_keys=True), tuple(targets or ()))
    if key not in _CACHE:
        import e20_search_benchmark as e20
        s = build_variant(var)
        tn = _targets_for(var, targets)
        s2, refs, tidx, kidx = e20.build_problem(s.name, np.random.default_rng(100), targets=tn)
        if var.get("products"):
            refs, tidx, kidx = _drop_product_dups(s2, refs, tidx)
        _CACHE[key] = (s2, refs, tidx, kidx, {})
    return _CACHE[key]


def write_meta(meta_dir, var, s, refs, tidx):
    """Описание варианта (эталоны, цели, область) — один раз на вариант; читает write_summaries."""
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    p = meta_dir / f"{plan.vname(var)}.json"
    if p.exists():
        return
    meta = {"system": plan.vname(var), "variant": var, "targets": [refs[k].name for k in tidx],
            "refs": [(r.name, r.kind) for r in refs], "dim": s.dim, "system_meta": dict(getattr(s, "meta", {}) or {}),
            "box_lo": s.box_lo.tolist(), "box_hi": s.box_hi.tolist(), "esc_r": float(s.esc_r),
            "n_tr": int(s.n_tr), "n_obs": int(s.n_obs), "dt": float(s.dt)}
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        os.replace(tmp, p)
    except OSError:
        tmp.unlink(missing_ok=True)


def run_one(task):
    """Один запуск стратегии; повторяет логику e20_search_benchmark.main для запуска r."""
    j, r, meta_dir = task
    from hsearch.search.harness import Problem, run_search
    from hsearch.search.strategies import STRATEGIES
    s, refs, tidx, kidx, pools = get_problem(j["var"], j.get("targets"))
    write_meta(meta_dir, j["var"], s, refs, tidx)
    sname = j["strategy"]
    base_name = sname[:-4] if sname.endswith("_cvs") else sname
    cls = STRATEGIES[base_name]
    pool = None
    if sname.endswith("_cvs"):
        if "cvs" not in pools:
            from hsearch.baselines.geometric import nullcline_samples
            prng = np.random.default_rng(777)
            per = 3000 // s.dim
            pp = [nullcline_samples(s, prng, i, n=per) for i in range(s.dim)]
            pools["cvs"] = np.vstack([q for q in pp if len(q)]) if any(len(q) for q in pp) else None
        pool = pools["cvs"]
    t0 = time.time()
    rng = np.random.default_rng(1000 + r)
    prob = Problem(s, refs, tidx, kidx, rng, screen=j.get("screen"))
    n_init = int(j.get("n_init", 50))
    kw = {"batch": int(j.get("batch", 20)), "n_init": n_init, "n_pop": n_init}
    if pool is not None:
        sampler = (lambda n, rg, P=pool: P[rg.integers(0, len(P), size=n)])
        if base_name.startswith("portfolio"):
            kw["uniform_sampler"] = sampler
            kw["ab_kw"] = {"cand_sampler": sampler}
            kw["seed_ics"] = np.vstack([sampler(n_init // 2, rng), s.sample_box(n_init - n_init // 2, rng)])
        else:
            kw["seed_ics"] = sampler(n_init, rng)
    if sname == "de_zelinka":
        kw["n_pop"] = max(10, n_init // 2)
        rm = max(float(refs[k].mu[4]) for k in tidx)
        kw["diam"] = (0.5 * rm, 1.5 * rm)
    if j.get("warmup") and j["id"] not in _WARM:
        # прогрев (не входит во время запуска): импорт sklearn, компиляция ядер и путей стратегии — один раз на
        # процесс и задание; свой генератор, поэтому результаты запусков не меняются
        _WARM.add(j["id"])
        wrng = np.random.default_rng(10 ** 6)
        wprob = Problem(s, refs, tidx, kidx, wrng, screen=j.get("screen"))
        run_search(wprob, cls(wprob, **{**kw, "seed_ics": None} if "seed_ics" in kw else kw),
                   min(int(j["budget"]), int(j["warmup"])), stop_on_target=False)
        t0 = time.time()
    strat = cls(prob, **kw)
    res = run_search(prob, strat, int(j["budget"]))
    rec = {"job": j["id"], "seed": r, "found_at": res["found_at"], "n_eval": int(res["n_eval"]),
           "n_disc": len(res["discoveries"]), "time_s": round(time.time() - t0, 3)}
    # стоимость (2026-10-12): шаги интегратора, время интегрирования и сопоставления; остальное время — накладные
    # расходы стратегии; для двухэтапной оценки — число остановленных траекторий и момент выключения правилом auto
    rec.update({"steps": int(prob.n_steps), "t_int": round(prob.t_int, 4), "t_match": round(prob.t_match, 4)})
    if j.get("screen"):
        lg = prob.screen_log
        rec.update({"n_stopped": int(prob.n_stopped), "screen_off_at": lg.get("off_at"),
                    "screen_batches": len(lg.get("hist", [])), "batches": lg.get("batch")})
    return rec


def run_one_safe(task):
    """run_one, но сбой одного запуска возвращается как запись с ошибкой, а не останавливает батч."""
    try:
        return run_one(task)
    except Exception:  # noqa: BLE001
        import traceback
        j, r, _ = task
        return {"job": j["id"], "seed": r, "error": traceback.format_exc(limit=8)}


# ============================================================================ хранение
def batch_dir(bid, out_root):
    d = Path(out_root) / bid
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_done(path):
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["seed"]] = rec
    return done


def append_line(path, rec):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def log(bdir, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(bdir / "run.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


class StopRequested(BaseException):
    """Менеджер rmq создал RMQ_STOP_FILE: сохранённое уже на диске, процесс должен завершиться за минуту.
    BaseException — чтобы цепочка не приняла остановку за сбой батча и не перешла к следующему."""


def stop_requested():
    p = os.environ.get("RMQ_STOP_FILE")
    return bool(p) and os.path.exists(p)


class Progress:
    """Прогресс задачи для дашборда rmq: строка «сделано/всего пояснение» в файл RMQ_PROGRESS_FILE, не чаще раза
    в 10 с. Единица — секунда оценки машинного времени (а не число запусков: их стоимость различается в тысячи
    раз), чтобы оставшееся время менеджер считал по стоимости работы."""

    def __init__(self, total_s):
        self.path = os.environ.get("RMQ_PROGRESS_FILE")
        self.total = max(1, int(round(total_s)))
        self.done = 0.0
        self.t = 0.0

    def add(self, est_s, note, force=False):
        self.done += est_s
        if self.path and (force or time.time() - self.t >= 10):
            self.t = time.time()
            self._write(min(int(self.done), self.total), note)

    def refresh(self, note, every_s=300):
        """Признак работы для менеджера, пока ни один запуск не закончился (долгие запуски идут десятки минут)."""
        if self.path and time.time() - self.t >= every_s:
            self.t = time.time()
            self._write(min(int(self.done), self.total), note)

    def finish(self, note):
        """Конец работы: 100 %, даже если часть запусков к старту батча уже досчитали другие (видеокарта)."""
        if self.path:
            self._write(self.total, note)

    def _write(self, done, note):
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                fh.write(f"{done}/{self.total} {note}")
        except OSError:
            pass                          # прогресс — подсказка, а не повод падать


def git_commit():
    try:
        g = "C:/Program Files/Git/cmd/git.exe" if os.name == "nt" else "git"
        c = subprocess.run([g, "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run([g, "status", "--porcelain", "--", "src", "experiments"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        return c + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return None


def summarize_job(j, recs):
    """Сводка задания в формате e20 (поле стратегии)."""
    from hsearch.search.harness import n_delta_from_runs, pn_curve
    seeds = sorted(recs)
    found = [recs[s]["found_at"] for s in seeds]
    B = j["budget"]
    q, lo, hi = n_delta_from_runs(found)
    grid, pn = pn_curve(found, B)
    ev = [x if x is not None else B for x in found]
    k = sum(x is not None for x in found)
    rec = {"found_at": found, "N_0.9": None if not np.isfinite(q) else q,
           "N_0.9_ci": [lo if np.isfinite(lo) else None, hi if np.isfinite(hi) else None],
           "success_rate": float(np.mean([f is not None for f in found])),
           "median": float(np.median([f if f is not None else np.inf for f in found])) if k else None,
           "pn_grid": grid.tolist(), "pn": pn.tolist(),
           "time_s": round(sum(recs[s]["time_s"] for s in seeds), 1), "budget": B, "R": len(found),
           "p_hit_mle": k / max(1, sum(ev)), "seeds": seeds}
    if j["strategy"] == "uniform" and k:
        rec["N0_theory"] = math.log(10) / rec["p_hit_mle"]
    if j.get("compare"):
        old = plan._old(j["compare"]["file"])
        st = (old or {}).get(j["compare"]["strategy"])
        if st:
            of = st["found_at"]
            n = min(len(of), len(found))
            mism = [seeds[i] for i in range(n) if of[i] != found[i]]
            rec["compare"] = {"file": j["compare"]["file"], "n": n, "equal": n - len(mism), "mismatch_seeds": mism[:20]}
    return rec


def write_summaries(bid, jobs, out_root, commit=None):
    bdir = batch_dir(bid, out_root)
    sdir = bdir / "summary"
    sdir.mkdir(exist_ok=True)
    by_var = {}
    for j in jobs:
        recs = read_done(bdir / f"{j['id']}.jsonl")
        if len(recs) < j["R"]:
            continue
        by_var.setdefault(plan.vname(j["var"]), []).append((j, summarize_job(j, recs)))
    for vn, items in by_var.items():
        p = sdir / f"{vn}.json"
        out = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        j0 = items[0][0]
        mp_ = bdir / "meta" / f"{vn}.json"
        meta = json.loads(mp_.read_text(encoding="utf-8")) if mp_.exists() else {"system": vn, "variant": j0["var"]}
        meta.update({"R": max(j["R"] for j, _ in items), "n_init": j0.get("n_init", 50), "batch_id": bid,
                     "commit": commit or git_commit()})
        out["_meta"] = meta
        for j, rec in items:
            out[j["id"].split("__", 1)[1]] = rec          # ключ — стратегия (или метка задания)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return by_var


# ============================================================================ команды
def resolve(names, B):
    if names == ["all"]:
        return [b for b in plan.ORDER if b not in plan.OPTIONAL]
    for n in names:
        if n not in B:
            raise SystemExit(f"нет батча {n}; есть: {', '.join(B)}")
    return names


def cmd_list(B):
    tot = tot_u = 0.0
    print(f"{'батч':6s} {'заданий':>7s} {'запусков':>8s} {'ожидаемо, ч':>11s} {'максимум, ч':>11s} "
          f"{f'при --cpu {CPU_DEFAULT}':>15s}  описание")
    for bid in plan.ORDER:
        b = B[bid]
        h = sum(x for x in (plan.est_hours(j) for j in b["jobs"]) if np.isfinite(x))
        hu = sum(x for x in (plan.est_hours(j, upper=True) for j in b["jobs"]) if np.isfinite(x))
        runs = sum(j["R"] for j in b["jobs"])
        if bid not in plan.OPTIONAL:
            tot += h
            tot_u += hu
        wt = f"{CPU_DEFAULT // b['threads']}×{b['threads']}"
        print(f"{bid:6s} {len(b['jobs']):7d} {runs:8d} {h:11.1f} {hu:11.1f} {wt:>15s}  {b['title']}"
              f"{'  [дополнительно]' if bid in plan.OPTIONAL else ''}")
    print(f"обязательные батчи: ожидаемо ≈ {tot:.0f} ч, максимум ≈ {tot_u:.0f} ч машинного времени при измеренной стоимости "
          f"(на 20 потоках); столько же длится задача rmq с --cpu {CPU_DEFAULT}, если процессор не делится с другими, "
          f"плюс ожидание и паузы")


def cmd_status(B, names, out_root):
    for bid in (resolve(names, B) if names else plan.ORDER):
        jobs = B[bid]["jobs"]
        bdir = Path(out_root) / bid
        done_runs = tot_runs = 0
        t_used = 0.0
        h_left = 0.0
        finished = 0
        for j in jobs:
            recs = read_done(bdir / f"{j['id']}.jsonl") if bdir.exists() else {}
            done_runs += len(recs)
            tot_runs += j["R"]
            t_used += sum(r["time_s"] for r in recs.values())
            eh = plan.est_hours(j)
            if np.isfinite(eh):
                h_left += eh * max(0.0, 1 - len(recs) / j["R"])
            finished += len(recs) >= j["R"]
        flag = "готов" if done_runs >= tot_runs else ("идёт/прерван" if done_runs else "не начат")
        print(f"{bid:6s} {flag:13s} задания {finished}/{len(jobs)}, запуски {done_runs}/{tot_runs}, "
              f"посчитано {t_used / 3600:.2f} ч (сумма по процессам), оценка остатка ≈ {h_left:.1f} ч")
        cmp = Path(out_root) / bid / "summary"
        if cmp.exists():
            bad = []
            for p in cmp.glob("*.json"):
                d = json.loads(p.read_text(encoding="utf-8"))
                for sname, rec in d.items():
                    if sname != "_meta" and isinstance(rec, dict) and rec.get("compare") and rec["compare"]["mismatch_seeds"]:
                        bad.append(f"{p.stem}:{sname} ({rec['compare']['equal']}/{rec['compare']['n']})")
            if bad:
                print(f"       расхождения со старыми результатами: {', '.join(bad[:8])}")


def _jobs_for(B, bid, args):
    jobs = []
    for j in B[bid]["jobs"]:
        j = dict(j)
        if args.R_cap:
            j["R"] = min(j["R"], args.R_cap)
        if args.budget_cap:
            j["budget"] = min(j["budget"], args.budget_cap)
        jobs.append(j)
    return jobs


def run_batch(bid, jobs, out_root, workers, threads, commit, prog=None, tag=""):
    """Все недосчитанные запуски батча в пуле процессов; результаты сразу на диск."""
    prog = prog or Progress(0)
    bdir = batch_dir(bid, out_root)
    tasks = []
    for j in jobs:
        done = read_done(bdir / f"{j['id']}.jsonl")
        eh = plan.est_hours(j)
        per_run = eh / j["R"] if np.isfinite(eh) else 0.0
        tasks += [(per_run, j, r) for r in range(j["R"]) if r not in done]
    if not tasks:
        write_summaries(bid, jobs, out_root, commit)
        log(bdir, "нечего считать: все запуски уже есть")
        return
    # сначала самые долгие запуски: меньше хвост, когда остальные процессы уже свободны
    tasks.sort(key=lambda t: (-t[0], t[1]["id"], t[2]))
    est = sum(t[0] for t in tasks)
    info = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "workers": workers, "threads": threads,
            "python": sys.version.split()[0], "numpy": np.__version__, "runs": len(tasks)}
    try:
        import numba
        info["numba"] = numba.__version__
    except Exception:  # noqa: BLE001
        pass
    with open(bdir / "run_info.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(info) + "\n")
    log(bdir, f"старт {bid}: {len(tasks)} запусков, оценка ≈ {est:.1f} ч машинного времени, "
              f"процессов {workers} × потоков {threads}, код {commit}")
    remaining, job_of, err_of = {}, {}, {}
    for _, j, _r in tasks:
        remaining[j["id"]] = remaining.get(j["id"], 0) + 1
        job_of[j["id"]] = j
    est_of = {(j["id"], r): pr for pr, j, r in tasks}
    t_start, n_done, est_done = time.time(), 0, 0.0
    # процессы пула наследуют окружение: в каждом не больше threads потоков numba и BLAS
    # (rmq выставляет эти переменные равными всей заявке --cpu)
    for k in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(threads)
    import multiprocessing as mp
    pool = mp.get_context("spawn").Pool(processes=workers, initializer=worker_init, initargs=(threads,))
    try:
        payload = [(j, r, str(bdir / "meta")) for _, j, r in tasks]
        it = pool.imap_unordered(run_one_safe, payload, chunksize=1)
        while n_done < len(tasks):
            # ждём результат не дольше 5 с, чтобы вовремя заметить RMQ_STOP_FILE (на остановку есть минута)
            try:
                rec = it.next(timeout=5)
            except mp.TimeoutError:
                if stop_requested():
                    raise StopRequested()
                prog.refresh(f"{bid}{tag}: {n_done}/{len(tasks)} запусков, идёт {(time.time() - t_start) / 60:.0f} мин")
                continue
            if stop_requested():
                raise StopRequested()
            n_done += 1
            est_done += est_of[(rec["job"], rec["seed"])]
            prog.add(3600.0 * est_of[(rec["job"], rec["seed"])], f"{bid}{tag}: {n_done}/{len(tasks)} запусков")
            remaining[rec["job"]] -= 1
            if "error" in rec:
                err_of[rec["job"]] = err_of.get(rec["job"], 0) + 1
                log(bdir, f"ошибка в запуске {rec['job']} r={rec['seed']} (не сохранён, повторится при следующем run):\n"
                          f"{rec['error']}")
            else:
                rec["commit"] = commit
                append_line(bdir / f"{rec['job']}.jsonl", rec)
            if remaining[rec["job"]] == 0:
                if rec["job"] in err_of:
                    log(bdir, f"задание {rec['job']}: запусков с ошибкой {err_of[rec['job']]}")
                else:
                    write_summaries(bid, [job_of[rec["job"]]], out_root, commit)
                    log(bdir, f"готово задание {rec['job']}")
            if n_done % max(1, len(tasks) // 100) == 0 or n_done == len(tasks):
                el = time.time() - t_start
                left = (est - est_done) * el / max(est_done, 1e-9)
                print(f"  {bid} [{n_done}/{len(tasks)}] {rec['job']} r={rec['seed']} {rec.get('time_s', 0):.1f} с; "
                      f"прошло {el / 3600:.2f} ч, осталось ≈ {left / 3600:.1f} ч", flush=True)
    except KeyboardInterrupt:
        pool.terminate()
        log(bdir, "остановлено; посчитанные запуски сохранены, продолжение — той же командой run")
        raise SystemExit(1)
    except StopRequested:
        pool.terminate()
        log(bdir, f"остановка по RMQ_STOP_FILE: {n_done} запусков этого раза на диске, недосчитанные начнутся заново")
        raise
    except BaseException:
        pool.terminate()                      # не досчитывать впустую остаток очереди пула
        raise
    finally:
        pool.close()
        pool.join()
    write_summaries(bid, jobs, out_root, commit)
    n_err = sum(err_of.values())
    log(bdir, f"батч {bid} завершён за {(time.time() - t_start) / 3600:.2f} ч"
              + (f"; запусков с ошибкой {n_err}" if n_err else ""))
    if n_err:
        raise RuntimeError(f"{bid}: запусков с ошибкой {n_err}, подробности в {bdir / 'run.log'}")


def _pool_shape(B, bid, cpu, args):
    """Процессы × потоки внутри выделенных cpu потоков."""
    th = max(1, min(args.threads or B[bid]["threads"], cpu))
    w = max(1, min(args.workers or cpu // th, cpu // th if cpu >= th else 1))
    return w, th


def cmd_run(B, names, args):
    out_root = Path(args.out) if args.out else OUT_ROOT
    bids = resolve(names, B)
    commit = git_commit()
    under_rmq = bool(os.environ.get("RMQ_TASK_ID"))
    cpu = int(os.environ.get("RMQ_CPU", "0") or 0) if under_rmq else (args.cpu or 0)
    if not under_rmq and not args.dry_run:
        raise SystemExit("тяжёлые расчёты на этом ПК идут только через очередь rmq: "
                         "python experiments/e50_batch_runner.py submit " + " ".join(names))
    plan_rows = []
    for bid in bids:
        jobs = _jobs_for(B, bid, args)
        bdir = Path(out_root) / bid
        left = sum(j["R"] - len(read_done(bdir / f"{j['id']}.jsonl")) for j in jobs)
        h = sum(x * max(0.0, 1 - len(read_done(bdir / f"{j['id']}.jsonl")) / j["R"])
                for j, x in ((j, plan.est_hours(j)) for j in jobs) if np.isfinite(x))
        w, th = _pool_shape(B, bid, cpu or CPU_DEFAULT, args)
        plan_rows.append((bid, jobs, left, h, w, th))
        print(f"  {bid}: осталось {left} запусков, ≈ {h:.1f} ч (оценка как на 20 потоках); процессов {w} × потоков {th}")
    print(f"код {commit}" + ("  — внимание: есть незафиксированные изменения" if commit and commit.endswith("+dirty") else ""))
    if under_rmq:
        print(f"задача rmq {os.environ.get('RMQ_TASK_ID')}, потоков {cpu}, ОЗУ {os.environ.get('RMQ_RAM_GB')} ГБ, "
              f"перезапусков {os.environ.get('RMQ_RESTART_COUNT', '0')}", flush=True)
    if args.dry_run:
        return
    failed = []
    todo = [row for row in plan_rows if row[2] > 0]
    prog = Progress(3600.0 * sum(row[3] for row in todo))      # оценка в секундах машинного времени
    for i, (bid, jobs, left, h, w, th) in enumerate(plan_rows):
        tag = f" (батч {todo.index(plan_rows[i]) + 1} из {len(todo)})" if left > 0 else ""
        try:
            run_batch(bid, jobs, out_root, w, th, commit, prog=prog, tag=tag)
        except StopRequested:
            print("остановка по просьбе менеджера rmq: при следующем запуске задача продолжит с сохранённых запусков",
                  flush=True)
            raise SystemExit(0)
        except Exception:  # noqa: BLE001 — сбой одного батча цепочки не останавливает следующие
            import traceback
            log(batch_dir(bid, out_root), "батч прерван ошибкой, перехожу к следующему:\n" + traceback.format_exc())
            failed.append(bid)
    prog.finish("готово" + (f", с ошибками: {', '.join(failed)}" if failed else ""))
    if failed:
        raise SystemExit(f"батчи с ошибками: {', '.join(failed)} (см. их run.log)")


RMQ = r"C:\Cursor\resources_manager\bin\rmq.cmd"
CPU_DEFAULT = 20          # руководство rmq: ПК простаивает, задача хорошо параллелится — заявлять все потоки
# пилот B0 через rmq (задача №1, 2026-10-10): 2 процесса × 2 потока — пик ОЗУ 815 МБ, т. е. ≈ 0.33 ГБ на процесс
# и ≈ 0.15 ГБ у главного; у длинных батчей рукав границ и бисекция хранят до ~10⁶ точек — +0.15 ГБ на процесс
RAM_PER_WORKER_GB = {2: 0.35, 4: 0.5}
RAM_PARENT_GB = 0.15
TASK_OVERHEAD_H = 1 / 60  # запуск процессов, загрузка numba и сводки — около минуты на батч (пилот B0: 27 с)


def _fmt_time(hours):
    m = max(2, int(math.ceil(hours * 60)))
    return f"{m // 60}h{m % 60:02d}m" if m >= 60 else f"{m}m"


def _request(B, bid, jobs, cpu, args, out_root):
    """Остаток батча и его заявка: (запусков, процессов, потоков, ОЗУ ГБ, часов работы без запаса).

    С --gpu-uniform в оценку времени не входят задания uniform батчей, которые считает видеокарта
    (e51_gpu_uniform.GPU_BATCHES): цепочка дойдёт до них, когда они уже готовы, и пропустит.
    """
    bdir = Path(out_root) / bid
    left = sum(j["R"] - len(read_done(bdir / f"{j['id']}.jsonl")) for j in jobs)
    gpu_b = set()
    if getattr(args, "gpu_uniform", False):
        import e51_gpu_uniform
        gpu_b = set(e51_gpu_uniform.GPU_BATCHES)
    cpu_jobs = [j for j in jobs if not (bid in gpu_b and j["strategy"] == "uniform")]
    h20 = sum(x * max(0.0, 1 - len(read_done(bdir / f"{j['id']}.jsonl")) / j["R"])
              for j, x in ((j, plan.est_hours(j, upper=args.upper)) for j in cpu_jobs) if np.isfinite(x))
    w, th = _pool_shape(B, bid, cpu, args)
    ram = (w * RAM_PER_WORKER_GB.get(th, 0.5) + RAM_PARENT_GB) * 1.2         # пик + 20 %
    return left, w, th, ram, h20 * 20.0 / (w * th) + TASK_OVERHEAD_H       # оценки сняты на 20 потоках


def cmd_submit(B, names, args):
    """Поставить батчи в очередь rmq с честной заявкой и --preempt requeue.

    По умолчанию — по задаче на батч. С --chain — одна задача, которая считает батчи по очереди: при заявке
    на весь процессор отдельные задачи шли бы одновременно и делили бы между собой те же потоки.
    """
    out_root = Path(args.out) if args.out else OUT_ROOT
    cpu = args.cpu or CPU_DEFAULT
    rows = []
    for bid in resolve(names, B):
        left, w, th, ram, t_h = _request(B, bid, _jobs_for(B, bid, args), cpu, args, out_root)
        if left <= 0:
            print(f"{bid}: все запуски уже есть, ставить нечего")
            continue
        rows.append((bid, left, w, th, ram, t_h))
        print(f"{bid}: {left} запусков, процессов {w} × потоков {th}; ОЗУ {ram:.1f} ГБ, ≈ {_fmt_time(t_h)} работы")
    if not rows:
        return
    groups = [rows] if args.chain else [[r] for r in rows]
    for g in groups:
        bids = [r[0] for r in g]
        ram = math.ceil(max(r[4] for r in g) * 10) / 10
        t_req = _fmt_time(sum(r[5] for r in g) * 1.3)                  # +30 % запаса по руководству rmq
        name = f"e50-{bids[0]}" if len(bids) == 1 else f"e50-{bids[0]}..{bids[-1]}"
        cmd = [RMQ, "submit", "--cpu", str(cpu), "--ram", f"{ram:g}", "--time", t_req,
               "--preempt", "requeue", "--name", name, "--cwd", str(ROOT), "--",
               sys.executable, "experiments/e50_batch_runner.py", "run", *bids]
        if args.out:
            cmd += ["--out", str(args.out)]
        if args.R_cap:
            cmd += ["--R-cap", str(args.R_cap)]
        if args.budget_cap:
            cmd += ["--budget-cap", str(args.budget_cap)]
        print(f"задача {name}: заявка --cpu {cpu} --ram {ram:g} --time {t_req}")
        if args.dry_run:
            print("   " + " ".join(cmd))
            continue
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout, r.stderr)
            raise SystemExit(f"rmq submit вернул {r.returncode}")
        if r.stderr.strip():
            print("   rmq: " + r.stderr.strip())
        tid = r.stdout.strip().splitlines()[-1].strip()
        print(f"   задача {tid}: http://127.0.0.1:8777/#task={tid}")
        for bid in bids:
            with open(batch_dir(bid, out_root) / "rmq_tasks.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"task": tid, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "chain": bids,
                                     "cmd": cmd}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["list", "submit", "run", "status", "summarize"])
    ap.add_argument("batches", nargs="*")
    ap.add_argument("--cpu", type=int, default=None, help=f"submit: потоков для задачи rmq (по умолчанию {CPU_DEFAULT})")
    ap.add_argument("--chain", action="store_true", help="submit: все батчи одной задачей rmq, по очереди")
    ap.add_argument("--gpu-uniform", dest="gpu_uniform", action="store_true",
                    help="submit: не считать в заявке времени задания uniform, которые берёт видеокарта (e51)")
    ap.add_argument("--upper", action="store_true", help="submit: заявлять время по верхней оценке")
    ap.add_argument("--workers", type=int, default=None, help="процессов внутри задачи (по умолчанию cpu // threads)")
    ap.add_argument("--threads", type=int, default=None, help="numba-потоков на процесс (по умолчанию — свой у батча)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None, help="каталог результатов (по умолчанию results/e50)")
    ap.add_argument("--R-cap", dest="R_cap", type=int, default=None, help="для пробного запуска: не больше R запусков")
    ap.add_argument("--budget-cap", type=int, default=None, help="для пробного запуска: не больше бюджета")
    args = ap.parse_args()
    B = plan.batches()
    if args.command == "list":
        cmd_list(B)
    elif args.command == "status":
        cmd_status(B, args.batches, Path(args.out) if args.out else OUT_ROOT)
    elif args.command == "summarize":
        for bid in resolve(args.batches, B):
            write_summaries(bid, B[bid]["jobs"], Path(args.out) if args.out else OUT_ROOT)
            print(f"{bid}: сводки обновлены")
    elif args.command == "submit":
        if not args.batches:
            raise SystemExit("укажите батчи, например: submit B0")
        cmd_submit(B, args.batches, args)
    else:
        if not args.batches:
            raise SystemExit("укажите батчи, например: run B0 --dry-run")
        cmd_run(B, args.batches, args)


if __name__ == "__main__":
    main()
