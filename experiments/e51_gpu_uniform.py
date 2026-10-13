"""Равномерный перебор батчей e50 на видеокарте (CuPy, FP64) — разгрузка цепочки на CPU.

Для R1 и R4 базовая линия «равномерный перебор» (стратегия uniform) — самая дорогая часть батчей B2a, B4a,
B4b, B2b1–B2b5, а её траектории почти все полной длины. Видеокарта считает эти задания теми же запусками,
что и CPU:
  * начальные условия запуска r — тот же поток default_rng(1000 + r) и sample_box пачками по 500;
  * отпечатки — hsearch.gpu.GpuFlow (тот же РК4 и те же признаки, совпадение бит в бит), метки — тем же
    hsearch.identify.match с теми же эталонами (e50_batch_runner.get_problem);
  * found_at, n_eval и n_disc — по правилам hsearch.search.harness.run_search.

Результаты пишутся в results/e50_gpu/<батч>/<задание>.jsonl (строка на запуск, это и чекпоинт). Законченное
задание переносится в results/e50/<батч>/ (с метаданными варианта), только если цепочка ещё не начала этот
батч (нет run_info.jsonl). Начав батч, цепочка видит готовые запуски и пропускает их. Если цепочка начала
батч раньше, задание остаётся в results/e50_gpu и служит сверкой.

Перенос разрешён только после проверки (команда check): признаки GPU и CPU на выборках всех вариантов и
первые запуски дешёвых заданий должны совпасть.

    python experiments/e51_gpu_uniform.py submit auto         # проверка, затем перебор — одной задачей rmq
    python experiments/e51_gpu_uniform.py submit check        # только проверка
    python experiments/e51_gpu_uniform.py status
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import e50_batch_runner as runner  # noqa: E402
import e50_batches as plan  # noqa: E402

GPU_BATCHES = ["B2a", "B4a", "B4b", "B2b1", "B2b2", "B2b3", "B2b4", "B2b5", "B8"]   # в порядке, в каком их ждёт цепочка
STAGE = ROOT / "results" / "e50_gpu"
CHECK_OK = STAGE / "check_ok.json"
CHUNK = plan.UNIFORM_BATCH            # 500: пачка равномерного перебора, по ней считается n_eval
HOST_DOUBLES = 50_000_000             # не больше ~400 МБ начальных условий за раунд


def log(msg):
    STAGE.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(STAGE / "run.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpu_jobs(bids):
    B = plan.batches()
    return [(bid, j) for bid in bids for j in B[bid]["jobs"] if j["strategy"] == "uniform"]


def _limit_vram():
    import cupy as cp
    g = float(os.environ.get("RMQ_VRAM_GB", "0") or 0)
    if g > 0:
        cp.get_default_memory_pool().set_limit(size=int(0.9 * g * 2**30))


_GF = {}


def _gpu_flow(s):
    from hsearch.gpu import GpuFlow
    if s.name not in _GF:
        _GF[s.name] = GpuFlow(s)
    return _GF[s.name]


def _max_gpu(d):
    return (1 << 20) if d <= 3 else (1 << 19) if d <= 6 else (1 << 18)


def _stop_check():
    if runner.stop_requested():
        raise runner.StopRequested()


def labels_gpu(s, refs, X):
    """Метки начальных условий X: отпечатки на видеокарте, сопоставление — тем же match, что на CPU."""
    from hsearch.identify import match
    gf = _gpu_flow(s)
    lab = np.empty(len(X), dtype=np.int64)
    m = _max_gpu(s.dim)
    for a in range(0, len(X), m):
        feats, finals = gf.run_batch(X[a:a + m], stop_check=_stop_check)
        lab[a:a + m], _ = match(s, feats, finals, refs, tol=6.0)
    return lab


def run_uniform_job(bid, j, commit, seeds=None, stage=STAGE, prog=None, note=""):
    """Запуски равномерного перебора задания j на видеокарте; каждая законченная строка сразу на диск."""
    prog = prog or runner.Progress(0)
    from hsearch.identify import LAB_UNKNOWN
    s, refs, tidx, kidx, _ = runner.get_problem(j["var"], j.get("targets"))
    sdir = Path(stage) / bid
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f"{j['id']}.jsonl"
    done = runner.read_done(spath)
    R, budget = j["R"], j["budget"]
    todo = [r for r in (range(R) if seeds is None else seeds) if r not in done]
    if not todo:
        return spath
    est = float(j.get("est_evals") or budget / 6)
    cap = max(CHUNK, (HOST_DOUBLES // (len(todo) * s.dim)) // CHUNK * CHUNK)
    blk = min(cap, max(CHUNK, int(math.ceil(0.3 * est / CHUNK)) * CHUNK))   # ~30 % ожидаемого first hit
    tset, kset = np.asarray(tidx, dtype=np.int64), np.asarray(kidx, dtype=np.int64)
    runs = {r: {"rng": np.random.default_rng(1000 + r), "pos": 0, "disc": 0} for r in todo}
    t0 = time.time()
    log(f"{bid} {j['id']}: запусков {len(todo)} из {R}, бюджет {budget}, пачка на запуск {blk}")
    while runs:
        parts = []
        for r, st in runs.items():
            m = min(blk, budget - st["pos"])
            ics = s.sample_box(int(math.ceil(m / CHUNK)) * CHUNK, st["rng"])[:m]   # поток чисел — как у CPU
            parts.append((r, ics))
        X = np.vstack([ics for _, ics in parts])
        lab_all = labels_gpu(s, refs, X)
        off = 0
        for r, ics in parts:
            st = runs[r]
            lab = lab_all[off:off + len(ics)]
            off += len(ics)
            disc = (lab == LAB_UNKNOWN) | ((lab >= 0) & ~np.isin(lab, kset) & ~np.isin(lab, tset))
            hit = np.flatnonzero(np.isin(lab, tset))
            rec = None
            if hit.size:
                h = st["pos"] + int(hit[0]) + 1
                n_eval = min(budget, int(math.ceil(h / CHUNK)) * CHUNK)
                n_disc = st["disc"] + int(disc[:n_eval - st["pos"]].sum())
                rec = {"found_at": h, "n_eval": n_eval, "n_disc": min(20, n_disc)}
            else:
                st["disc"] += int(disc.sum())
                st["pos"] += len(ics)
                if st["pos"] >= budget:
                    rec = {"found_at": None, "n_eval": budget, "n_disc": min(20, st["disc"])}
            if rec is not None:
                rec = {"job": j["id"], "seed": r, **rec, "time_s": round(time.time() - t0, 3), "dev": "gpu",
                       "commit": commit}
                runner.append_line(spath, rec)
                del runs[r]
                prog.add(1.0, f"{note}{bid} {j['id']}")
    log(f"{bid} {j['id']}: готово за {(time.time() - t0) / 60:.1f} мин")
    return spath


def merge_job(bid, j):
    """Перенести законченное задание в results/e50/<батч>/, если цепочка ещё не начала этот батч."""
    recs = runner.read_done(STAGE / bid / f"{j['id']}.jsonl")
    if len(recs) < j["R"]:
        return "не закончено"
    tdir = runner.OUT_ROOT / bid
    if (tdir / "run_info.jsonl").exists():
        return "цепочка уже считает этот батч — оставлено в results/e50_gpu для сверки"
    tpath = tdir / f"{j['id']}.jsonl"
    if tpath.exists():
        return "файл уже есть — не трогаю"
    tdir.mkdir(parents=True, exist_ok=True)
    s, refs, tidx, kidx, _ = runner.get_problem(j["var"], j.get("targets"))
    runner.write_meta(tdir / "meta", j["var"], s, refs, tidx)
    tmp = tpath.with_name(tpath.name + ".gpu.tmp")
    tmp.write_text("".join(json.dumps(recs[r], ensure_ascii=False) + "\n" for r in sorted(recs)), encoding="utf-8")
    os.replace(tmp, tpath)
    return "перенесено в results/e50"


# ============================================================================ команды
def cmd_check(args):
    """Совпадение GPU и CPU: признаки на выборках всех вариантов, затем первые запуски дешёвых заданий."""
    from hsearch.integrate import F_LAM, run_batch
    _limit_vram()
    commit = runner.git_commit()
    report = {"commit": commit, "features": {}, "runs": {}, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    ok = True
    variants = {}
    for bid, j in gpu_jobs(GPU_BATCHES):
        variants.setdefault(plan.vname(j["var"]), j["var"])
    for vn, var in variants.items():
        s = runner.build_variant(var)
        n = 128 if s.n_tr >= 100_000 else 512
        X = s.sample_box(n, np.random.default_rng(12345))
        t = time.perf_counter()
        fc, zc = run_batch(s, X)
        t_cpu = time.perf_counter() - t
        gf = _gpu_flow(s)
        g0 = gf.gpu_time
        fg, zg = gf.run_batch(X)
        t_gpu = gf.gpu_time - g0
        cols = [c for c in range(fc.shape[1]) if c != F_LAM]
        neq = lambda a, b: ~((a == b) | (np.isnan(a) & np.isnan(b)))  # noqa: E731 — NaN у ушедших = NaN
        diff_rows = int(np.sum(np.any(neq(fc[:, cols], fg[:, cols]), axis=1) | np.any(neq(zc, zg), axis=1)))
        dlam = float(np.max(np.abs(fc[:, F_LAM] - fg[:, F_LAM])))
        esc = float(fc[:, 0].mean())
        report["features"][vn] = {"n": n, "rows_diff": diff_rows, "max_dlam": dlam, "esc_frac": esc,
                                  "t_cpu_s": round(t_cpu, 2), "t_gpu_s": round(t_gpu, 2)}
        ok &= diff_rows == 0 and dlam < 1e-12
        print(f"признаки {vn:28s} n={n:4d}: строк с отличием {diff_rows}, max|dλ| = {dlam:.1e}, ушли {esc:.2f}; "
              f"CPU {t_cpu:.1f} с, GPU {t_gpu:.1f} с", flush=True)
    # первые запуски дешёвых заданий: CPU (run_one из запускателя) против GPU
    picks = {"se11_box2__uniform": 5, "le1_box2__uniform": 3, "chua_I_box2__uniform": 3,
             "jafari2026_lc_box2__uniform": 3, "coupled_se11_k2_eps0.01_c1__uniform": 1}
    tmp_stage = STAGE / "check"
    for bid, j in gpu_jobs(GPU_BATCHES):
        if j["id"] not in picks:
            continue
        seeds = list(range(picks[j["id"]]))
        p = tmp_stage / bid / f"{j['id']}.jsonl"
        if p.exists():
            p.unlink()
        run_uniform_job(bid, j, commit, seeds=seeds, stage=tmp_stage)
        g = runner.read_done(p)
        res = []
        for r in seeds:
            c = runner.run_one((j, r, str(tmp_stage / "meta")))
            same = all(c[k] == g[r][k] for k in ("found_at", "n_eval", "n_disc"))
            ok &= same
            res.append({"seed": r, "cpu": [c["found_at"], c["n_eval"], c["n_disc"]],
                        "gpu": [g[r]["found_at"], g[r]["n_eval"], g[r]["n_disc"]], "same": same})
        report["runs"][j["id"]] = res
        print(f"запуски {j['id']}: " + "; ".join(f"r={x['seed']} CPU {x['cpu']} GPU {x['gpu']}" for x in res), flush=True)
    report["ok"] = bool(ok)
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "check_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if ok:
        CHECK_OK.write_text(json.dumps({"commit": commit, "time": report["time"]}), encoding="utf-8")
        print("проверка пройдена: GPU совпадает с CPU", flush=True)
    else:
        CHECK_OK.unlink(missing_ok=True)
        raise SystemExit("проверка НЕ пройдена: см. results/e50_gpu/check_report.json")


def cmd_run(args):
    if not os.environ.get("RMQ_TASK_ID") and not args.dry_run:
        raise SystemExit("видеокарта на этом ПК — только через очередь rmq: python experiments/e51_gpu_uniform.py submit run")
    commit = runner.git_commit()
    jobs = gpu_jobs(args.batches or GPU_BATCHES)
    merge_ok = CHECK_OK.exists()
    for bid, j in jobs:
        print(f"  {bid} {j['id']}: R = {j['R']}, бюджет {j['budget']}", flush=True)
    if args.dry_run:
        return
    _limit_vram()
    log(f"старт: заданий {len(jobs)}, код {commit}, задача rmq {os.environ.get('RMQ_TASK_ID')}, "
        f"перезапусков {os.environ.get('RMQ_RESTART_COUNT', '0')}; перенос в results/e50 {'разрешён' if merge_ok else 'ЗАПРЕЩЁН (нет check_ok)'}")
    # прогресс для дашборда rmq — в запусках (внутри задания они близки по стоимости)
    prog = runner.Progress(sum(j["R"] - len(runner.read_done(STAGE / bid / f"{j['id']}.jsonl")) for bid, j in jobs))
    for k, (bid, j) in enumerate(jobs):
        if (runner.OUT_ROOT / bid / "run_info.jsonl").exists() and not (STAGE / bid / f"{j['id']}.jsonl").exists():
            log(f"{bid} {j['id']}: цепочка уже начала батч — пропускаю")
            continue
        if (runner.OUT_ROOT / bid / f"{j['id']}.jsonl").exists():
            log(f"{bid} {j['id']}: уже есть в results/e50 — пропускаю")
            continue
        run_uniform_job(bid, j, commit, prog=prog, note=f"задание {k + 1} из {len(jobs)}: ")
        if merge_ok:
            log(f"{bid} {j['id']}: {merge_job(bid, j)}")
    prog.finish("готово")
    log("все задания пройдены")


def cmd_status(args):
    for bid, j in gpu_jobs(GPU_BATCHES):
        g = runner.read_done(STAGE / bid / f"{j['id']}.jsonl")
        c = runner.read_done(runner.OUT_ROOT / bid / f"{j['id']}.jsonl")
        gpu_in_main = sum(1 for r in c.values() if r.get("dev") == "gpu")
        print(f"{bid:5s} {j['id']:40s} GPU {len(g):3d}/{j['R']:3d}   в results/e50: {len(c):3d} (из них GPU {gpu_in_main})")
    if (STAGE / "check_report.json").exists():
        rep = json.loads((STAGE / "check_report.json").read_text(encoding="utf-8"))
        print(f"проверка: {'пройдена' if rep.get('ok') else 'НЕ пройдена'} ({rep.get('time')}, код {rep.get('commit')})")


RMQ = runner.RMQ


def cmd_auto(args):
    """Проверка (если её ещё нет для этого кода), затем перебор — одной задачей rmq: перенос в results/e50
    начинается только после пройденной проверки, и ни одной лишней постановки в очередь."""
    commit = runner.git_commit()
    ok = json.loads(CHECK_OK.read_text(encoding="utf-8")) if CHECK_OK.exists() else {}
    if ok.get("commit") != commit:
        cmd_check(args)
    cmd_run(args)


def cmd_submit(args):
    what = args.batches[0] if args.batches else "auto"
    rest = args.batches[1:]
    # видеокарте нужен один поток CPU: ядра, выборка начальных условий и match; numba-эталоны — тоже на нём.
    # Небольшая задача стартует и рядом с задачей на весь процессор: менеджер ужимает работающие (до четверти)
    if what == "check":
        req = ["--cpu", "1", "--ram", "3", "--vram", "2", "--time", "15m"]
    else:
        req = ["--cpu", str(args.cpu or 1), "--ram", str(args.ram or 3), "--vram", "2", "--time", args.time or "5h"]
    cmd = [RMQ, "submit", *req, "--preempt", "requeue", "--name", f"e51-gpu-{what}", "--cwd", str(ROOT), "--",
           sys.executable, "experiments/e51_gpu_uniform.py", what, *rest]
    print(" ".join(cmd))
    if args.dry_run:
        return
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"rmq submit вернул {r.returncode}")
    tid = r.stdout.strip().splitlines()[-1].strip()
    print(f"задача {tid}: http://127.0.0.1:8777/#task={tid}")
    STAGE.mkdir(parents=True, exist_ok=True)
    with open(STAGE / "rmq_tasks.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"task": tid, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "cmd": cmd}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "run", "auto", "status", "submit"])
    ap.add_argument("batches", nargs="*")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu", type=int, default=None)
    ap.add_argument("--ram", type=float, default=None)
    ap.add_argument("--time", default=None)
    args = ap.parse_args()
    if args.command == "check":
        if not os.environ.get("RMQ_TASK_ID"):
            raise SystemExit("проверка использует видеокарту — только через rmq: python experiments/e51_gpu_uniform.py submit check")
        cmd_check(args)
    elif args.command in ("run", "auto"):
        if args.command == "auto" and not os.environ.get("RMQ_TASK_ID"):
            raise SystemExit("видеокарта на этом ПК — только через rmq: python experiments/e51_gpu_uniform.py submit auto")
        try:
            cmd_auto(args) if args.command == "auto" else cmd_run(args)
        except runner.StopRequested:
            # менеджер просит освободить ресурсы: законченные запуски уже в results/e50_gpu, остальные начнутся заново
            log("остановка по RMQ_STOP_FILE; при следующем запуске задача продолжит с сохранённых запусков")
            raise SystemExit(0)
    elif args.command == "status":
        cmd_status(args)
    else:
        cmd_submit(args)


if __name__ == "__main__":
    main()
