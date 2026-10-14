"""Рисунки аттракторов для статьи -> results/figures/fig3X_*.png|pdf

fig30_chua_I_hidden        — Чуа I: два скрытых аттрактора, равновесия, неустойчивые многообразия F±
fig31_chua_II_coexisting   — Чуа II: пять сосуществующих аттракторов + сечение бассейнов
fig32_gallery_3d           — галерея скрытых аттракторов уровня A в 3D с равновесиями
fig33_li_sprott_4d         — 4D Ли–Спротт: тор и пара странных аттракторов, сечение бассейнов
fig34_maps_phase_basin     — фазово-бассейновые портреты отображений (Хуинь, Jiang)
fig35_jafari2026           — система Jafari 2026 без FP и PP: цикл, хрупкий хаос, λ(a3)
fig36_lorenz_transient     — отрицательный контроль: переходный хаос Лоренца (4, 29, 2)
fig37_d2_mechanism         — семейство D2: время ухода у границы и цепи расщепления по уровням
fig38_sus_chains           — цепи расщепления на карте бассейнов отображения Jiang NF I e
fig40_hiddenness           — мера скрытости h_ε
fig41_verification         — показатель Ляпунова по окнам и выживание при длинном интегрировании
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from hsearch import systems, viz  # noqa: E402
from hsearch.identify import LAB_ESCAPE, calibrate_reference, fp_references_from_equilibria, match  # noqa: E402
from hsearch.integrate import F_ESC, F_LAM, F_TESC, nfeat, run_batch  # noqa: E402
from hsearch.viz import ESCAPE_GREY, INK, INK2, MUTED, SERIES, save  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)


def attractor(s, ic, T=None, skip=None):
    """Точки траектории на аттракторе: переходный участок отбрасывается."""
    n_tr = s.n_tr if skip is None else skip
    n = n_tr + (int(T / s.dt) if T else s.n_obs)
    tr = s.traj(np.asarray(ic, float), n=n)
    tr = tr[n_tr:]
    return tr[~np.isnan(tr).any(axis=1)]


def style3d(ax, labels=("x", "y", "z")):
    ax.set_xlabel(labels[0], fontsize=8, labelpad=1); ax.set_ylabel(labels[1], fontsize=8, labelpad=1)
    ax.set_zlabel(labels[2], fontsize=8, labelpad=1)
    ax.tick_params(labelsize=6, pad=0)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.98, 0.98, 0.97, 1.0)); pane.set_edgecolor("#e1e0d9")
    ax.grid(False)


def basin_grid(s, refs, plane, fixed, n1=360, n2=240, rng=None):
    """Карта меток на плоскости: plane = (i, j, lo_i, hi_i, lo_j, hi_j), fixed — значения прочих координат."""
    i, j, li, hi_, lj, hj = plane
    a = np.linspace(li, hi_, n1); b = np.linspace(lj, hj, n2)
    A, Bm = np.meshgrid(a, b)
    ics = np.tile(np.asarray(fixed, float), (A.size, 1))
    ics[:, i] = A.ravel(); ics[:, j] = Bm.ravel()
    feats, finals = run_batch(s, ics)
    labels, z = match(s, feats, finals, refs)
    return a, b, labels.reshape(n2, n1)


def draw_basin(ax, a, b, img, names, colors, unknown_color=INK):
    keys = [LAB_ESCAPE, -2] + list(range(len(names)))
    cols = [ESCAPE_GREY, unknown_color] + colors
    lut = {k: q for q, k in enumerate(keys)}
    idx = np.vectorize(lambda v: lut.get(int(v), 1))(img)
    ax.imshow(idx, origin="lower", extent=[a[0], a[-1], b[0], b[-1]], cmap=ListedColormap(cols),
              vmin=-0.5, vmax=len(keys) - 0.5, aspect="auto", interpolation="nearest")
    ax.grid(False)


# ---------------------------------------------------------------------------------------------
def fig_chua_I():
    s = systems.get("chua_I")
    rng = np.random.default_rng(30)
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for name, c in (("HA_a", SERIES[0]), ("HA_b", SERIES[2])):
        tr = attractor(s, s.known[name], T=300)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=0.35)
    eq = s.equilibria
    ax.scatter(*eq[0], color=SERIES[1], s=40, depthshade=False, label="F0 (устойчивый фокус-узел)")
    for e in eq[1:]:
        ax.scatter(*e, color=SERIES[7], marker="x", s=50, depthshade=False)
    # неустойчивые многообразия F±: траектории из малых окрестностей уходят на бесконечность
    from hsearch.baselines.geometric import _jac
    J = _jac(s)
    for e in eq[1:]:
        w, V = np.linalg.eig(J(e))
        k = int(np.argmax(np.real(w)))
        v = np.real(V[:, k]); v /= np.linalg.norm(v)
        for sgn in (1, -1):
            tr = s.traj(e + sgn * 1e-4 * v, n=int(60 / s.dt))
            tr = tr[~np.isnan(tr).any(axis=1)]
            tr = tr[np.linalg.norm(tr, axis=1) < 25]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=MUTED, lw=0.8, ls="--")
    ax.scatter([], [], color=SERIES[7], marker="x", label="F± (седло-фокусы)")
    ax.plot([], [], color=MUTED, ls="--", label="неустойчивые многообразия F±: на ∞ и к F0")
    ax.plot([], [], color=SERIES[0], label="скрытые аттракторы A и B")
    ax.set_xlim(-12, 12); ax.set_ylim(-4, 4); ax.set_zlim(-15, 15)
    style3d(ax); ax.view_init(elev=18, azim=-60)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Чуа I: скрытые аттракторы и равновесия", fontsize=10)
    # сечение бассейнов z = 0
    ax2 = fig.add_subplot(1, 2, 2)
    p = RES / "e01_chua_basin_z0_labels.npy"
    if p.exists():
        img = np.load(p)
        d = json.loads((RES / "e01_chua_table1.json").read_text(encoding="utf-8"))["basin_z0"]
        a = np.linspace(d["x"][0], d["x"][1], img.shape[1]); b = np.linspace(d["y"][0], d["y"][1], img.shape[0])
        draw_basin(ax2, a, b, img, ["F0", "Fp", "Fm", "HA_a", "HA_b"],
                   [SERIES[1], SERIES[7], SERIES[6], SERIES[0], SERIES[2]])
        for name, c in (("HA_a", "#0d366b"), ("HA_b", "#0d4f3a")):
            tr = attractor(s, s.known[name], T=200)
            sel = np.abs(tr[:, 2]) < 0.15
            ax2.plot(tr[sel, 0], tr[sel, 1], ".", ms=1.2, color=c)
        ax2.set_xlabel("x(0)"); ax2.set_ylabel("y(0)")
        ax2.set_title("Бассейны на плоскости z(0) = 0 (точки — сечение аттракторов |z| < 0.15)", fontsize=9)
        ax2.legend(handles=[Patch(color=ESCAPE_GREY, label="уход"), Patch(color=SERIES[1], label="F0"),
                            Patch(color=SERIES[0], label="аттрактор A"), Patch(color=SERIES[2], label="аттрактор B")],
                   fontsize=7, loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, FIG / "fig30_chua_I_hidden")


def fig_chua_II():
    s = systems.get("chua_II")
    rng = np.random.default_rng(31)
    fig = plt.figure(figsize=(15, 4.8))
    eq = s.equilibria
    trs = {"LC": attractor(s, s.known["LC"], T=12), "HA_a": attractor(s, s.known["HA_a"], T=150),
           "HA_b": attractor(s, s.known["HA_b"], T=150)}
    for col, zoom in ((1, False), (2, True)):
        ax = fig.add_subplot(1, 3, col, projection="3d")
        if not zoom:
            ax.plot(trs["LC"][:, 0], trs["LC"][:, 1], trs["LC"][:, 2], color=SERIES[6], lw=0.9, label="предельный цикл LC (0.99 бокса)")
        for name, c, lab in (("HA_a", SERIES[0], "хаос A (0.0014)"), ("HA_b", SERIES[2], "хаос B (0.0013)")):
            tr = trs[name]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=0.35 if zoom else 0.8, label=lab)
        ax.scatter(*eq[0], color=SERIES[7], marker="x", s=50, depthshade=False, label="F0 (седло-фокус)")
        for e in eq[1:]:
            ax.scatter(*e, color=SERIES[1], s=40, depthshade=False)
        ax.scatter([], [], color=SERIES[1], label="F± (устойчивые)")
        style3d(ax); ax.view_init(elev=15, azim=-55)
        if zoom:
            ax.set_title("увеличено: хаотические аттракторы A, B и равновесия", fontsize=9)
        else:
            ax.legend(fontsize=6.5, loc="upper left")
            ax.set_title("Чуа II: пять сосуществующих аттракторов", fontsize=10)
    ax2 = fig.add_subplot(1, 3, 3)
    from e20_search_benchmark import build_problem
    s2, refs, tidx, kidx = build_problem("chua_II", np.random.default_rng(100))
    names = [r.name for r in refs]
    colors = []
    for n in names:
        colors.append({"E0": SERIES[7], "E1": SERIES[1], "E2": "#b85a2a", "LC": SERIES[6], "HA_a": SERIES[0], "HA_b": SERIES[2]}.get(n, SERIES[5]))
    a, b, img = basin_grid(s, refs, (0, 1, -3, 3, -1.5, 1.5), [0, 0, 0], n1=300, n2=200)
    np.save(RES / "e40_chua_II_basin_z0.npy", img)
    draw_basin(ax2, a, b, img, names, colors)
    ax2.set_xlabel("x(0)"); ax2.set_ylabel("y(0)")
    ax2.set_title("Бассейны на плоскости z(0) = 0", fontsize=9)
    ax2.legend(handles=[Patch(color=SERIES[6], label="цикл LC"), Patch(color=SERIES[1], label="F+"),
                        Patch(color="#b85a2a", label="F−"), Patch(color=SERIES[0], label="хаос A"),
                        Patch(color=SERIES[2], label="хаос B")], fontsize=7, loc="upper right", frameon=True)
    fig.tight_layout()
    # подпись оси z соседней трёхмерной панели не входит в её рамку: сдвигаем плоскую панель
    pos = ax2.get_position()
    ax2.set_position([pos.x0 + 0.035, pos.y0, pos.width - 0.035, pos.height])
    save(fig, FIG / "fig31_chua_II_coexisting")


GALLERY = [("chua_I", "HA_a", "Чуа I: устойчивое равновесие"), ("chua_II", "HA_a", "Чуа II: устойчивые F±"),
           ("ne4", "HA", "NE4: без равновесий"), ("wangchen1", "HA", "Ванг–Чен #1: без равновесий"),
           ("wei_ne2", "HA", "Вэй (NE2): без равновесий"), ("wangchen2", "HA", "Ванг–Чен #2: одно устойчивое"),
           ("se3", "HA", "SE3: одно устойчивое"), ("se11", "HA", "SE11: одно устойчивое"),
           ("le1", "HA", "LE1: линия равновесий"), ("rabinovich_100", "HA", "Рабинович, r = 100"),
           ("jafari2026_lc", "LC", "Jafari 2026: цикл (нет FP и PP)"), ("jafari2026_chaos", "HA", "Jafari 2026: хаос")]


def fig_gallery():
    fig = plt.figure(figsize=(12, 11.5))
    for k, (name, key, title) in enumerate(GALLERY):
        s = systems.get(name)
        ax = fig.add_subplot(4, 3, k + 1, projection="3d")
        T = 400 if name not in ("ne4",) else 200
        tr = attractor(s, s.known[key], T=T)
        if len(tr) > 80000:
            tr = tr[:: len(tr) // 80000 + 1]
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=SERIES[0] if k % 2 == 0 else SERIES[6], lw=0.3, alpha=0.85)
        for e in s.equilibria[:3]:
            ax.scatter(*e, color=SERIES[7], marker="x", s=30, depthshade=False)
        if name == "le1":
            zz = np.linspace(tr[:, 2].min(), tr[:, 2].max(), 2)
            ax.plot([0, 0], [0, 0], zz, color=SERIES[7], lw=1.2)
        style3d(ax)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Скрытые аттракторы эталонного набора (× — равновесия; у LE1 — линия равновесий)", fontsize=11)
    fig.tight_layout()
    save(fig, FIG / "fig32_gallery_3d")


def fig_li_sprott():
    s = systems.get("li_sprott_4d_a6")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
    for name, c, lab in (("T", SERIES[0], "тор"), ("SA_a", SERIES[1], "странный аттрактор A"), ("SA_b", SERIES[2], "странный аттрактор B")):
        tr = attractor(s, s.known[name], T=300)
        axes[0].plot(tr[:, 0], tr[:, 1], color=c, lw=0.2, label=lab)
        axes[1].plot(tr[:, 3], tr[:, 0], color=c, lw=0.2, label=lab)
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].set_title("проекция (x, y)", fontsize=9)
    axes[1].set_xlabel("u"); axes[1].set_ylabel("x"); axes[1].set_title("проекция (u, x)", fontsize=9)
    axes[0].legend(fontsize=7, markerscale=10)
    from e20_search_benchmark import build_problem
    s2, refs, tidx, kidx = build_problem("li_sprott_4d_a6", np.random.default_rng(100))
    names = [r.name for r in refs]
    cols = [{"T": SERIES[0], "SA_a": SERIES[1], "SA_b": SERIES[2]}.get(n, SERIES[5]) for n in names]
    a, b, img = basin_grid(s, refs, (0, 2, -15, 15, -15, 15), [0, 0, 0, 0], n1=220, n2=220)
    draw_basin(axes[2], a, b, img, names, cols)
    axes[2].set_xlabel("x(0)"); axes[2].set_ylabel("z(0)")
    axes[2].set_title("бассейны в сечении y = u = 0", fontsize=9)
    fig.suptitle("4D упрощённая система Лоренца (Li & Sprott 2014), a = 6, b = 0.1: равновесий нет, все аттракторы скрыты", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "fig33_li_sprott_4d")


def fig_maps():
    cases = [("huynh2019", "Хуинь и др. (2019)", (-3, 3, -3, 3)), ("jiang_nfe", "Jiang NF I e", (-8, 8, -8, 8)),
             ("jiang_nff", "Jiang NF I f: хаос + цикл периода 2", (-5, 5, -5, 5)), ("jiang_sfa", "Jiang SF I a: хаос + устойчивая точка", (-5, 5, -5, 5))]
    from e20_search_benchmark import build_problem
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    for ax, (name, title, box) in zip(axes, cases):
        s, refs, tidx, kidx = build_problem(name, np.random.default_rng(100))
        names = [r.name for r in refs]
        cols = []
        for n in names:
            cols.append(SERIES[0] if n == "HA" else (SERIES[1] if n in ("FP", "E0") else SERIES[3]))
        a, b, img = basin_grid(s, refs, (0, 1, box[0], box[1], box[2], box[3]), [0, 0], n1=360, n2=360)
        draw_basin(ax, a, b, img, names, cols)
        tr = attractor(s, s.known["HA"], T=20000)
        ax.plot(tr[:, 0], tr[:, 1], ",", color=INK, alpha=0.6)
        for r in refs:
            if r.name.startswith("other") and r.example is not None:
                orb = attractor(s, r.example, T=40)
                ax.plot(orb[:, 0], orb[:, 1], "o", color=SERIES[7], ms=4)
        for e in s.equilibria:
            ax.plot(e[0], e[1], "o", color=SERIES[1], ms=5, mec=INK)
        ax.set_title(title, fontsize=9); ax.set_xlabel("x"); ax.set_ylabel("y")
    axes[0].legend(handles=[Patch(color=ESCAPE_GREY, label="уход"), Patch(color=SERIES[0], label="бассейн хаоса"),
                            Patch(color=SERIES[3], label="бассейн цикла"), Patch(color=SERIES[1], label="бассейн точки")],
                   fontsize=7, loc="lower left", frameon=True)
    fig.suptitle("Отображения без неподвижных точек или с единственной устойчивой: фазово-бассейновые портреты (чёрное — аттрактор)", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "fig34_maps_phase_basin")


def fig_jafari():
    fig = plt.figure(figsize=(12, 4.2))
    s = systems.get("jafari2026_lc")
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    tr = attractor(s, s.known["LC"], T=60)
    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=SERIES[0], lw=1.0)
    style3d(ax); ax.set_title("a₃ = −1: устойчивый цикл", fontsize=9)
    s2 = systems.get("jafari2026_chaos")
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    tr = attractor(s2, s2.known["HA"], T=400, skip=0)
    ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=SERIES[1], lw=0.25)
    style3d(ax); ax.set_title("a₃ = −3.548: хрупкий хаос (λ ≈ 0.05)", fontsize=9)
    ax = fig.add_subplot(1, 3, 3)
    p = RES / "e12_jafari2026_scan.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = [r for r in d["rows"] if r["a3"] <= -3.50]
        ax.plot([r["a3"] for r in rows], [r["lam"] for r in rows], ".-", color=SERIES[1], ms=3, lw=0.8)
        esc = [r for r in d["rows"] if r["esc"]]
        if esc:
            ax.axvline(esc[0]["a3"], color=SERIES[7], lw=1, ls="--", label=f"уход при a₃ ≤ {esc[0]['a3']:.4f}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.invert_xaxis(); ax.set_xlabel("a₃"); ax.set_ylabel("λ (окно T = 400)")
        ax.set_title("окно хаоса у границы ограниченности", fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("Система Jafari et al. (2026) без неподвижных и перпетуальных точек: ż = 6z − xy + a₃yz − zx − 1", fontsize=10)
    fig.tight_layout()
    pos = ax.get_position()
    ax.set_position([pos.x0 + 0.045, pos.y0, pos.width - 0.045, pos.height])
    save(fig, FIG / "fig35_jafari2026")


def fig_lorenz_transient():
    s = systems.get("lorenz_transient")
    dt = 0.005
    T = 20000
    n = int(T / dt)
    a_, r_, b_ = s.params[0], s.params[1], s.params[2]
    c = np.sqrt(b_ * (r_ - 1.0))
    C = np.array([[c, c, r_ - 1.0], [-c, -c, r_ - 1.0]])
    # начальное условие, у которого переходный хаос заканчивается внутри окна (часть копий у (5, 5, 5))
    rng = np.random.default_rng(36)
    X0 = np.array([5.0, 5.0, 5.0]) + 0.5 * rng.normal(size=(64, 3))
    feats = np.zeros((64, nfeat(3))); fin = np.zeros((64, 3))
    s.batch(X0, s.params, dt, n, 2000, s.esc_r, 20, feats, fin)
    dist_end = np.min(np.linalg.norm(fin[:, None, :] - C[None, :, :], axis=2), axis=1)
    pick = None
    for i in np.where(dist_end < 1e-2)[0]:
        tr = s.trajectory(X0[i], s.params, dt, n, 1e6)
        d = np.min(np.linalg.norm(tr[:, None, :] - C[None, :, :], axis=2), axis=1)
        t_c = (np.where(d > 2.0)[0].max() + 1) * dt
        if 3000 < t_c < 15000:
            pick = (X0[i], tr, t_c)
            break
    if pick is None:
        x0 = X0[int(np.argmin(dist_end))]
        tr = s.trajectory(x0, s.params, dt, n, 1e6)
        d = np.min(np.linalg.norm(tr[:, None, :] - C[None, :, :], axis=2), axis=1)
        pick = (x0, tr, (np.where(d > 2.0)[0].max() + 1) * dt)
    x0, tr, t_c = pick
    t = np.arange(len(tr)) * dt
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True)
    step = 20
    axes[0].plot(t[::step], tr[::step, 0], color=SERIES[0], lw=0.25)
    axes[0].axvline(t_c, color=SERIES[7], lw=1, ls="--")
    axes[0].text(t_c + 150, 10, f"t ≈ {t_c:.0f}: переход к устойчивому фокусу C±", fontsize=8, color=INK)
    axes[0].set_ylabel("x(t)")
    # конечновременной показатель по окнам T = 100
    win = int(100 / dt)
    lam, tw = [], []
    for k in range(0, n - win, win):
        f1 = np.zeros((1, nfeat(3))); e1 = np.zeros((1, 3))
        s.batch(tr[k][None, :], s.params, dt, 0, win, s.esc_r, 20, f1, e1)
        lam.append(f1[0, F_LAM]); tw.append(t[k] + 50)
    axes[1].plot(tw, lam, ".-", color=SERIES[1], ms=2, lw=0.8)
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].axvline(t_c, color=SERIES[7], lw=1, ls="--")
    axes[1].set_ylabel("λ по окнам T = 100"); axes[1].set_xlabel("t")
    fig.suptitle("Отрицательный контроль: переходный хаос системы Лоренца (a, r, b) = (4, 29, 2), "
                 f"начальное условие ({x0[0]:.2f}, {x0[1]:.2f}, {x0[2]:.2f})", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "fig36_lorenz_transient")


def _capture_sus(system_name, seed=1000, budget=4000, score="trans", n0=50):
    from e20_search_benchmark import build_problem
    from hsearch.search.harness import Problem, run_search
    from hsearch.search.sus import SubsetSim
    s, refs, tidx, kidx = build_problem(system_name, np.random.default_rng(100))
    rng = np.random.default_rng(seed)
    prob = Problem(s, refs, tidx, kidx, rng)
    st = SubsetSim(prob, n0=n0, score=score)
    levels = []
    orig = st._start_level

    def patched(X, XI):
        orig(X, XI)
        levels.append((st.n_restarts, st.n_level, st.chain_x.copy()))
    st._start_level = patched
    res = run_search(prob, st, budget)
    return s, refs, tidx, levels, res


def fig_d2_mechanism():
    from e20_search_benchmark import get_system
    name = "edge_se3_m3_p0.0001"
    s = get_system(name)
    rho = s.meta["rho"]
    rng = np.random.default_rng(37)
    ics = s.sample_box(30000, rng)
    feats, finals = run_batch(s, ics)
    w = np.linalg.norm(ics[:, 3:], axis=1) / rho
    esc = feats[:, F_ESC] > 0.5
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    axes[0].loglog(w[esc] - 1, feats[esc, F_TESC] * s.dt, ",", color=SERIES[0], alpha=0.5)
    dd = np.logspace(-6, 1.5, 100)
    axes[0].loglog(dd, np.maximum(np.log(1 / (2 * dd)), 1e-3) / 2.0, color=SERIES[7], lw=1.2, label="ln(1/2Δ) / 2κ")
    axes[0].set_xlabel("Δ = |w₀|/ρ − 1"); axes[0].set_ylabel("время ухода")
    axes[0].set_title("Время ухода расходится у границы бассейна", fontsize=9); axes[0].legend(fontsize=8)
    s2, refs, tidx, levels, res = _capture_sus(name, budget=6000)
    for k, (rst, lv, X) in enumerate(levels[:12]):
        ww = np.linalg.norm(X[:, 3:], axis=1) / rho
        axes[1].scatter(np.full(len(ww), k) + 0.15 * rng.normal(size=len(ww)), ww, s=10, color=SERIES[k % 8], alpha=0.8)
    axes[1].axhline(1.0, color=INK, lw=1, ls="--", label="граница бассейна |w| = ρ")
    axes[1].set_yscale("log"); axes[1].set_xlabel("номер уровня (с учётом перезапусков)")
    axes[1].set_ylabel("|w| / ρ у цепей уровня")
    axes[1].set_title(f"Цепи расщепления стягиваются к шару; цель найдена на {res['found_at']}-й оценке", fontsize=9)
    axes[1].legend(fontsize=8)
    fig.suptitle("Семейство D2 (n = 6, p = 10⁻⁴): механизм выигрыша расщепления", fontsize=10)
    fig.tight_layout()
    save(fig, FIG / "fig37_d2_mechanism")


def fig_sus_chains():
    name = "jiang_nfe"
    s, refs, tidx, levels, res = _capture_sus(name, budget=3000)
    names = [r.name for r in refs]
    cols = [SERIES[0] if n == "HA" else SERIES[3] for n in names]
    a, b, img = basin_grid(s, refs, (0, 1, -8, 8, -8, 8), [0, 0], n1=360, n2=360)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    draw_basin(ax, a, b, img, names, cols)
    for k, (rst, lv, X) in enumerate(levels[:8]):
        ax.scatter(X[:, 0], X[:, 1], s=14, color=SERIES[(k + 1) % 8], edgecolors=INK, linewidths=0.3, label=f"уровень {k}")
    ax.set_xlabel("x(0)"); ax.set_ylabel("y(0)")
    ax.set_title(f"Цепи расщепления по уровням на карте бассейнов (Jiang NF I e); цель на {res['found_at']}-й оценке", fontsize=9)
    ax.legend(fontsize=7, loc="upper right", frameon=True)
    fig.tight_layout()
    save(fig, FIG / "fig38_sus_chains")


def fig_hiddenness():
    p = RES / "e11_hiddenness.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    targets = {"chua_I": ["HA_a", "HA_b"], "chua_II": ["new0", "new1"], "wangchen2": ["HA"], "se3": ["HA"], "se11": ["HA"],
               "rabinovich_100": ["HA"], "lorenz_transient": ["TC"]}
    label = {"chua_I": "Чуа I (оба скрытых)", "chua_II": "Чуа II (цикл + хаос)", "wangchen2": "Ванг–Чен #2", "se3": "SE3",
             "se11": "SE11", "rabinovich_100": "Рабинович r = 100", "lorenz_transient": "Лоренц (4, 29, 2), переходное"}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for k, (name, tg) in enumerate(targets.items()):
        if name not in d:
            continue
        eps = sorted(float(e) for e in d[name]["eps"])
        h = [sum(d[name]["eps"][str(e) if str(e) in d[name]["eps"] else f"{e:g}"]["fractions"].get(t, 0.0) for t in tg) for e in eps]
        ax.plot(eps, np.maximum(h, 1e-4), "o-", color=SERIES[k % 8], label=label[name])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(8e-5, 1.2)
    ax.set_xlabel("ε / L (радиус окрестности равновесий)"); ax.set_ylabel("h_ε: доля, попадающая в аттрактор")
    ax.set_title("Мера скрытости: нуль при малых ε у скрытых аттракторов", fontsize=10)
    ax.text(1.4e-2, 1.25e-4, "нижняя граница оси — «ни одной точки»", fontsize=7, color=INK2)
    ax.legend(fontsize=7)
    save(fig, FIG / "fig40_hiddenness")


from hsearch.viz import nice_label  # noqa: E402


def fig_verification():
    p = RES / "e13_verify_targets.json"
    q = RES / "e13b_long_lifetime.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for name, rec in d.items():
        for t, v in rec.items():
            if isinstance(v, dict) and v.get("lam_windows"):
                rows.append((nice_label(name, t), v["lam_windows"], v.get("kind")))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.28 * len(rows) + 1.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for k, (lab, lw, kind) in enumerate(rows):
        axes[0].scatter(lw, np.full(len(lw), k), s=9, color=SERIES[0] if kind == "chaotic" else SERIES[1])
    axes[0].axvline(0, color=MUTED, lw=0.8)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([r[0] for r in rows], fontsize=7)
    axes[0].set_xlabel("λ по окнам наблюдения"); axes[0].set_title("Показатель Ляпунова по окнам (синие — хаос, оранжевые — циклы/тор)", fontsize=9)
    axes[0].grid(axis="y", visible=False)
    if q.exists():
        e = json.loads(q.read_text(encoding="utf-8"))
        lab = {"lorenz_transient": "Лоренц (4, 29, 2)", "rabinovich_68": "Рабинович (6.8, −0.5, 0.99)", "jafari2026_chaos": "хаос Jafari 2026"}
        for k, (name, rec) in enumerate(e.items()):
            te = np.array([np.inf if v is None else v for v in rec["t_end"]])
            tt = np.logspace(2, 5, 60)
            alive = [(te > t).mean() for t in tt]
            axes[1].step(tt, alive, where="post", color=SERIES[k + 1], label=lab.get(name, name))
        axes[1].set_xscale("log"); axes[1].set_ylim(-0.02, 1.05)
        axes[1].set_xlabel("t"); axes[1].set_ylabel("доля копий на множестве")
        axes[1].set_title("Длинная проверка времени жизни (шум 10⁻³)", fontsize=9); axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, FIG / "fig41_verification")


def main(which=None):
    viz.setup()
    funcs = [fig_chua_I, fig_chua_II, fig_gallery, fig_li_sprott, fig_maps, fig_jafari, fig_lorenz_transient,
             fig_d2_mechanism, fig_sus_chains, fig_hiddenness, fig_verification]
    for f in funcs:
        if which and f.__name__ not in which:
            continue
        try:
            f()
            print("ok:", f.__name__, flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print("skip:", f.__name__, repr(e), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
