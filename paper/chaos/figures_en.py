"""English-only redraws of article figures (Chaos submission); used by make_figures_en.py.

The Russian figure scripts (experiments/e10, e26, e40) stay unchanged. Where the English figure differs in content
from the Russian one (panels dropped or merged, legends moved outside the data, panel labels instead of titles,
grey-scale-safe styles), the whole figure function is redrawn here. Each function receives the namespace of the
executed script (its helpers, data loaders and colours) and replaces the function of the same name (OVERRIDE);
EXTRA adds figures without a Russian original. Figure sizes come from FIGSIZE in make_figures_en.py.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.transforms import blended_transform_factory, offset_copy

from hsearch import viz
from hsearch.viz import ESCAPE_GREY, INK, INK2, MUTED, SERIES, panel_label

# basin maps: light grey = escape, dark blue = chaos, white = cycle, orange = fixed point (distinct in grey scale)
CHAOS_BLUE, CYCLE_WHITE, POINT_ORANGE = "#1c5cab", "#ffffff", SERIES[1]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _ticks3d(ax, n=4):
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(n))


def _legend_below(ax, dy=-0.18, ncol=2, **kw):
    kw.setdefault("columnspacing", 1.0)
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, dy), ncol=ncol, **kw)


# ============================================================================================ e10
def fig_chua_transient(ns):
    """fig02_chua_transient: (a) survival of the chaotic set (fig02) and (b) basin fractions vs the step (fig03)."""
    d = ns["load"]("e03_chua_lifetime.json")
    e = ns["load"]("e04_chua_dt_convergence.json")
    if d is None or e is None:
        return
    fig, axes = plt.subplots(2, 1)
    ax = axes[0]
    for i, key in enumerate(("survival_dt0.01", "survival_dt0.0025")):
        r = d.get(key)
        if r:
            ax.plot(r["T"], r["alive"], color=SERIES[i], ls="-" if i == 0 else "--",
                    label=f"dt = {r['dt']}, {r['n_copies']} copies")
    ax.set_xlabel("t (time units)"); ax.set_ylabel("fraction of copies still chaotic")
    ax.set_ylim(0, 1.05)
    ax.text(0.32, 0.22, "DOP853, rtol = 10⁻¹¹:\nchaos persists up to t = 4000", transform=ax.transAxes, color=INK2)
    ax.legend(loc="center right")
    panel_label(ax, "(a)")
    ax = axes[1]
    runs = e["runs"]
    dts = sorted(float(k) for k in runs)
    x = np.array(dts) * 1e3
    series = [("inf", "escape to infinity", MUTED, "-"), ("F0", "equilibrium F0", ns["color_for"]("F0"), "-"),
              ("HA_a", "hidden attractor A", ns["color_for"]("HA_a"), "-"),
              ("HA_b", "hidden attractor B", ns["color_for"]("HA_b"), "--"), ("unknown", "unmatched", INK, "-")]
    for key, lab, c, ls in series:
        y = [runs[str(dt)][key]["p"] for dt in dts]
        lo = [runs[str(dt)][key]["ci95"][0] for dt in dts]
        hi = [runs[str(dt)][key]["ci95"][1] for dt in dts]
        ax.plot(x, y, marker="o", color=c, ls=ls, label=lab, zorder=3 if ls == "--" else 2)
        ax.fill_between(x, lo, hi, color=c, alpha=0.15, lw=0)
    ax.set_xticks(x); ax.set_xticklabels([f"{v:g}" for v in x])
    ax.set_xlabel("integration step dt (RK4), ×10⁻³ time units"); ax.set_ylabel("fraction of initial conditions (ball r = 3)")
    ax.set_ylim(-0.01, 0.5)
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.62), ncol=2, columnspacing=1.0)
    panel_label(ax, "(b)")
    fig.tight_layout()
    viz.save(fig, ns["FIG"] / "fig02_chua_transient")


# ============================================================================================ e26
def fig_scaling_p(ns):
    """fig23: no 'splitting: escape time' curve (identical to transient length), one shared legend below the panels,
    the fitted exponents s as a text block in each panel, panel labels."""
    LABEL, COLOR, q09, slope_ci, _sweep = ns["LABEL"], ns["COLOR"], ns["q09"], ns["slope_ci"], ns["_sweep"]
    ps = [1e-2, 1e-3, 1e-4, 1e-5]
    short = {"uniform": "uniform", "sus_rarity": "splitting (rarity)", "sus": "splitting (transient)",
             "portfolio_no_sus": "active learning", "portfolio": "portfolio"}
    fig, axes = plt.subplots(1, 2, sharey=True)
    fits = {}
    for ax, lab, prefix, title, show, lo_exp in (
            (axes[0], "(a)", "edge", "D2: edge state (graded signal)",
             ["uniform", "sus_rarity", "sus", "portfolio_no_sus", "portfolio"], -5.3),
            (axes[1], "(b)", "emb", "D1: gate (no signal)", ["uniform", "sus", "portfolio_no_sus", "portfolio"], -4.3)):
        data = _sweep(prefix, 3, ps)
        pp = np.logspace(lo_exp, -1.7, 50)
        ax.plot(pp, np.log(10) / pp, color=MUTED, lw=0.9, ls="-", label="ln 10 / p (uniform, theory)", zorder=1)
        lines = []
        for sname in show:
            xs, ys, series = [], [], []
            for p, d in sorted(data.items()):
                if sname in d:
                    v = q09(d[sname]["found_at"])
                    series.append((p, d[sname]["found_at"]))
                    if np.isfinite(v):
                        xs.append(p); ys.append(v)
            if len(series) >= 3:
                s_fit, lo, hi = slope_ci(series)
                fits[(prefix, sname)] = {"s": s_fit, "ci95": [lo, hi], "n_points": len(series)}
                lines.append((f"{short[sname]}: s = {s_fit:.2f} [{lo:.2f}, {hi:.2f}]", COLOR[sname]))
            if xs:
                ax.plot(xs, ys, "o-", color=COLOR[sname], label=LABEL[sname])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("target basin fraction p")
        panel_label(ax, lab, title)
        for i, (text, c) in enumerate(lines):
            ax.text(0.03, 0.97, text, transform=offset_copy(ax.transAxes, fig=fig, x=0, y=-10.5 * i, units="points"),
                    va="top", ha="left", color=INK if c == MUTED else c)
    axes[0].set_ylabel("$N_{0.9}$ (integrations)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5, handlelength=2.4)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    viz.save(fig, ns["FIG"] / "fig23_scaling_p")
    (ns["RES"] / "e26_scaling_fits.json").write_text(
        json.dumps({f"{a}:{b}": v for (a, b), v in fits.items()}, ensure_ascii=False, indent=1), encoding="utf-8")
    return {k: round(v["s"], 2) for k, v in fits.items()}


def fig_signal(ns):
    """fig26: only the median-distance panel (the gain bars repeat the legend); solid lines for systems with a
    splitting gain (G > 1), dashed without; legend below the axes."""
    load, q09, RES = ns["load"], ns["q09"], ns["RES"]
    systems = [("chua_II", "Chua II"), ("le1", "LE1"), ("se11", "SE11"), ("edge_se3_m3_p0.001", "D2, p = 10⁻³"),
               ("huynh2019", "Huynh"), ("jafari2026_lc", "Jafari cycle"), ("emb_se3_m3_p0.001", "D1, p = 10⁻³")]
    fig, ax = plt.subplots()
    for i, (name, lab) in enumerate(systems):
        p = RES / f"e22_signal_{name}.json"
        d = load(name)
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        cur = [c for c in rec.get("level_curve", []) if c["median_dist_rel"] is not None]
        if not cur:
            continue
        g = None
        if d and "uniform" in d and "sus" in d:
            nu, nsu = q09(d["uniform"]["found_at"]), q09(d["sus"]["found_at"])
            g = nu / nsu if np.isfinite(nsu) and nsu > 0 else None
        gain = g is not None and g > 1
        ax.plot([c["q"] for c in cur], [c["median_dist_rel"] for c in cur], ls="-" if gain else "--",
                marker=MARKERS[i % len(MARKERS)], ms=4.5, lw=1.6 if gain else 1.2, color=SERIES[i % 8],
                label=f"{lab} (G = {g:.1f})" if g else lab)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("upper-set fraction q of the coordinate"); ax.set_ylabel("d(q), relative to q = 1")
    _legend_below(ax, dy=-0.3, ncol=2, handlelength=2.6)
    fig.tight_layout()
    viz.save(fig, ns["FIG"] / "fig26_signal_curves")


# ============================================================================================ e40
def fig_chua_I(ns):
    """fig30: legends below both panels, separate legend entries for attractors A and B, panel labels."""
    systems, attractor, style3d, draw_basin, RES = ns["systems"], ns["attractor"], ns["style3d"], ns["draw_basin"], ns["RES"]
    s = systems.get("chua_I")
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for name, c, lab in (("HA_a", SERIES[0], "hidden attractor A"), ("HA_b", SERIES[2], "hidden attractor B")):
        tr = attractor(s, s.known[name], T=300)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=0.35)
        ax.plot([], [], color=c, label=lab)
    eq = s.equilibria
    ax.scatter(*eq[0], color=SERIES[1], s=30, depthshade=False, label="F0 (stable focus-node)")
    for e in eq[1:]:
        ax.scatter(*e, color=SERIES[7], marker="x", s=40, depthshade=False)
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
    ax.scatter([], [], color=SERIES[7], marker="x", label="F± (saddle-foci)")
    ax.plot([], [], color=MUTED, ls="--", label="unstable manifolds of F±")
    ax.set_xlim(-12, 12); ax.set_ylim(-4, 4); ax.set_zlim(-15, 15)
    style3d(ax); _ticks3d(ax); ax.view_init(elev=18, azim=-60)
    ax.set_box_aspect(None, zoom=1.15)
    panel_label(ax, "(a)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handlelength=1.6)
    ax2 = fig.add_subplot(1, 2, 2)
    p = RES / "e01_chua_basin_z0_labels.npy"
    if p.exists():
        img = np.load(p)
        d = json.loads((RES / "e01_chua_table1.json").read_text(encoding="utf-8"))["basin_z0"]
        a = np.linspace(d["x"][0], d["x"][1], img.shape[1]); b = np.linspace(d["y"][0], d["y"][1], img.shape[0])
        draw_basin(ax2, a, b, img, ["F0", "Fp", "Fm", "HA_a", "HA_b"], [SERIES[1], SERIES[7], SERIES[6], SERIES[0], SERIES[2]])
        for name, c in (("HA_a", "#0d366b"), ("HA_b", "#0d4f3a")):
            tr = attractor(s, s.known[name], T=200)
            sel = np.abs(tr[:, 2]) < 0.15
            ax2.plot(tr[sel, 0], tr[sel, 1], ".", ms=2.0, color=c)
        ax2.set_xlabel("x(0)"); ax2.set_ylabel("y(0)")
        panel_label(ax2, "(b)")
        ax2.legend(handles=[Patch(color=ESCAPE_GREY, label="escape"), Patch(color=SERIES[1], label="F0"),
                            Patch(color=SERIES[0], label="attractor A"), Patch(color=SERIES[2], label="attractor B")],
                   loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, columnspacing=1.0, handlelength=1.4)
    fig.tight_layout()
    _p = ax2.get_position(); ax2.set_position([_p.x0 + 0.04, _p.y0, _p.width - 0.04, _p.height])
    viz.save(fig, ns["FIG"] / "fig30_chua_I_hidden")


def fig_chua_II(ns):
    """fig31: legends below the panels (they covered the x label and the basin map), panel labels, no titles."""
    systems, attractor, style3d, draw_basin, basin_grid, RES = (ns["systems"], ns["attractor"], ns["style3d"],
                                                               ns["draw_basin"], ns["basin_grid"], ns["RES"])
    s = systems.get("chua_II")
    eq = s.equilibria
    trs = {"LC": attractor(s, s.known["LC"], T=12), "HA_a": attractor(s, s.known["HA_a"], T=150),
           "HA_b": attractor(s, s.known["HA_b"], T=150)}
    # explicit layout: tight_layout cannot place 3D axes together with legends outside the axes
    fig = plt.figure()
    boxes3d = ([0.00, 0.20, 0.37, 0.80], [0.33, 0.20, 0.37, 0.80])
    for k, zoom in enumerate((False, True)):
        ax = fig.add_axes(boxes3d[k], projection="3d")
        if not zoom:
            ax.plot(trs["LC"][:, 0], trs["LC"][:, 1], trs["LC"][:, 2], color=SERIES[6], lw=0.9, label="limit cycle LC (0.992)")
        for name, c, lab in (("HA_a", SERIES[0], "chaos A (0.0014)"), ("HA_b", SERIES[2], "chaos B (0.0013)")):
            tr = trs[name]
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=0.35 if zoom else 0.8, label=lab)
        ax.scatter(*eq[0], color=SERIES[7], marker="x", s=40, depthshade=False, label="F$_0$ (saddle-focus)")
        for e in eq[1:]:
            ax.scatter(*e, color=SERIES[1], s=30, depthshade=False)
        ax.scatter([], [], color=SERIES[1], label="F$_\\pm$ (stable)")
        style3d(ax); _ticks3d(ax); ax.view_init(elev=15, azim=-55)
        ax.set_zlabel("z", labelpad=-2)
        ax.set_box_aspect(None, zoom=0.92)
        panel_label(ax, "(b)" if zoom else "(a)")
        if not zoom:
            handles, labels = ax.get_legend_handles_labels()
    lg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.36, 0.0), ncol=3, columnspacing=1.0,
                    handlelength=1.6, frameon=False)
    for ln in lg.get_lines():
        ln.set_linewidth(1.6)
    ax2 = fig.add_axes([0.815, 0.34, 0.175, 0.60])
    from e20_search_benchmark import build_problem
    s2, refs, tidx, kidx = build_problem("chua_II", np.random.default_rng(100))
    names = [r.name for r in refs]
    colors = [{"E0": SERIES[7], "E1": SERIES[1], "E2": "#b85a2a", "LC": SERIES[6], "HA_a": SERIES[0], "HA_b": SERIES[2]}.get(n, SERIES[5])
              for n in names]
    cache = [p for p in (RES / "e40_chua_II_basin_z0.npy", ns["ROOT"] / "results" / "e40_chua_II_basin_z0.npy") if p.exists()]
    a, b = np.linspace(-3, 3, 300), np.linspace(-1.5, 1.5, 200)
    img = np.load(cache[0]) if cache else None
    if img is None or img.shape != (200, 300):
        a, b, img = basin_grid(s, refs, (0, 1, -3, 3, -1.5, 1.5), [0, 0, 0], n1=300, n2=200)
        np.save(RES / "e40_chua_II_basin_z0.npy", img)
    draw_basin(ax2, a, b, img, names, colors)
    ax2.set_xlabel("x(0)"); ax2.set_ylabel("y(0)", labelpad=2)
    panel_label(ax2, "(c)")
    fig.legend(handles=[Patch(color=SERIES[6], label="cycle LC"), Patch(color=SERIES[1], label="F+"),
                        Patch(color="#b85a2a", label="F−"), Patch(color=SERIES[0], label="chaos A"),
                        Patch(color=SERIES[2], label="chaos B")],
               loc="lower center", bbox_to_anchor=(0.87, 0.0), ncol=2, columnspacing=0.8, handlelength=1.2, frameon=False)
    viz.save(fig, ns["FIG"] / "fig31_chua_II_coexisting")


def fig_li_sprott(ns):
    """fig33: the (x, y) projection is dropped; (a) projection (u, x), (b) basin section; B in a dark colour."""
    systems, attractor, draw_basin, basin_grid = ns["systems"], ns["attractor"], ns["draw_basin"], ns["basin_grid"]
    s = systems.get("li_sprott_4d_a6")
    col = {"T": "#6da7ec", "SA_a": SERIES[1], "SA_b": SERIES[6]}
    fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})
    for name, lab in (("T", "torus"), ("SA_a", "strange attractor A"), ("SA_b", "strange attractor B")):
        tr = attractor(s, s.known[name], T=300)
        axes[0].plot(tr[:, 3], tr[:, 0], color=col[name], lw=0.2, label=lab)
    axes[0].set_xlabel("u"); axes[0].set_ylabel("x")
    panel_label(axes[0], "(a)")
    from e20_search_benchmark import build_problem
    s2, refs, tidx, kidx = build_problem("li_sprott_4d_a6", np.random.default_rng(100))
    names = [r.name for r in refs]
    a, b, img = basin_grid(s, refs, (0, 2, -15, 15, -15, 15), [0, 0, 0, 0], n1=220, n2=220)
    draw_basin(axes[1], a, b, img, names, [col.get(n, SERIES[5]) for n in names])
    axes[1].set_xlabel("x(0)"); axes[1].set_ylabel("z(0)")
    panel_label(axes[1], "(b)", "section y = u = 0")
    lg = fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    for ln in lg.get_lines():
        ln.set_linewidth(2.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    viz.save(fig, ns["FIG"] / "fig33_li_sprott_4d")


def fig_maps(ns):
    """fig34: grey-scale-safe basin palette, black markers for the cycle and the fixed point, one legend below the
    panels, equal axes, panel labels with the system names."""
    attractor, draw_basin, basin_grid = ns["attractor"], ns["draw_basin"], ns["basin_grid"]
    from e20_search_benchmark import build_problem
    cases = [("huynh2019", "(a)", "Huynh et al. (2019)", (-3, 3, -3, 3)), ("jiang_nfe", "(b)", "Jiang NF I e", (-8, 8, -8, 8)),
             ("jiang_nff", "(c)", "Jiang NF I f:\nchaos + period-2 cycle", (-5, 5, -5, 5)),
             ("jiang_sfa", "(d)", "Jiang SF I a:\nchaos + stable point", (-5, 5, -5, 5))]
    fig, axes = plt.subplots(1, 4)
    for ax, (name, lab, title, box) in zip(axes, cases):
        s, refs, tidx, kidx = build_problem(name, np.random.default_rng(100))
        names = [r.name for r in refs]
        cols = [CHAOS_BLUE if n == "HA" else (POINT_ORANGE if n in ("FP", "E0") else CYCLE_WHITE) for n in names]
        a, b, img = basin_grid(s, refs, (0, 1, box[0], box[1], box[2], box[3]), [0, 0], n1=360, n2=360)
        draw_basin(ax, a, b, img, names, cols)
        ax.set_aspect("equal")
        for sp in ax.spines.values():
            sp.set_visible(True)
        tr = attractor(s, s.known["HA"], T=20000)
        ax.plot(tr[:, 0], tr[:, 1], ",", color=INK)
        for r in refs:
            if r.name.startswith("other") and r.example is not None:
                orb = attractor(s, r.example, T=40)
                ax.plot(orb[:, 0], orb[:, 1], "o", color=INK, mec="white", mew=0.5, ms=4.5)
        for e in s.equilibria:
            ax.plot(e[0], e[1], "s", color=INK, mec="white", mew=0.5, ms=5)
        panel_label(ax, lab, title, fontsize=8.5)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    handles = [Patch(color=ESCAPE_GREY, label="escape"), Patch(color=CHAOS_BLUE, label="chaos basin"),
               Patch(facecolor=CYCLE_WHITE, edgecolor=INK2, lw=0.6, label="cycle basin"), Patch(color=POINT_ORANGE, label="point basin"),
               Line2D([], [], ls="", marker="s", ms=2.5, color=INK, label="chaotic attractor"),
               Line2D([], [], ls="", marker="o", ms=4.5, color=INK, mec="white", label="period-2 cycle"),
               Line2D([], [], ls="", marker="s", ms=5, color=INK, mec="white", label="stable fixed point")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2, handlelength=1.4)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    viz.save(fig, ns["FIG"] / "fig34_maps_phase_basin")


def fig_jafari(ns):
    """fig35: only the λ(a₃) panel as a column-width figure (the cycle and the chaotic attractor are in the gallery)."""
    p = ns["RES"] / "e12_jafari2026_scan.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    fig, ax = plt.subplots()
    rows = [r for r in d["rows"] if r["a3"] <= -3.50]
    ax.plot([r["a3"] for r in rows], [r["lam"] for r in rows], ".-", color=SERIES[1], ms=3, lw=0.8)
    esc = [r for r in d["rows"] if r["esc"]]
    if esc:
        ax.axvline(esc[0]["a3"], color=SERIES[7], lw=1, ls="--")
        ax.text(0.5, 0.6, f"dashed: escape to infinity\nfor a₃ ≤ {esc[0]['a3']:.4f}".replace("-", "−"),
                transform=ax.transAxes, ha="center", va="center", color=INK2)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("a₃ (axis reversed: a₃ decreases to the right)"); ax.set_ylabel("λ (window T = 400)")
    fig.tight_layout()
    viz.save(fig, ns["FIG"] / "fig35_jafari2026")


def fig_d2_mechanism(ns):
    """fig37: (a) grey points under the exact escape-time law t = ln[(1+Δ)²/((1+Δ)²−1)]/2κ over the whole range;
    (b) one colour for all levels, ticks 0.5/1/2/5, the level where the target is found is marked."""
    from e20_search_benchmark import get_system
    run_batch, F_ESC, F_TESC, _capture_sus = ns["run_batch"], ns["F_ESC"], ns["F_TESC"], ns["_capture_sus"]
    name = "edge_se3_m3_p0.0001"
    s = get_system(name)
    rho = s.meta["rho"]
    rng = np.random.default_rng(37)
    ics = s.sample_box(30000, rng)
    feats, finals = run_batch(s, ics)
    w = np.linalg.norm(ics[:, 3:], axis=1) / rho
    esc = feats[:, F_ESC] > 0.5
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    ax.loglog(w[esc] - 1, feats[esc, F_TESC] * s.dt, ".", color=MUTED, ms=2.0, alpha=0.5, mew=0, zorder=1,
              label="3·10⁴ uniform initial conditions")
    dd = np.logspace(-6, 1.6, 300)
    ax.loglog(dd, np.log((1 + dd) ** 2 / ((1 + dd) ** 2 - 1)) / 2.0, color=INK, lw=0.9, zorder=3,
              label="t = ln[(1+Δ)²/((1+Δ)²−1)] / 2κ")
    ax.set_xlabel("Δ = |w₀|/ρ − 1"); ax.set_ylabel("escape time $t_\\mathrm{esc}$ (time units)")
    panel_label(ax, "(a)")
    ax.legend(loc="lower left", markerscale=3)
    ax = axes[1]
    s2, refs, tidx, levels, res = _capture_sus(name, budget=6000)
    levels = levels[:12]
    for k, (rst, lv, X) in enumerate(levels):
        ww = np.linalg.norm(X[:, 3:], axis=1) / rho
        ax.scatter(np.full(len(ww), k) + 0.15 * rng.normal(size=len(ww)), ww, s=10, color=SERIES[0], alpha=0.8, lw=0)
    ax.axhline(1.0, color=INK, lw=1, ls="--", label="basin boundary |w| = ρ")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5]); ax.set_yticklabels(["0.5", "1", "2", "5"]); ax.yaxis.set_minor_locator(NullLocator())
    k_last = len(levels) - 1
    ax.annotate(f"target found\nat evaluation {res['found_at']}", xy=(k_last, 1.0), xytext=(k_last - 0.3, 2.6),
                ha="right", va="bottom", color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.set_xlabel("level index (including restarts)"); ax.set_ylabel("|w|/ρ of the level chains")
    panel_label(ax, "(b)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    viz.save(fig, ns["FIG"] / "fig37_d2_mechanism")


def fig_sus_chains(ns):
    """fig38: level-0 seeds as white-filled circles, level-1 seeds as black triangles; legend below; no title."""
    _capture_sus, basin_grid, draw_basin = ns["_capture_sus"], ns["basin_grid"], ns["draw_basin"]
    s, refs, tidx, levels, res = _capture_sus("jiang_nfe", budget=3000)
    names = [r.name for r in refs]
    cols = [CHAOS_BLUE if n == "HA" else SERIES[3] for n in names]
    a, b, img = basin_grid(s, refs, (0, 1, -8, 8, -8, 8), [0, 0], n1=360, n2=360)
    fig, ax = plt.subplots()
    draw_basin(ax, a, b, img, names, cols)
    ax.set_aspect("equal")
    styles = [dict(marker="o", facecolor="white", edgecolor=INK), dict(marker="^", facecolor=INK, edgecolor=INK),
              dict(marker="s", facecolor=SERIES[3], edgecolor=INK), dict(marker="D", facecolor=SERIES[1], edgecolor=INK),
              dict(marker="v", facecolor=SERIES[4], edgecolor=INK), dict(marker="P", facecolor=SERIES[5], edgecolor=INK),
              dict(marker="X", facecolor=SERIES[7], edgecolor=INK), dict(marker="*", facecolor=SERIES[6], edgecolor=INK)]
    for k, (rst, lv, X) in enumerate(levels[:8]):
        ax.scatter(X[:, 0], X[:, 1], s=26, linewidths=0.6, label=f"level {k}", zorder=3, **styles[k])
    ax.set_xlabel("x(0)"); ax.set_ylabel("y(0)")
    _legend_below(ax, dy=-0.14, ncol=4)
    fig.tight_layout()
    viz.save(fig, ns["FIG"] / "fig38_sus_chains")


def fig_hiddenness(ns):
    """fig40: a marker per system; zero hits shown on a broken axis labelled 0 instead of the floor 10⁻⁴."""
    p = ns["RES"] / "e11_hiddenness.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    targets = {"chua_I": ["HA_a", "HA_b"], "chua_II": ["new0", "new1"], "wangchen2": ["HA"], "se3": ["HA"], "se11": ["HA"],
               "rabinovich_100": ["HA"], "lorenz_transient": ["TC"]}
    label = {"chua_I": "Chua I (both hidden)", "chua_II": "Chua II (cycle + chaos)", "wangchen2": "Wang–Chen #2", "se3": "SE3",
             "se11": "SE11", "rabinovich_100": "Rabinovich r = 100", "lorenz_transient": "Lorenz (4, 29, 2), transient"}
    zero = 1e-4   # position of the "0" row on the broken axis
    fig, ax = plt.subplots()
    for k, (name, tg) in enumerate(targets.items()):
        if name not in d:
            continue
        eps = sorted(float(e) for e in d[name]["eps"])
        h = [sum(d[name]["eps"][str(e) if str(e) in d[name]["eps"] else f"{e:g}"]["fractions"].get(t, 0.0) for t in tg) for e in eps]
        ax.plot(eps, np.maximum(h, zero), ls="-", marker=MARKERS[k % len(MARKERS)], ms=5, mfc="white", mew=1.2,
                color=SERIES[k % 8], label=label[name])
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(6e-5, 1.3)
    ax.set_yticks([zero, 1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(["0", "10⁻³", "10⁻²", "10⁻¹", "10⁰"])
    ax.yaxis.set_minor_locator(NullLocator())
    tr = blended_transform_factory(ax.transAxes, ax.transData)
    for yb in (2.6e-4, 3.8e-4):   # axis break between the "0" row and 10⁻³
        ax.plot([-0.015, 0.015], [yb / 1.12, yb * 1.12], color=INK2, lw=0.8, transform=tr, clip_on=False, zorder=5)
    ax.set_xlabel("ε/L (radius around equilibria)"); ax.set_ylabel("hiddenness measure $h_\\varepsilon$")
    _legend_below(ax, dy=-0.3, ncol=2, columnspacing=0.8)
    fig.tight_layout()
    viz.save(fig, ns["FIG"] / "fig40_hiddenness")


def fig_verification(ns):
    """fig41: panel labels instead of titles; regime kind encoded by marker and colour with a legend."""
    nice_label, RES = ns["nice_label"], ns["RES"]
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
    # the original figsize is passed on: FIGSIZE scales the height with the number of rows
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.28 * len(rows) + 1.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for k, (lab, lw, kind) in enumerate(rows):
        chaotic = kind == "chaotic"
        axes[0].scatter(lw, np.full(len(lw), k), s=9 if chaotic else 14, marker="o" if chaotic else "s",
                        color=SERIES[0] if chaotic else SERIES[1])
    axes[0].scatter([], [], s=9, marker="o", color=SERIES[0], label="chaotic")
    axes[0].scatter([], [], s=14, marker="s", color=SERIES[1], label="regular (cycle, torus)")
    axes[0].axvline(0, color=MUTED, lw=0.8)
    axes[0].set_yticks(range(len(rows))); axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].set_xlabel("λ over observation windows")
    axes[0].grid(axis="y", visible=False)
    axes[0].legend(loc="lower right")
    panel_label(axes[0], "(a)")
    if q.exists():
        e = json.loads(q.read_text(encoding="utf-8"))
        lab = {"lorenz_transient": "Lorenz", "rabinovich_68": "Rabinovich", "jafari2026_chaos": "Jafari chaos"}
        for k, ((name, rec), ls) in enumerate(zip(e.items(), ("-", "--", ":"))):
            te = np.array([np.inf if v is None else v for v in rec["t_end"]])
            tt = np.logspace(2, 5, 60)
            alive = [(te > t).mean() for t in tt]
            axes[1].step(tt, alive, where="post", color=SERIES[k + 1], ls=ls, label=lab.get(name, name))
        axes[1].set_xscale("log"); axes[1].set_ylim(-0.02, 1.05)
        axes[1].set_xlabel("t (time units)"); axes[1].set_ylabel("fraction of copies on the set")
        axes[1].legend(loc="center left", bbox_to_anchor=(0.12, 0.55), handlelength=1.6)
    panel_label(axes[1], "(b)")
    fig.tight_layout()
    viz.save(fig, ns["FIG"] / "fig41_verification")


OVERRIDE = {
    "e26:fig_scaling_p": fig_scaling_p, "e26:fig_signal": fig_signal,
    "e40:fig_chua_I": fig_chua_I, "e40:fig_chua_II": fig_chua_II, "e40:fig_li_sprott": fig_li_sprott,
    "e40:fig_maps": fig_maps, "e40:fig_jafari": fig_jafari, "e40:fig_d2_mechanism": fig_d2_mechanism,
    "e40:fig_sus_chains": fig_sus_chains, "e40:fig_hiddenness": fig_hiddenness, "e40:fig_verification": fig_verification,
}
EXTRA = {"e10:fig_chua_transient": fig_chua_transient}
