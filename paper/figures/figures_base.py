"""Redraws used by make_figures_base.py (same data as the plotting scripts, journal page sizes).

OVERRIDE replaces figure functions of the executed plotting scripts, as OVERRIDE of paper/figures/figures_en.py does;
each function receives the namespace of the script. The functions below OVERRIDE draw the figures of the separate
scripts e53, e54 and e61 from their saved results (results/e53_tables.json, results/e54_cost_final.json,
results/e58_eps_scan_*.json) without running those scripts, which also write the Russian versions.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from hsearch import viz
from hsearch.viz import ESCAPE_GREY, INK, INK2, MUTED, SERIES, panel_label

import figures_en as fe

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"


# Layout patches for the translated sources (as PATCH of make_figures_en.py, applied after it)
PATCH = [
    # fig25: the title repeats the caption
    ("e26", 'ax.set_title("Ablation: portfolio components ($G_{0.9}$ relative to uniform sampling)")', "pass"),
    # fig24: the two-column legend is wider than the column; one column below the axes
    ("e26", 'bbox_to_anchor=(0.5, -0.22), ncol=2, columnspacing=0.8, handlelength=2.0)',
            'bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=2.6)'),
]


# ============================================================================================ e26
def fig_scaling_p(ns):
    """fig23: as figures_en.fig_scaling_p on the base page: short names in the exponent lines, legend in two
    columns below the panels."""
    LABEL, COLOR, q09, slope_ci, _sweep = ns["LABEL"], ns["COLOR"], ns["q09"], ns["slope_ci"], ns["_sweep"]
    from matplotlib.transforms import offset_copy
    ps = [1e-2, 1e-3, 1e-4, 1e-5]
    short = {"uniform": "uniform", "sus_rarity": "rarity", "sus": "transient length",
             "portfolio_no_sus": "active learning", "portfolio": "portfolio"}
    fig, axes = plt.subplots(1, 2, sharey=True)
    for ax, lab, prefix, title, show, lo_exp in (
            (axes[0], "(a)", "edge", "D2 (graded signal)",
             ["uniform", "sus_rarity", "sus", "portfolio_no_sus", "portfolio"], -5.3),
            (axes[1], "(b)", "emb", "D1 (no signal)", ["uniform", "sus", "portfolio_no_sus", "portfolio"], -4.3)):
        data = _sweep(prefix, 3, ps)
        pp = np.logspace(lo_exp, -1.7, 50)
        ax.plot(pp, np.log(10) / pp, color=MUTED, lw=0.9, ls="-", label="ln 10 / p (theory)", zorder=1)
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
                lines.append((f"{short[sname]}: s = {s_fit:.2f} [{lo:.2f}, {hi:.2f}]", COLOR[sname]))
            if xs:
                ax.plot(xs, ys, "o-", color=COLOR[sname], label=LABEL[sname])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
        ax.set_xlabel("target basin fraction p")
        panel_label(ax, lab, title)
        for i, (text, c) in enumerate(lines):
            ax.text(0.03, 0.97, text, transform=offset_copy(ax.transAxes, fig=fig, x=0, y=-10.0 * i, units="points"),
                    va="top", ha="left", color=INK if c == MUTED else c)
    axes[0].set_ylim(60, 1e7)
    axes[0].set_ylabel("$N_{0.9}$ (integrations)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), columnspacing=2.0, handlelength=2.4,
               handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.0)
    viz.save(fig, ns["FIG"] / "fig23_scaling_p")


def fig_signal(ns):
    """fig26: as in figures_en.fig_signal; the upper-set fraction is called alpha (q is the nonescaping measure)."""
    load, q09 = ns["load"], ns["q09"]
    systems = [("chua_II", "Chua II"), ("le1", "LE1"), ("se11", "SE11"), ("edge_se3_m3_p0.001", "D2, p = 10⁻³"),
               ("huynh2019", "Huynh"), ("jafari2026_lc", "Jafari cycle"), ("emb_se3_m3_p0.001", "D1, p = 10⁻³")]
    fig, ax = plt.subplots()
    for i, (name, lab) in enumerate(systems):
        p = ns["RES"] / f"e22_signal_{name}.json"
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
                marker=fe.MARKERS[i % len(fe.MARKERS)], ms=4.5, lw=1.6 if gain else 1.2, color=SERIES[i % 8],
                label=f"{lab} (G = {g:.1f})" if g else lab)
    ax.set_xscale("log"); ax.invert_xaxis(); ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 1.5)
    ax.set_xlabel("upper-set fraction $\\alpha$ of the coordinate"); ax.set_ylabel("$d(\\alpha)$, relative to $\\alpha = 1$")
    fe._legend_below(ax, dy=-0.3, ncol=2, handlelength=2.6)
    fig.tight_layout()
    viz.save(fig, ns["FIG"] / "fig26_signal_curves")


# ============================================================================================ e40
def fig_maps(ns):
    """fig34: as in figures_en.fig_maps on the narrower base page: one row of four maps with short panel titles
    (the cases are named in the caption), legend below in two rows."""
    attractor, draw_basin, basin_grid = ns["attractor"], ns["draw_basin"], ns["basin_grid"]
    from e20_search_benchmark import build_problem
    cases = [("huynh2019", "(a)", "Huynh", (-3, 3, -3, 3)), ("jiang_nfe", "(b)", "NF I e", (-8, 8, -8, 8)),
             ("jiang_nff", "(c)", "NF I f", (-5, 5, -5, 5)), ("jiang_sfa", "(d)", "SF I a", (-5, 5, -5, 5))]
    fig, axes = plt.subplots(1, 4)
    for ax, (name, lab, title, box) in zip(axes, cases):
        s, refs, tidx, kidx = build_problem(name, np.random.default_rng(100))
        names = [r.name for r in refs]
        cols = [fe.CHAOS_BLUE if n == "HA" else (fe.POINT_ORANGE if n in ("FP", "E0") else fe.CYCLE_WHITE) for n in names]
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
        ax.set_xlabel("x", labelpad=1)
        ax.set_ylabel("y", labelpad=0)
        ax.tick_params(axis="both", pad=1.5)
    handles = [Patch(color=ESCAPE_GREY, label="escape"), Patch(color=fe.CHAOS_BLUE, label="chaos basin"),
               Patch(facecolor=fe.CYCLE_WHITE, edgecolor=INK2, lw=0.6, label="cycle basin"),
               Patch(color=fe.POINT_ORANGE, label="point basin"),
               Line2D([], [], ls="", marker="s", ms=2.5, color=INK, label="chaotic attractor"),
               Line2D([], [], ls="", marker="o", ms=4.5, color=INK, mec="white", label="period-2 cycle"),
               Line2D([], [], ls="", marker="s", ms=5, color=INK, mec="white", label="stable fixed point")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), columnspacing=1.0,
               handlelength=1.3, handletextpad=0.5)
    fig.tight_layout(rect=(0, 0.2, 1, 1), w_pad=0.6)
    viz.save(fig, ns["FIG"] / "fig34_maps_phase_basin")


def _style3d(ax, n=3, labelpad=-5):
    """3D axes on the narrow base panels: three ticks per axis, labels close to the axes."""
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(n))
        axis.pane.set_facecolor((0.98, 0.98, 0.97, 1.0)); axis.pane.set_edgecolor("#e1e0d9")
    ax.set_xlabel("x", labelpad=labelpad); ax.set_ylabel("y", labelpad=labelpad); ax.set_zlabel("z", labelpad=labelpad)
    ax.tick_params(axis="both", pad=-2)
    ax.tick_params(axis="z", pad=-1)
    ax.grid(False)
    ax.patch.set_alpha(0.0)              # neighbouring 3D panels must not hide each other's axes


def fig_chua_I(ns):
    """fig30: as figures_en.fig_chua_I with an explicit layout for the base width: a larger 3D panel, legends below
    the panels in two columns with short entries."""
    systems, attractor, draw_basin, RES = ns["systems"], ns["attractor"], ns["draw_basin"], ns["RES"]
    s = systems.get("chua_I")
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.25, 0.56, 0.75], projection="3d")
    for name, c, lab in (("HA_a", SERIES[0], "attractor A"), ("HA_b", SERIES[2], "attractor B")):
        tr = attractor(s, s.known[name], T=300)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=c, lw=0.35)
        ax.plot([], [], color=c, label=lab)
    eq = s.equilibria
    ax.scatter(*eq[0], color=SERIES[1], s=30, depthshade=False, label="F$_0$ (stable)")
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
            tr = tr[(np.abs(tr[:, 0]) <= 12) & (np.abs(tr[:, 1]) <= 4) & (np.abs(tr[:, 2]) <= 15)]   # inside the box
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=MUTED, lw=0.8, ls="--")
    ax.scatter([], [], color=SERIES[7], marker="x", label="F$_\\pm$ (saddle-foci)")
    ax.plot([], [], color=MUTED, ls="--", label="unstable manifolds")
    ax.set_xlim(-12, 12); ax.set_ylim(-4, 4); ax.set_zlim(-15, 15)
    _style3d(ax); ax.view_init(elev=18, azim=-60)
    ax.set_box_aspect(None, zoom=1.0)
    ax.text2D(0.04, 0.95, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", bbox_to_anchor=(0.28, 0.0), ncol=2, columnspacing=1.0, handlelength=1.6)
    ax2 = fig.add_axes([0.665, 0.31, 0.32, 0.63])
    img = np.load(RES / "e01_chua_basin_z0_labels.npy")
    d = json.loads((RES / "e01_chua_table1.json").read_text(encoding="utf-8"))["basin_z0"]
    a = np.linspace(d["x"][0], d["x"][1], img.shape[1]); b = np.linspace(d["y"][0], d["y"][1], img.shape[0])
    draw_basin(ax2, a, b, img, ["F0", "Fp", "Fm", "HA_a", "HA_b"], [SERIES[1], SERIES[7], SERIES[6], SERIES[0], SERIES[2]])
    for name, c in (("HA_a", "#0d366b"), ("HA_b", "#0d4f3a")):
        tr = attractor(s, s.known[name], T=200)
        sel = np.abs(tr[:, 2]) < 0.15
        ax2.plot(tr[sel, 0], tr[sel, 1], ".", ms=2.0, color=c)
    ax2.set_xlabel("x(0)", labelpad=1); ax2.set_ylabel("y(0)", labelpad=0)
    panel_label(ax2, "(b)")
    fig.legend(handles=[Patch(color=ESCAPE_GREY, label="escape"), Patch(color=SERIES[1], label="F$_0$"),
                        Patch(color=SERIES[0], label="attractor A"), Patch(color=SERIES[2], label="attractor B")],
               loc="lower center", bbox_to_anchor=(0.80, 0.0), ncol=2, columnspacing=0.8, handlelength=1.2)
    viz.save(fig, ns["FIG"] / "fig30_chua_I_hidden")


def fig_chua_II(ns):
    """fig31: as figures_en.fig_chua_II with an explicit layout for the base width (two 3D panels, the basin map,
    legends in two columns below them)."""
    systems, attractor, draw_basin = ns["systems"], ns["attractor"], ns["draw_basin"]
    s = systems.get("chua_II")
    eq = s.equilibria
    trs = {"LC": attractor(s, s.known["LC"], T=12), "HA_a": attractor(s, s.known["HA_a"], T=150),
           "HA_b": attractor(s, s.known["HA_b"], T=150)}
    fig = plt.figure()
    boxes3d = ([0.0, 0.25, 0.335, 0.75], [0.335, 0.25, 0.335, 0.75])
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
        _style3d(ax); ax.view_init(elev=15, azim=-55)
        if not zoom:
            ax.set_zlabel("")      # it would fall into the area of panel (b); the z axis is labelled there
        ax.set_box_aspect(None, zoom=0.95)
        ax.text2D(0.04, 0.95, "(b)" if zoom else "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)
        if not zoom:
            handles, labels = ax.get_legend_handles_labels()
    lg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.34, 0.0), ncol=2, columnspacing=1.0,
                    handlelength=1.6)
    for ln in lg.get_lines():
        ln.set_linewidth(1.6)
    ax2 = fig.add_axes([0.825, 0.34, 0.17, 0.6])
    from e20_search_benchmark import build_problem
    s2, refs, tidx, kidx = build_problem("chua_II", np.random.default_rng(100))
    names = [r.name for r in refs]
    colors = [{"E0": SERIES[7], "E1": SERIES[1], "E2": "#b85a2a", "LC": SERIES[6], "HA_a": SERIES[0], "HA_b": SERIES[2]}.get(n, SERIES[5])
              for n in names]
    a, b = np.linspace(-3, 3, 300), np.linspace(-1.5, 1.5, 200)
    img = np.load(RES / "e40_chua_II_basin_z0.npy")
    draw_basin(ax2, a, b, img, names, colors)
    ax2.set_xlabel("x(0)", labelpad=1); ax2.set_ylabel("y(0)", labelpad=0)
    panel_label(ax2, "(c)")
    fig.legend(handles=[Patch(color=SERIES[6], label="cycle LC"), Patch(color=SERIES[1], label="F+"),
                        Patch(color="#b85a2a", label="F−"), Patch(color=SERIES[0], label="chaos A"),
                        Patch(color=SERIES[2], label="chaos B")],
               loc="lower center", bbox_to_anchor=(0.87, 0.0), ncol=2, columnspacing=0.5, handlelength=0.8,
               handletextpad=0.4)
    viz.save(fig, ns["FIG"] / "fig31_chua_II_coexisting")


GALLERY_TITLES = {"chua_I": "Chua I", "chua_II": "Chua II", "ne4": "NE4", "wangchen1": "Wang–Chen #1",
                  "wei_ne2": "Wei (NE2)", "wangchen2": "Wang–Chen #2", "se3": "SE3", "se11": "SE11", "le1": "LE1",
                  "rabinovich_100": "Rabinovich, r = 100", "jafari2026_lc": "Jafari 2026, cycle",
                  "jafari2026_chaos": "Jafari 2026, chaos"}


def fig_gallery(ns):
    """fig32: the gallery of e40 on the base page: system names as titles (the equilibria are described in the
    caption), three ticks per axis, explicit spacing."""
    systems, attractor = ns["systems"], ns["attractor"]
    fig = plt.figure()
    for k, (name, key, _title) in enumerate(ns["GALLERY"]):
        s = systems.get(name)
        ax = fig.add_subplot(4, 3, k + 1, projection="3d")
        T = 400 if name not in ("ne4",) else 200
        tr = attractor(s, s.known[key], T=T)
        if len(tr) > 80000:
            tr = tr[:: len(tr) // 80000 + 1]
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], color=SERIES[0], lw=0.3, alpha=0.85)
        for e in s.equilibria[:3]:
            ax.scatter(*e, color=SERIES[7], marker="x", s=24, depthshade=False)
        if name == "le1":
            zz = np.linspace(tr[:, 2].min(), tr[:, 2].max(), 2)
            ax.plot([0, 0], [0, 0], zz, color=SERIES[7], lw=1.2)
        _style3d(ax)
        ax.set_box_aspect(None, zoom=0.86)
        ax.set_title(GALLERY_TITLES.get(name, name), fontsize=8.5, pad=-2)
    fig.subplots_adjust(left=0.0, right=0.93, bottom=0.01, top=0.97, wspace=0.12, hspace=0.12)
    viz.save(fig, ns["FIG"] / "fig32_gallery_3d")


OVERRIDE = {"e26:fig_scaling_p": fig_scaling_p, "e26:fig_signal": fig_signal, "e40:fig_maps": fig_maps,
            "e40:fig_chua_I": fig_chua_I, "e40:fig_chua_II": fig_chua_II, "e40:fig_gallery": fig_gallery}


# ============================================================================================ post-hooks
def _post_d2_mechanism(fig):
    """fig37: short legend entries of (a) (the caption names the law and the sample), so that the legend fits the
    empty lower left without touching the curve."""
    h, _ = fig.axes[0].get_legend_handles_labels()
    fig.axes[0].legend(h, ["uniform initial conditions", "exact law"], loc="lower left", markerscale=3)


def _post_verification(fig):
    """fig41: short legend entries in the empty lower right of (a); the legend of (b) below the decaying curves."""
    a, b = fig.axes[0], fig.axes[1]
    h, _ = a.get_legend_handles_labels()
    a.legend(h, ["chaotic", "regular"], loc="lower right")
    h, lab = b.get_legend_handles_labels()
    b.legend(h, lab, loc="lower left", handlelength=1.6)


POST = {"fig37_d2_mechanism": _post_d2_mechanism, "fig41_verification": _post_verification}


# ============================================================================================ e54, e61, e53
def fig_cost_gains(width, save):
    """fig30_cost_gains (e54.figure, English): the legend in the empty upper right of the axes instead of the lower
    left, where it covered the bounded systems."""
    import e54_cost_figure as e54
    sel = json.loads((RES / "e54_cost_final.json").read_text(encoding="utf-8"))
    t = e54.T["en"]
    fig, ax = plt.subplots(figsize=(width, 3.55))
    y = np.arange(len(sel))[::-1]
    col = {"G_N": MUTED, "G_S": SERIES[1], "G_T": SERIES[0]}
    lab = {k: t[k[-1]] for k in ("G_N", "G_S", "G_T")}
    for yi, r in zip(y, sel):
        vals = [r[k] for k in ("G_N", "G_S", "G_T") if r[k]]
        ax.plot([min(vals), max(vals)], [yi, yi], "-", color=INK2, lw=0.8, alpha=0.6, zorder=1)
        if r["G_T"]:
            ax.plot(r["G_T"], yi, "D", ms=7, mfc="white", mec=col["G_T"], mew=1.3, zorder=3,
                    label=lab["G_T"] if yi == y[0] else None)
        if r["G_S"]:
            ax.plot(r["G_S"], yi, "s", ms=4.5, color=col["G_S"], mec="white", mew=0.5, zorder=4,
                    label=lab["G_S"] if yi == y[0] else None)
        if r["G_N"]:
            ax.plot(r["G_N"], yi, "o", ms=4, color=col["G_N"], mec="white", mew=0.5, zorder=5,
                    label=lab["G_N"] if yi == y[0] else None)
    ax.axvline(1.0, color=INK, lw=0.8, ls="--", zorder=0)
    nb = sum(1 for r in sel if r["system"].startswith(("chua_II", "coupled_chua")))
    ax.set_xscale("log")
    ax.set_xlim(0.05, 4e4)               # room for the legend to the right of the escaping systems
    ax.set_yticks(y)
    ax.set_yticklabels([r["en"] for r in sel])
    ax.set_xlabel(t["x"])
    ax.set_ylim(-0.7, len(sel) - 0.3)
    ax.grid(axis="y", visible=False)
    if 0 < nb < len(sel):
        import matplotlib.transforms as mtr
        tr = mtr.blended_transform_factory(ax.transAxes, ax.transData)
        ax.axhspan(-0.5, nb - 0.5, color=SERIES[0], alpha=0.06, zorder=0, lw=0)
        for y0, y1, key in ((nb - 0.5, len(sel) - 0.5, "esc"), (-0.5, nb - 0.5, "bnd")):
            ax.plot([1.015, 1.015], [y0 + 0.15, y1 - 0.15], "-", color=INK2, lw=0.8, transform=tr, clip_on=False)
            ax.text(1.035, 0.5 * (y0 + y1), t[key], transform=tr, rotation=90, va="center", ha="left", color=INK2)
    ax.legend(loc="upper right", handletextpad=0.3, borderaxespad=0.3, labelspacing=0.3, frameon=True,
              framealpha=1.0, edgecolor="#d0d0d0", fancybox=False)
    fig.tight_layout()
    save(fig, "fig30_cost_gains")


def fig_new_attractor(width, save):
    """fig31_new_attractor (e61, English) at the base text width: shorter legend entries under the panels."""
    import numba
    import e50_batch_runner as runner
    numba.set_num_threads(2)
    var = {"coupled": "chua_II", "k": 2, "eps": 0.1, "coord": 0, "ha": ["HA_a", "HA_b"], "products": True, "alleq": True}
    s, refs, tidx, kidx, _ = runner.get_problem(var, None)
    r = next(q for q in refs if q.name == "P_LC-HA_a")
    tr = s.traj(np.asarray(r.example, float), n=120000)[40000:]
    b = runner.get_problem({"system": "chua_II"}, None)
    ref_b = {q.name: q for q in b[1]}
    single = {nm: b[0].traj(np.asarray(ref_b[nm].example, float), n=80000)[20000:] for nm in ("LC", "HA_a", "HA_b")}
    scan = []
    for tag in ("_down", "_up"):
        scan += json.loads((RES / f"e58_eps_scan{tag}.json").read_text(encoding="utf-8"))["points"]
    seen, pts = set(), []
    for q in sorted(scan, key=lambda q: q["eps"]):
        if q["eps"] not in seen:
            seen.add(q["eps"]); pts.append(q)
    fig, ax = plt.subplots(1, 3, figsize=(width, 2.25))
    a = ax[0]
    a.plot(single["LC"][:, 0], single["LC"][:, 2], "-", color=ESCAPE_GREY, lw=1.6, label="single circuit: LC")
    a.plot(tr[:, 0], tr[:, 2], "-", color=INK, lw=0.3, label="circuit 1")
    a.set_xlabel("$x_1$", labelpad=1); a.set_ylabel("$z_1$", labelpad=0)
    panel_label(a, "(a)", "circuit 1")
    a = ax[1]
    for nm in ("HA_a", "HA_b"):
        a.plot(single[nm][:, 0], single[nm][:, 2], "-", color=ESCAPE_GREY, lw=0.5,
               label="single circuit: HA$_a$, HA$_b$" if nm == "HA_a" else None)
    a.plot(tr[:, 3], tr[:, 5], "-", color=INK, lw=0.22, label="circuit 2")
    a.set_xlabel("$x_2$", labelpad=1); a.set_ylabel("$z_2$", labelpad=0)
    panel_label(a, "(b)", "circuit 2")
    a = ax[2]
    xs, lam, kind = [], [], []
    for q in pts:
        g = q["regimes"].get("P_LC-HA_a") or {}
        if g.get("status") == "ok" and g.get("lam") is not None:
            xs.append(q["eps"]); lam.append(g["lam"]); kind.append(g["kind"])
    xs, lam = np.array(xs), np.array(lam)
    cha = np.array([k == "chaotic" for k in kind])
    a.plot(xs, lam, "-", color=INK2, lw=0.6)
    a.plot(xs[cha], lam[cha], "o", color=SERIES[0], ms=3, label="chaotic")
    a.plot(xs[~cha], lam[~cha], "s", color=SERIES[1], mfc="white", mew=1.0, ms=3.2, label="periodic")
    a.axhline(0, color=MUTED, lw=0.6, ls="--")
    a.set_xlabel("coupling $\\epsilon$", labelpad=1); a.set_ylabel("largest Lyapunov exponent λ", labelpad=1)
    panel_label(a, "(c)")
    for a, nc in zip(ax, (1, 1, 2)):
        lg = a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=nc, handlelength=1.6,
                      columnspacing=0.8, borderaxespad=0.0)
        for ln in lg.get_lines():
            ln.set_linewidth(1.6) if ln.get_marker() in ("None", None, "") else None
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig31_new_attractor")


def fig_enlarged_and_smallp(w2, w1, save):
    """fig27_enlarged_regions and fig29_d2_small_p (e53, English) from results/e53_tables.json at the base sizes."""
    import e53_paper_results as e53
    out = json.loads((RES / "e53_tables.json").read_text(encoding="utf-8"))
    sizes = {(6.69, 2.7): (w2, 3.0), (3.37, 2.8): (w1, 2.85)}
    e53._wh = lambda lang, w, h, scale_ru=1.45: sizes.get((w, h), (w, h))
    e53.T["en"].update({"t27a": "basin measure", "t27b": "gain"})      # the caption describes the panels
    fig = e53.fig27(out, "en")
    ax0 = fig.axes[0]                    # the system legend serves both panels: below them, not over the data of (a)
    h, l = ax0.get_legend_handles_labels()
    ax0.get_legend().remove()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), columnspacing=1.5, handlelength=2.0)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, "fig27_enlarged_regions")
    save(e53.fig29(out, "en"), "fig29_d2_small_p")
