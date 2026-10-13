"""Единый стиль графиков (палитра из skill dataviz: проверена на дальтонизм, фиксированный порядок)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
            "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
ESCAPE_GREY = "#bbbbb6"

# фиксированные роли цветов для меток аттракторов
ROLE = {"inf": ESCAPE_GREY, "escape": ESCAPE_GREY, "unknown": INK,
        "F0": SERIES[1], "FP": SERIES[1], "E0": SERIES[1],
        "HA_a": SERIES[0], "HA_b": SERIES[2], "HA": SERIES[0],
        "Fp": SERIES[7], "Fm": SERIES[6], "new0": SERIES[3], "new1": SERIES[4], "new2": SERIES[5]}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.markersize": 6, "legend.frameon": False, "font.size": 10,
        "font.family": ["DejaVu Sans"], "axes.titlesize": 11, "axes.titleweight": "normal",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        # embed text as Type 42 (TrueType) fonts, as journals require, instead of Type 3
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def color_for(name, i=0):
    return ROLE.get(name, SERIES[i % len(SERIES)])


# Подписи систем и режимов для рисунков статьи (вместо внутренних имён реестра)
NICE_SYS = {"chua_I": "Чуа I", "chua_II": "Чуа II", "ne4": "NE4", "wangchen1": "Ванг–Чен #1", "wei_ne2": "Вэй (NE2)",
            "wangchen2": "Ванг–Чен #2", "se3": "SE3", "se11": "SE11", "le1": "LE1", "rabinovich_100": "Рабинович, r = 100",
            "rabinovich_68": "Рабинович, r = 6.8", "jafari2026_lc": "Jafari 2026", "jafari2026_chaos": "Jafari 2026",
            "jiang_a": "Цзян (а)", "jiang_b": "Цзян (б)", "jiang_c": "Цзян (в)", "huynh2019": "Хуинь 2019",
            "li_sprott_4d_a6": "Ли–Спротт 4D", "li_sprott_4d_a7": "Ли–Спротт 4D, a = 7", "li_sprott_4d_hyper": "Ли–Спротт 4D, гиперхаос",
            "jahanshahi2019_4d": "Джаханшахи 4D", "lorenz_transient": "Лоренц (4, 29, 2)",
            "jiang_nfe": "Цзян NF I e", "jiang_nff": "Цзян NF I f", "jiang_nfg": "Цзян NF I g", "jiang_nfi": "Цзян NF I i",
            "jiang_sfa": "Цзян SF I a", "jiang_sfb": "Цзян SF I b", "jiang_sfc": "Цзян SF I c", "jiang_sfd": "Цзян SF I d"}
NICE_TGT = {("chua_II", "new0"): "цикл LC", ("chua_II", "new1"): "хаос A", ("chua_II", "new2"): "хаос B",
            ("chua_II", "LC"): "цикл LC", ("chua_II", "HA_a"): "хаос A", ("chua_II", "HA_b"): "хаос B",
            ("chua_I", "HA_a"): "хаос A", ("chua_I", "HA_b"): "хаос B", ("li_sprott_4d_a6", "T"): "тор",
            ("li_sprott_4d_a6", "SA_a"): "странный A", ("li_sprott_4d_a6", "SA_b"): "странный B",
            ("jahanshahi2019_4d", "new0"): "хаос", ("jafari2026_lc", "LC"): "цикл", ("jafari2026_chaos", "HA"): "хаос",
            ("lorenz_transient", "TC"): "переходный хаос", ("rabinovich_68", "HA"): "переходный хаос",
            # режимы, найденные кластеризацией в e02 (new0 у отображений Цзяна NF — периодические)
            ("jiang_nff", "new0"): "цикл", ("jiang_nfg", "new0"): "цикл", ("jiang_nfi", "new0"): "цикл",
            ("li_sprott_4d_a7", "SA_a"): "странный A", ("li_sprott_4d_a7", "SA_b"): "странный B",
            ("li_sprott_4d_a7", "LC"): "цикл", ("li_sprott_4d_hyper", "HC"): ""}


def nice_label(name, target):
    """Только имя системы для её (единственного) скрытого аттрактора HA, иначе «система: режим»."""
    t = NICE_TGT.get((name, target))
    if t is None:
        t = "" if target == "HA" else target
    s = NICE_SYS.get(name, name)
    return f"{s}: {t}" if t else s


def save(fig, path_without_ext):
    fig.savefig(str(path_without_ext) + ".png")
    fig.savefig(str(path_without_ext) + ".pdf")
    plt.close(fig)
