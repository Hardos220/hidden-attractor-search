"""English, journal-size versions of the article figures (Chaos submission) -> paper/chaos/figures/*.pdf|png

The original plotting scripts (experiments/e10, e26, e35, e40) are executed unchanged on disk, with:
  1. Russian display strings replaced by English ones in the source text before execution (TR below);
  2. figure sizes set for the journal column widths (3.37 in / 6.69 in) and fonts of 7-8.5 pt (FIGSIZE, _fs);
  3. layout tweaks for the smaller canvases (PATCH);
  4. suptitles dropped (their content goes to the LaTeX captions; the texts are logged to _build/suptitles.txt);
  5. output redirected to paper/chaos/figures/, any write into results/ diverted to _build/side/.
Data are the same as in the Russian figures: the scripts read results/*.json|npy|npz and use fixed seeds.

    python paper/chaos/make_figures_en.py                 # all figures
    python paper/chaos/make_figures_en.py e40:fig_chua_I  # selected figure functions
    python paper/chaos/make_figures_en.py --esn           # also the ESN figures (e35) -> paper/esn/figures_en
"""
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
BUILD = Path(__file__).resolve().parent / "_build"
SIDE = BUILD / "side"
OUT.mkdir(exist_ok=True)
BUILD.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure, FigureBase  # noqa: E402

from hsearch import viz  # noqa: E402

W1, W2 = 3.37, 6.69          # AIP single- and double-column widths, inches
SCRIPTS = {"e10": "e10_figures.py", "e26": "e26_stage2_figures.py", "e35": "e35_esn_figures.py",
           "e40": "e40_attractor_figures.py"}
# the echo-state-network figures (e35) belong to the separate ESN paper: paper/esn/figures_en, run with --esn
OUTDIRS = {"e35": ROOT / "paper" / "esn" / "figures_en"}
OPTIONAL = {"e35"}

# ------------------------------------------------------------------------------------------------
# 1. Translation. Keys are exact substrings of the script sources (f-string fragments included);
#    replacement is longest-first, short words are keyed with their quotes.
TR = {
    # --- e10
    "Цепь Чуа, набор I: бассейны притяжения на плоскости z(0) = 0": "Chua circuit, set I: basins of attraction in the plane z(0) = 0",
    "уход на бесконечность": "escape to infinity",
    "равновесие F0": "equilibrium F0",
    "скрытый аттрактор A": "hidden attractor A",
    "скрытый аттрактор B": "hidden attractor B",
    "доля копий, остающихся хаотическими": "fraction of copies still chaotic",
    "Чуа I: «выживание» на скрытом хаотическом множестве (шум 1e-3)": "Chua I: survival on the hidden chaotic set (noise 10⁻³)",
    "DOP853, rtol = 1e-11: хаос сохраняется до t = 4000": "DOP853, rtol = 10⁻¹¹: chaos persists up to t = 4000",
    "не сопоставлено": "unmatched",
    "шаг интегрирования dt (RK4)": "integration step dt (RK4)",
    "доля начальных условий (шар r = 3)": "fraction of initial conditions (ball r = 3)",
    "Чуа I: сходимость долей бассейнов по шагу (T_tr = 1000, n = 20000)": "Chua I: basin fractions vs step ($T_\\mathrm{tr}$ = 1000, n = 20000)",
    "шар радиуса 3": "ball of radius 3",
    "шар радиуса 5": "ball of radius 5",
    "обзор Guan & Xie (2025)": "review by Guan & Xie (2025)",
    "доля начальных условий": "fraction of initial conditions",
    "Чуа I: доли бассейнов при равномерной выборке в шаре — три источника": "Chua I: basin fractions under uniform sampling in a ball, three sources",
    "N_0.9 — число интегрирований для вероятности обнаружения 0.9 (равномерный перебор)": "$N_{0.9}$ (integrations), uniform sampling",
    "Базовая стоимость равномерного перебора": "Baseline cost of uniform sampling",
    "p_изм = p_аналит": "measured = analytic",
    "аналитическая доля бассейна p": "analytic basin fraction p",
    "измеренная доля (перебор, 95% ДИ)": "measured fraction (uniform, 95% CI)",
    "Синтетическое семейство D1: вложение ": "Synthetic family D1: ",
    " в ℝ³⁺ᵐ": " embedded in ℝ³⁺ᵐ",
    "нульклины (CVS, лучшая)": "nullclines (CVS, best)",
    "соединяющие кривые (CC)": "connecting curves (CC)",
    "не найдено": "not found",
    "PP: нет точек": "PP: no points",
    "число интегрирований до обнаружения с вероятностью 0.9": "integrations to detection with probability 0.9",
    "Классические эвристики против перебора (уровень A); справа — перпетуальные точки: попаданий/всего": "Classical heuristics vs uniform sampling (level A); right: perpetual points, hits/total",
    "Скрытые аттракторы эталонного набора (уровень A), проекции на плоскость (x, z)": "Hidden attractors of the benchmark set (level A), projections onto the (x, z) plane",
    # --- e26
    "равномерный перебор": "uniform sampling",
    "DE Зелинки (оракул размера)": "Zelinka's DE (size oracle)",
    "AMS (1-е поколение)": "AMS (1st generation)",
    "MAP-Elites (1-е поколение)": "MAP-Elites (1st generation)",
    "расщепление: время ухода": "splitting: escape time",
    "расщепление: редкость": "splitting: rarity",
    "расщепление: длит. переходного процесса": "splitting: transient length",
    "активное обучение + 20 % равн.": "active learning + 20% uniform",
    "расщепление + 20 % равн.": "splitting + 20% uniform",
    "портфель без равн. доли": "portfolio, no uniform share",
    "портфель (предлагаемый метод)": "portfolio (proposed)",
    "портфель + нульклины": "portfolio + nullclines",
    "Jafari 2026 (цикл)": "Jafari 2026 (cycle)",
    "отображение Хуиня": "Huynh map",
    "Jafari 2026 (хаос)": "Jafari 2026 (chaos)",
    "оба хаотических": "both chaotic",
    "N, интегрирований": "N, integrations",
    "Вероятность обнаружения скрытого аттрактора как функция бюджета (R = 100 запусков)": "Detection probability vs budget (R = 100 runs)",
    "| Система | p | стратегия | успех | N_0.9 | G_0.9 [95 % ДИ] |": "| System | p | strategy | success | N_0.9 | G_0.9 [95% CI] |",
    "G_0.9 = N_0.9(перебор) / N_0.9(метод)": "$G_{0.9} = N_{0.9}^\\mathrm{unif}/N_{0.9}^\\mathrm{method}$",
    "Коэффициент выигрыша на редких целях уровней A/B (95 % бутстрап-интервалы)": "Gain on rare targets of levels A/B (95% bootstrap intervals)",
    "D2: «краевое состояние» (градиентный сигнал)": "D2: edge state (graded signal)",
    "D1: «ворота» (сигнала нет)": "D1: gate (no signal)",
    "ln 10 / p (перебор, теория)": "ln 10 / p (uniform, theory)",
    "доля бассейна цели p": "target basin fraction p",
    "N_0.9 (интегрирований)": "$N_{0.9}$ (integrations)",
    "Стоимость обнаружения против меры бассейна (n = 6, R = 100); s — показатель в N_0.9 ∝ p^(−s)": "Detection cost vs basin measure (n = 6, R = 100); s is the exponent in N_0.9 ∝ p^(−s)",
    "размерность n": "dimension n",
    "Зависимость стоимости от размерности фазового пространства (D2)": "Cost vs phase-space dimension (D2)",
    'set_ylabel("N_0.9")': 'set_ylabel("$N_{0.9}$")',
    "Абляция: вклад компонентов портфеля (G_0.9 относительно перебора)": "Ablation: portfolio components ($G_{0.9}$ relative to uniform sampling)",
    'set_label("log10 G_0.9")': 'set_label("$\\log_{10} G_{0.9}$")',
    '"Хуинь"': '"Huynh"',
    "цикл Jafari": "Jafari cycle",
    "доля q верхнего множества координаты": "upper-set fraction q of the coordinate",
    "медиана расстояния до цели (отн.)": "median distance to target (rel.)",
    "Сходятся ли «долгие» переходные процессы к цели?": "Do long transients approach the target?",
    "G_0.9 расщепления": "splitting $G_{0.9}$",
    "Наблюдаемый выигрыш": "Observed gain",
    # --- e35
    "реконструированный аттрактор": "reconstructed attractor",
    "паразитная устойчивая точка": "spurious stable point",
    "паразитный регулярный режим": "spurious regular regime",
    "долгий переходный процесс": "long transient",
    "Система Лоренца (10, 28, 8/3)": "Lorenz system (10, 28, 8/3)",
    "Эхо-сеть N = 300 (реализация {seed}) в замкнутом режиме": "ESN, N = 300 (network {seed}), closed loop",
    "истинный ряд": "true series",
    "прогноз эхо-сети": "ESN forecast",
    "ляп. врем.": "Lyap. times",
    "время, ляпуновские времена (λ·t)": "time, Lyapunov times (λt)",
    "Краткосрочный прогноз с конца обучающего ряда": "Short-term forecast from the end of training",
    "VPT, ляп. времена (справа — λ «климата»; у системы Лоренца 0.906)": "VPT, Lyapunov times (right: climate λ; Lorenz 0.906)",
    "Восемь реализаций: краткосрочный прогноз и долгосрочная динамика": "Eight networks: short-term forecast and long-term dynamics",
    "обрушение в неподвижную точку": "collapse to a fixed point",
    "периодический «климат»": "periodic climate",
    "нет данных": "no data",
    "паразитных режимов не найдено": "no spurious regimes found",
    "хаос переходный, утечка в паразитную точку": "transient chaos, leak to a spurious point",
    "сосуществуют паразитные аттракторы": "coexisting spurious attractors",
    "паразитный {'хаос' if a['kind'] == 'chaos' else 'регулярный режим'} (куб {a['p_cube']:.0%})":
        "spurious {'chaos' if a['kind'] == 'chaos' else 'regular regime'} (cube {a['p_cube']:.0%})",
    "устойчивая точка сети": "stable point of the network",
    "неустойчивые точки сети": "unstable points of the network",
    "C± системы Лоренца": "Lorenz C±",
    "реализация {s}: обрушение с аттрактора {coll:.0%} за 1000 ед. вр.": "network {s}: {coll:.0%} collapse",
    "Паразитные режимы обученных эхо-сетей в физических координатах выхода (серое — реконструированный «аттрактор»)":
        "Spurious regimes of trained ESNs in physical output coordinates (grey: reconstructed attractor)",
    "доля 1000 равномерных состояний куба [−1, 1]³⁰⁰": "fraction of 1000 uniform states of the cube [−1, 1]³⁰⁰",
    "Исходы из куба (горизонт 2·10⁴ шагов + проверка 5·10⁴)": "Outcomes from the cube (2·10⁴ steps + check 5·10⁴)",
    "амплитуда толчка σ от состояния на реконструированном аттракторе": "kick amplitude σ from the reconstructed attractor",
    "доля исходов вне аттрактора": "fraction of outcomes off the attractor",
    "Достижимость паразитных режимов толчком (горизонт 6000 шагов)": "Reachability by kicks (horizon 6000 steps)",
    "нижняя граница — ни одного исхода из 200": "axis floor: no outcomes out of 200",
    "копия C± системы Лоренца": "copy of C±",
    "устойчивая паразитная точка": "stable spurious",
    "неустойчивая точка у копии нуля": "unstable, near copy of 0",
    "номер собственного значения (по убыванию |μ|)": "eigenvalue index (descending |μ|)",
    "|μ| якобиана отображения": "|μ| of the map Jacobian",
    "Спектры в неподвижных точках (реализация {s})": "Spectra at fixed points⏎(network {s})",
    "система Лоренца: Re λ = 0.0940": "Lorenz: Re λ = 0.094",
    "ln|μ_max| / h у копий C±": "$\\ln|\\mu_\\max|/h$ at copies of C±",
    "Копии C±: ": "Copies of C±:⏎",
    " против 0.094": " vs 0.094",
    "устойчивые (паразитные)": "stable (spurious)",
    "копии C±": "copies of C±",
    "прочие неустойчивые": "other unstable",
    "Неподвижные точки восьми сетей в плоскости (x, z)": "Fixed points of eight networks⏎in the (x, z) plane",
    "время, ляпуновские времена": "time, Lyapunov times",
    "доля траекторий, ещё не попавших в точку": "fraction not yet captured by a point",
    "«Выживание» на реконструированном хаотическом множестве": "Survival on the reconstructed chaotic set",
    "выход сети (физические единицы)": "network output (physical units)",
    ": «климат Лоренца», затем обрушение в паразитную точку": ": Lorenz climate, then collapse to a spurious point",
    "паразитный хаос (самовозбуждающийся)": "spurious chaos (self-excited)",
    "скрытый регулярный аттрактор": "hidden regular attractor",
    '"устойчивая точка"': '"stable point"',
    '"неустойчивые точки"': '"unstable points"',
    "Эхо-сеть N = 100: режимы в координатах выхода": "ESN, N = 100: regimes",
    "точка скрытого аттрактора": "point on the hidden attractor",
    "точка реконструированного аттрактора": "point on the reconstructed attractor",
    '"паразитный хаос"': '"spurious chaos"',
    '"скрытый аттрактор"': '"hidden attractor"',
    "α (направление к скрытому аттрактору)": "α (toward the hidden attractor)",
    "β (направление к аттрактору)": "β (toward the attractor)",
    "Сечение бассейнов плоскостью в ℝ¹⁰⁰": "Basin slice by a plane in ℝ¹⁰⁰",
    "скрытый аттрактор и устойчивая точка: 0 из ": "hidden attractor and stable point:⏎0 of ",
    " при каждом ε": " at every ε",
    "ε — радиус окрестности неустойчивых неподвижных точек": "ε, radius around unstable fixed points",
    "доля траекторий h_ε": "fraction of trajectories $h_\\varepsilon$",
    "Мера скрытости по всем неустойчивым точкам": "Hiddenness over all unstable points",
    '"расщепление"': '"splitting"',
    '"портфель"': '"portfolio"',
    "перебор, теория: p̂ = ": "uniform, theory: $\\hat{{p}}$ = ",      # inside an f-string
    "N, оценок (прогонов сети)": "N, evaluations (network runs)",
    "Сеть {meta['seed']} (N = 300): поиск редкой паразитной точки, R = {meta['R']}":
        "Network {meta['seed']} (N = 300):⏎rare spurious point, R = {meta['R']}",
    '"реализация"': '"network"',
    "реализация {s}": "network {s}",
    "Реализация {s}": "Network {s}",
    # --- e40
    "F0 (устойчивый фокус-узел)": "F0 (stable focus-node)",
    "F± (седло-фокусы)": "F± (saddle-foci)",
    "неустойчивые многообразия F±: на ∞ и к F0": "unstable manifolds of F±: to ∞ and to F0",
    "скрытые аттракторы A и B": "hidden attractors A and B",
    "Чуа I: скрытые аттракторы и равновесия": "Chua I: hidden attractors and equilibria",
    "Бассейны на плоскости z(0) = 0 (точки — сечение аттракторов |z| < 0.15)": "Basins at z(0) = 0 (dots: attractors at |z| < 0.15)",
    '"уход"': '"escape"',
    '"аттрактор A"': '"attractor A"',
    '"аттрактор B"': '"attractor B"',
    "предельный цикл LC (0.99 бокса)": "limit cycle LC (0.99 of the box)",
    "хаос A (0.0014)": "chaos A (0.0014)",
    "хаос B (0.0013)": "chaos B (0.0013)",
    "F0 (седло-фокус)": "F0 (saddle-focus)",
    "F± (устойчивые)": "F± (stable)",
    "увеличено: хаотические аттракторы A, B и равновесия": "zoom: chaotic attractors A, B and equilibria",
    "Чуа II: пять сосуществующих аттракторов": "Chua II: five coexisting attractors",
    "Бассейны на плоскости z(0) = 0": "Basins in the plane z(0) = 0",
    '"цикл LC"': '"cycle LC"',
    '"хаос A"': '"chaos A"',
    '"хаос B"': '"chaos B"',
    "Чуа I: устойчивое равновесие": "Chua I: stable equilibrium",
    "Чуа II: устойчивые F±": "Chua II: stable F±",
    "NE4: без равновесий": "NE4: no equilibria",
    "Ванг–Чен #1: без равновесий": "Wang–Chen #1: no equilibria",
    "Вэй (NE2): без равновесий": "Wei (NE2): no equilibria",
    "Ванг–Чен #2: одно устойчивое": "Wang–Chen #2: one stable",
    "SE3: одно устойчивое": "SE3: one stable",
    "SE11: одно устойчивое": "SE11: one stable",
    "LE1: линия равновесий": "LE1: line of equilibria",
    "Рабинович, r = 100": "Rabinovich, r = 100",
    "Jafari 2026: цикл (нет FP и PP)": "Jafari 2026: cycle (no FP, no PP)",
    "Jafari 2026: хаос": "Jafari 2026: chaos",
    "Скрытые аттракторы эталонного набора (× — равновесия; у LE1 — линия равновесий)": "Hidden attractors of the benchmark set (×: equilibria; LE1: line of equilibria)",
    '"тор"': '"torus"',
    "странный аттрактор A": "strange attractor A",
    "странный аттрактор B": "strange attractor B",
    "проекция (x, y)": "projection (x, y)",
    "проекция (u, x)": "projection (u, x)",
    "бассейны в сечении y = u = 0": "basins in the section y = u = 0",
    "4D упрощённая система Лоренца (Li & Sprott 2014), a = 6, b = 0.1: равновесий нет, все аттракторы скрыты":
        "4D simplified Lorenz system (Li & Sprott 2014), a = 6, b = 0.1: no equilibria, all attractors hidden",
    "Хуинь и др. (2019)": "Huynh et al. (2019)",
    "Jiang NF I f: хаос + цикл периода 2": "Jiang NF I f:⏎chaos + period-2 cycle",
    "Jiang SF I a: хаос + устойчивая точка": "Jiang SF I a:⏎chaos + stable point",
    "бассейн хаоса": "chaos basin",
    "бассейн цикла": "cycle basin",
    "бассейн точки": "point basin",
    "Отображения без неподвижных точек или с единственной устойчивой: фазово-бассейновые портреты (чёрное — аттрактор)":
        "Maps without fixed points or with a single stable one: phase-basin portraits (black: attractor)",
    "a₃ = −1: устойчивый цикл": "a₃ = −1: stable cycle",
    "a₃ = −3.548: хрупкий хаос (λ ≈ 0.05)": "a₃ = −3.548: fragile chaos (λ ≈ 0.05)",
    "уход при a₃ ≤ ": "escape for a₃ ≤ ",
    "λ (окно T = 400)": "λ (window T = 400)",
    "окно хаоса у границы ограниченности": "chaos window at the boundedness limit",
    "Система Jafari et al. (2026) без неподвижных и перпетуальных точек: ż = 6z − xy + a₃yz − zx − 1":
        "Jafari et al. (2026) system without fixed or perpetual points: ż = 6z − xy + a₃yz − zx − 1",
    ": переход к устойчивому фокусу C±": ": transition to the stable focus C±",
    "λ по окнам T = 100": "λ in windows T = 100",
    "Отрицательный контроль: переходный хаос системы Лоренца (a, r, b) = (4, 29, 2), ":
        "Negative control: transient chaos of the Lorenz system (a, r, b) = (4, 29, 2), ",
    "начальное условие (": "initial condition (",
    '"время ухода"': '"escape time"',
    "Время ухода расходится у границы бассейна": "Escape time diverges at the basin boundary",
    "граница бассейна |w| = ρ": "basin boundary |w| = ρ",
    "номер уровня (с учётом перезапусков)": "level index (including restarts)",
    "|w| / ρ у цепей уровня": "|w|/ρ of the level chains",
    "Цепи расщепления стягиваются к шару; цель найдена на {res['found_at']}-й оценке":
        "Chains contract to the ball; target found at evaluation {res['found_at']}",
    "Семейство D2 (n = 6, p = 10⁻⁴): механизм выигрыша расщепления": "Family D2 (n = 6, p = 10⁻⁴): mechanism of the splitting gain",
    "уровень {k}": "level {k}",
    "Цепи расщепления по уровням на карте бассейнов (Jiang NF I e); цель на {res['found_at']}-й оценке":
        "Splitting chains by level (Jiang NF I e); target at evaluation {res['found_at']}",
    "Чуа I (оба скрытых)": "Chua I (both hidden)",
    "Чуа II (цикл + хаос)": "Chua II (cycle + chaos)",
    "Ванг–Чен #2": "Wang–Chen #2",
    "Рабинович r = 100": "Rabinovich r = 100",
    "Лоренц (4, 29, 2), переходное": "Lorenz (4, 29, 2), transient",
    "ε / L (радиус окрестности равновесий)": "ε/L (radius around equilibria)",
    "h_ε: доля, попадающая в аттрактор": "$h_\\varepsilon$: fraction reaching the attractor",
    "Мера скрытости: нуль при малых ε у скрытых аттракторов": "Hiddenness measure:⏎zero at small ε for hidden attractors",
    "нижняя граница оси — «ни одной точки»": "axis floor: no points",
    "λ по окнам наблюдения": "λ over observation windows",
    "Показатель Ляпунова по окнам (синие — хаос, оранжевые — циклы/тор)": "Windowed λ (blue: chaos, orange: cycles/torus)",
    "Лоренц (4, 29, 2)": "Lorenz (4, 29, 2)",
    "Рабинович (6.8, −0.5, 0.99)": "Rabinovich (6.8, −0.5, 0.99)",
    "хаос Jafari 2026": "Jafari 2026 chaos",
    "доля копий на множестве": "fraction of copies on the set",
    "Длинная проверка времени жизни (шум 10⁻³)": "Long lifetime check (noise 10⁻³)",
    # shared short labels (quoted)
    '"цикл"': '"cycle"',
    '"хаос"': '"chaos"',
    '"Чуа I"': '"Chua I"',
    '"Чуа II"': '"Chua II"',
}

NICE_SYS_EN = {"chua_I": "Chua I", "chua_II": "Chua II", "wangchen1": "Wang–Chen #1", "wei_ne2": "Wei (NE2)",
               "wangchen2": "Wang–Chen #2", "rabinovich_100": "Rabinovich, r = 100", "rabinovich_68": "Rabinovich, r = 6.8",
               "jiang_a": "Jiang (a)", "jiang_b": "Jiang (b)", "jiang_c": "Jiang (c)", "huynh2019": "Huynh 2019",
               "li_sprott_4d_a6": "Li–Sprott 4D", "li_sprott_4d_a7": "Li–Sprott 4D, a = 7",
               "li_sprott_4d_hyper": "Li–Sprott 4D, hyperchaos", "jahanshahi2019_4d": "Jahanshahi 4D",
               "lorenz_transient": "Lorenz (4, 29, 2)",
               # the Russian figures showed internal registry names for these
               "jiang_nfe": "Jiang NF I e", "jiang_nff": "Jiang NF I f", "jiang_nfg": "Jiang NF I g",
               "jiang_nfi": "Jiang NF I i", "jiang_sfa": "Jiang SF I a", "jiang_sfb": "Jiang SF I b",
               "jiang_sfc": "Jiang SF I c", "jiang_sfd": "Jiang SF I d"}
TGT_EN = {"цикл LC": "cycle LC", "хаос A": "chaos A", "хаос B": "chaos B", "тор": "torus", "странный A": "strange A",
          "странный B": "strange B", "хаос": "chaos", "цикл": "cycle", "переходный хаос": "transient chaos"}
# clustered regimes of e02 ("new0" of the Jiang NF maps is periodic, see results/e02_uniform.json refs)
TGT_EXTRA = {("jiang_nff", "new0"): "cycle", ("jiang_nfg", "new0"): "cycle", ("jiang_nfi", "new0"): "cycle",
             ("li_sprott_4d_a7", "SA_a"): "strange A", ("li_sprott_4d_a7", "SA_b"): "strange B",
             ("li_sprott_4d_a7", "LC"): "cycle", ("li_sprott_4d_hyper", "HC"): ""}

# ------------------------------------------------------------------------------------------------
# 2. Sizes (inches) per figure function; a callable gets the original (w, h).
FIGSIZE = {
    "e10:fig_chua_basin": (W1, 2.75), "e10:fig_chua_survival": (W1, 2.35), "e10:fig_chua_dt": (W1, 2.6),
    "e10:fig_table1": (W2, 2.7), "e10:fig_uniform": lambda w, h: (W1, 0.155 * (h - 1.5) / 0.32 + 0.8),
    "e10:fig_synthetic": (W1, 3.0), "e10:fig_classic": lambda w, h: (W2, 0.3 * (h - 2.2) / 0.45 + 1.6),
    "e10:fig_gallery": (W2, 6.3),
    "e26:fig_pn": (W2, 6.0), "e26:fig_gain_and_table": (W2, 3.1), "e26:fig_scaling_p": (W2, 4.2),
    "e26:fig_scaling_n": (W1, 2.6), "e26:fig_ablation": (W2, 3.9), "e26:fig_signal": (W2, 2.8),
    "e35:fig_reconstruction": (W2, 5.6), "e35:fig_spurious": (W2, 4.5),
    "e35:fig_outcomes": (W2, 3.4), "e35:fig_fixed_points": (W2, 2.9), "e35:fig_survival": (W2, 2.7),
    "e35:fig_hidden": (W2, 3.3), "e35:fig_search": lambda w, h: (W1, 3.5) if w < 6 else (W2, 2.8),
    "e40:fig_chua_I": (W2, 3.3), "e40:fig_chua_II": (W2, 2.9), "e40:fig_gallery": (W2, 7.0),
    "e40:fig_li_sprott": (W2, 2.6), "e40:fig_maps": (W2, 2.3), "e40:fig_jafari": (W2, 2.4),
    "e40:fig_lorenz_transient": (W2, 2.9), "e40:fig_d2_mechanism": (W2, 2.7), "e40:fig_sus_chains": (W1, 3.1),
    "e40:fig_hiddenness": (W1, 3.3), "e40:fig_verification": lambda w, h: (W2, 0.17 * (h - 1.8) / 0.28 + 1.2),
}

# 3. Layout patches for the journal-size canvases: (script, old, new). They apply to the translated source
#    (English strings, explicit font sizes not yet mapped by _fs); each old string must occur exactly once.
PATCH = [
    # fig02: the note sat on the decaying curve
    ("e10", 'ax.text(0.02, 0.08, "DOP853, rtol = 10⁻¹¹: chaos', 'ax.text(0.3, 0.2, "DOP853, rtol = 10⁻¹¹:\\nchaos'),
    # fig03: the legend covered the F0 and escape curves
    ("e10", 'ax.legend(fontsize=8, ncol=2)', 'ax.legend(fontsize=7, ncol=2, loc="center", bbox_to_anchor=(0.58, 0.66))'),
    # fig21: drop the uninformative ": hidden attractor" suffix; legend into the empty grid slots
    ("e26", 'tg = {"chua_I": "both chaotic", "chua_II": "both chaotic", "jafari2026_lc": "cycle"}.get(name, "hidden attractor")\n'
            '        ax.set_title(f"{TITLE.get(name, name)}: {tg}", fontsize=9)',
            'tg = {"chua_I": "both chaotic", "chua_II": "both chaotic"}.get(name)\n'
            '        ax.set_title(f"{TITLE.get(name, name)}: {tg}" if tg else TITLE.get(name, name), fontsize=9)'),
    ("e26", 'fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))',
            'fig.legend(h, l, loc="center", ncol=1, fontsize=8, bbox_to_anchor=(0.67, 0.165))'),
    ("e26", 'fig.tight_layout(rect=(0, 0.07, 1, 0.97))\n    save(fig, FIG / "fig21_pn_levelAB")',
            'fig.tight_layout()\n    save(fig, FIG / "fig21_pn_levelAB")'),
    # fig22: two-line tick labels instead of overlapping ones
    ("e26", 'ax.set_xticklabels([TITLE.get(n, n) for n in names], fontsize=8)',
            'ax.set_xticklabels([TITLE.get(n, n).replace(" (", "\\n(").replace("Huynh map", "Huynh\\nmap") for n in names], fontsize=8)'),
    # fig05: p labels inside the axes, so that the figure keeps the column width
    ("e10", 'f"p = {r[1]:.3g}", va="center", fontsize=8, color=INK2)', 'f"p = {r[1]:.3g}", va="center", fontsize=7, color=INK2)'),
    ("e10", 'ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8)\n    ax.set_xscale("log"); ax.set_xlabel("$N_{0.9}$',
            'ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)\n    ax.set_xscale("log"); ax.set_xlim(0.6, 3e6); ax.set_xlabel("$N_{0.9}$'),
    ("e10", 'ax.grid(axis="y", visible=False)\n    save(fig, FIG / "fig05_uniform_costs")',
            'ax.grid(axis="y", visible=False)\n    fig.tight_layout()\n    save(fig, FIG / "fig05_uniform_costs")'),
    # fig23: legends below the panels (inside they covered the curves)
    ("e26", 'ax.legend(fontsize=6.5, loc="upper left")', 'ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1)'),
    # fig26: legend above the curves
    ("e26", 'axes[0].set_title("Do long transients approach the target?", fontsize=10)\n    axes[0].legend(fontsize=7)',
            'axes[0].set_title("Do long transients approach the target?", fontsize=10)\n'
            '    axes[0].set_ylim(0, 2.25)\n    axes[0].legend(fontsize=7, ncol=2, loc="upper center")'),
    # fig50: headroom for the VPT note and the legend
    ("e35", 'ax.text(v + 0.2, 17, f"VPT = {v:.1f} Lyap. times", fontsize=8, color=INK2)',
            'ax.text(v + 0.2, 24, f"VPT = {v:.1f} Lyap. times", fontsize=8, color=INK2); ax.set_ylim(-19, 30)'),
    ("e35", 'ax.set_title("Short-term forecast from the end of training", fontsize=10)\n    ax.legend(fontsize=8, loc="lower left")',
            'ax.set_title("Short-term forecast from the end of training", fontsize=10)\n    ax.legend(fontsize=7.5, loc="upper left", ncol=1)'),
    # fig51: one shared legend in the empty sixth slot instead of a legend over every 3D panel
    ("e35", '        ax.legend(fontsize=6, loc="upper left")\n    fig.suptitle(',
            '        pass\n    _h = {}\n    for _a in fig.axes:\n        for _hh, _ll in zip(*_a.get_legend_handles_labels()):\n'
            '            _h.setdefault(_ll, _hh)\n'
            '    fig.legend(list(_h.values()), list(_h.keys()), loc="center", bbox_to_anchor=(0.86, 0.25), fontsize=7.5)\n'
            '    fig.suptitle('),
    ("e35", 'fig.tight_layout()\n    save(fig, FIG / "fig51_esn_spurious")',
            'fig.tight_layout()\n    fig.subplots_adjust(hspace=0.25)\n    save(fig, FIG / "fig51_esn_spurious")'),
    # fig53: two-line legend entries that fit the empty upper right of the spectra panel; room for the
    # legend under the points of the (x, z) panel
    ("e35", 'label=f"{lab}: ρ = {r[\'fixed_points\'][i][\'rho\']:.4f}")', 'label=f"{lab}\\nρ = {r[\'fixed_points\'][i][\'rho\']:.4f}")'),
    ("e35", 'ax.legend(fontsize=7, loc="lower left")', 'ax.set_ylim(bottom=-82); ax.legend(fontsize=7, loc="lower left")'),
    # fig52: legends below the panels without collisions; the floor note under the floor
    ("e35", 'loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)', 'loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)'),
    ("e35", 'ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))',
            'ax.legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=0.6, handlelength=1.0)'),
    ("e35", 'ax.set_ylim(1.5e-3, 1.6)', 'ax.set_ylim(1.0e-3, 1.6)'),
    ("e35", 'ax.text(1.1e-3, 1.75e-3, "axis floor', 'ax.text(1.1e-3, 1.25e-3, "axis floor'),
    # fig53: the note on the dashed line sat on the markers
    ("e35", 'ax.text(-0.3, RE_C + 0.0012, "Lorenz: Re λ = 0.094", fontsize=8, color=INK2)',
            'ax.text(-0.3, 0.1255, "dashed: Lorenz, Re λ = 0.094", fontsize=7.5, color=INK2)'),
    # fig55: legend below the axes
    ("e35", 'ax.legend(fontsize=7, loc="upper left")', 'ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=1)'),
    # fig56: legends below the 3D panel and the basin slice
    ("e35", 'ax.legend(fontsize=6.5, loc="upper left")',
            'ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=0.6, handlelength=1.2)'),
    ("e35", 'fontsize=7, loc="upper right", frameon=True)',
            'fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)'),
    ("e35", 'ax2.set_position([pos.x0 + 0.03, pos.y0, pos.width - 0.03, pos.height])',
            'ax2.set_position([pos.x0 + 0.05, pos.y0, pos.width - 0.05, pos.height])'),
    # fig30: legend below the 3D panel; room for the 3D z label
    ("e40", 'ax.legend(fontsize=7, loc="upper left")\n    ax.set_title("Chua I: hidden attractors and equilibria", fontsize=10)',
            'ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, columnspacing=0.8, handlelength=1.5)\n'
            '    ax.set_title("Chua I: hidden attractors and equilibria", fontsize=10)'),
    ("e40", 'fig.tight_layout()\n    save(fig, FIG / "fig30_chua_I_hidden")',
            'fig.tight_layout()\n    _p = ax2.get_position(); ax2.set_position([_p.x0 + 0.04, _p.y0, _p.width - 0.04, _p.height])\n'
            '    save(fig, FIG / "fig30_chua_I_hidden")'),
    # fig31: legend below the 3D panel; more room for the z label of the zoom panel
    ("e40", 'ax.legend(fontsize=6.5, loc="upper left")',
            'ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, columnspacing=0.6, handlelength=1.2)'),
    ("e40", 'ax2.set_position([pos.x0 + 0.035, pos.y0, pos.width - 0.035, pos.height])',
            'ax2.set_position([pos.x0 + 0.06, pos.y0, pos.width - 0.06, pos.height])'),
    # fig32: tight_layout does not separate 3D panels; explicit spacing
    ("e40", 'fig.tight_layout()\n    save(fig, FIG / "fig32_gallery_3d")',
            'fig.subplots_adjust(left=0.0, right=0.97, bottom=0.03, top=0.97, wspace=0.12, hspace=0.45)\n'
            '    save(fig, FIG / "fig32_gallery_3d")'),
    # fig33: one legend below the panels with visible line samples
    ("e40", 'axes[0].legend(fontsize=7, markerscale=10)',
            '_lg = fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))\n'
            '    [_l.set_linewidth(2.0) for _l in _lg.get_lines()]'),
    ("e40", 'fig.tight_layout()\n    save(fig, FIG / "fig33_li_sprott_4d")',
            'fig.tight_layout(rect=(0, 0.09, 1, 1))\n    save(fig, FIG / "fig33_li_sprott_4d")'),
    # fig35: room for the z label of the 3D panel
    ("e40", 'ax.set_position([pos.x0 + 0.045, pos.y0, pos.width - 0.045, pos.height])',
            'ax.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])'),
    # fig40: legend below the axes
    ("e40", 'ax.legend(fontsize=7)\n    save(fig, FIG / "fig40_hiddenness")',
            'ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.45, -0.2), columnspacing=0.8)\n'
            '    save(fig, FIG / "fig40_hiddenness")'),
]


def _fs(x):
    """Explicit font sizes of the originals were set for 10-15 in canvases; at journal width 7-8.5 pt."""
    return min(max(float(x), 7.0), 8.5)


# ------------------------------------------------------------------------------------------------
CYR = re.compile(r"[А-Яа-яЁё]")
STATE = {"current": None}
SUPTITLES = []
SIZES = []


def translate(src, script):
    for old, new in sorted(TR.items(), key=lambda kv: -len(kv[0])):
        # TeX backslashes must survive the target string literal; "⏎" marks a line break
        src = src.replace(old, new.replace("\\", "\\\\").replace("⏎", "\\n"))
    for s, old, new in PATCH:
        if s != script:
            continue
        if src.count(old) != 1:
            raise RuntimeError(f"{script}: patch target found {src.count(old)} times: {old[:60]!r}")
        src = src.replace(old, new)
    src = re.sub(r"\b(fontsize|labelsize)=([0-9.]+)", r"\1=_fs(\2)", src)
    left = []
    for t in tokenize.generate_tokens(io.StringIO(src).readline):
        if t.type == tokenize.STRING and t.string.lstrip("rbfuRBFU")[:3] in ('"""', "'''"):
            continue
        if t.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)) and CYR.search(t.string):
            left.append(f"{script}:{t.start[0]}: {t.string}")
    if left:
        raise RuntimeError("untranslated strings:\n" + "\n".join(left))
    return src


MIN_FONT = 8.0   # AIP figure requirement: minimum label size 8-point type


def setup_en():
    plt.rcParams.update({
        "font.family": ["Arial", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": MIN_FONT, "ytick.labelsize": MIN_FONT, "legend.fontsize": MIN_FONT, "figure.titlesize": 9,
        "lines.linewidth": 1.5, "lines.markersize": 4, "axes.linewidth": 0.7, "grid.linewidth": 0.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.handlelength": 1.6,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.fallback": "stixsans",
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def install_hooks():
    # English labels in the shared helpers
    viz.NICE_SYS.update(NICE_SYS_EN)
    for k, v in list(viz.NICE_TGT.items()):
        viz.NICE_TGT[k] = TGT_EN.get(v, v)

    viz.NICE_TGT.update(TGT_EXTRA)

    def nice_label(name, target):
        """System name alone for its (only) hidden attractor "HA", "system: target" otherwise."""
        t = viz.NICE_TGT.get((name, target))
        if t is None:
            t = "" if target == "HA" else target
        sysname = viz.NICE_SYS.get(name, name)
        return f"{sysname}: {t}" if t else sysname
    viz.nice_label = nice_label

    orig_setup = viz.setup

    def setup():
        orig_setup()
        setup_en()
    viz.setup = setup

    def save(fig, path_without_ext):
        name = Path(str(path_without_ext)).name
        out = OUTDIRS.get((STATE["current"] or "").split(":")[0], OUT)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.pdf")
        fig.savefig(out / f"{name}.png")
        SIZES.append((name, STATE["current"], tuple(round(float(v), 2) for v in fig.get_size_inches())))
        plt.close(fig)
    viz.save = save

    orig_figure = plt.figure

    def figure(*args, **kw):
        spec = FIGSIZE.get(STATE["current"])
        if spec is not None:
            w, h = kw.get("figsize") or plt.rcParams["figure.figsize"]
            kw["figsize"] = spec(w, h) if callable(spec) else spec
        return orig_figure(*args, **kw)
    plt.figure = figure

    def suptitle(self, t, **kw):
        SUPTITLES.append((STATE["current"], t))
    FigureBase.suptitle = suptitle

    orig_tl = Figure.tight_layout

    def tight_layout(self, *, pad=1.08, h_pad=None, w_pad=None, rect=None):
        if rect is not None:
            rect = (rect[0], rect[1], rect[2], 1.0)      # the space was reserved for the dropped suptitle
        return orig_tl(self, pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
    Figure.tight_layout = tight_layout

    # AIP: minimum label size 8 pt. Explicit smaller font sizes in the scripts and patches are raised to 8.
    from matplotlib.axes import Axes

    def _floor8(method):
        def wrapped(self, *args, **kw):
            for key in ("fontsize", "size"):
                v = kw.get(key)
                if isinstance(v, (int, float)) and v < MIN_FONT:
                    kw[key] = MIN_FONT
            return method(self, *args, **kw)
        return wrapped
    for cls, names in ((Axes, ("legend", "text", "annotate", "set_title", "set_xlabel", "set_ylabel",
                               "set_xticklabels", "set_yticklabels", "tick_params")),
                       (FigureBase, ("legend", "text"))):
        for nm in names:
            setattr(cls, nm, _floor8(getattr(cls, nm)))

    # nothing is written into results/
    results = (ROOT / "results").resolve()
    orig_write_text = Path.write_text

    def write_text(self, *a, **k):
        p = Path(self).resolve()
        if results in p.parents or p == results:
            SIDE.mkdir(parents=True, exist_ok=True)
            p = SIDE / p.name
        return orig_write_text(p, *a, **k)
    Path.write_text = write_text
    orig_save = np.save

    def np_save(file, arr, *a, **k):
        p = Path(file).resolve()
        if results in p.parents:
            SIDE.mkdir(parents=True, exist_ok=True)
            file = SIDE / p.name
        return orig_save(file, arr, *a, **k)
    np.save = np_save


def run(script, only):
    path = ROOT / "experiments" / SCRIPTS[script]
    src = translate(path.read_text(encoding="utf-8"), script)
    ns = {"__name__": f"en_{script}", "__file__": str(path), "_fs": _fs}
    exec(compile(src, str(path), "exec"), ns)
    for name, f in list(ns.items()):
        if name.startswith("fig_") and callable(f):
            key = f"{script}:{name}"
            if only and key not in only:
                ns[name] = lambda *a, **k: None
                continue

            def wrapped(*a, _f=f, _key=key, **k):
                STATE["current"] = _key
                try:
                    return _f(*a, **k)
                finally:
                    STATE["current"] = None
            wrapped.__name__ = name
            ns[name] = wrapped
    ns["main"]()


def main(argv):
    if argv == ["--check"]:
        for script, fname in SCRIPTS.items():
            translate((ROOT / "experiments" / fname).read_text(encoding="utf-8"), script)
        print("translation complete")
        return
    esn = "--esn" in argv
    only = {a for a in argv if not a.startswith("--")}
    install_hooks()
    for script in SCRIPTS:
        if only and not any(o.startswith(script + ":") for o in only):
            continue
        if not only and script in OPTIONAL and not esn:
            continue
        print(f"== {script}", flush=True)
        run(script, only)
    # dropped suptitles, one line per figure function (a partial run updates only its own lines)
    log = BUILD / "suptitles.txt"
    lines = dict(line.split("\t", 1) for line in log.read_text(encoding="utf-8").splitlines() if "\t" in line) \
        if log.exists() else {}
    lines.update({key: t.replace("\n", " ") for key, t in SUPTITLES})
    log.write_text("".join(f"{k}\t{v}\n" for k, v in sorted(lines.items())), encoding="utf-8")
    for name, key, wh in SIZES:
        print(f"{name:28s} {key:28s} {wh}")


if __name__ == "__main__":
    main(sys.argv[1:])
