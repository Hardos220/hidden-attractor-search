"""Вставляет строки таблиц статьи (results/e53_snippets/tab_cost.{tex,html} — e54_cost_figure.py; tab_time.{tex,html} —
e59_paper_table.py; tab_newattr.{tex,html} — e62_newattr_table.py) в обе версии статьи между метками
TAB<ИМЯ>-BEGIN и TAB<ИМЯ>-END. Файлы без меток и отсутствующие сниппеты пропускаются."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SN = ROOT / "results" / "e53_snippets"
TEX = ROOT / "paper" / "chaos" / "sections" / "50_results.tex"
HTML = ROOT / "paper" / "src" / "50_results.html"
for name in ("COST", "TIME", "NEWATTR"):
    for path, rows, beg, end in ((TEX, SN / f"tab_{name.lower()}.tex", f"%TAB{name}-BEGIN\n", f"%TAB{name}-END\n"),
                                 (HTML, SN / f"tab_{name.lower()}.html", f"<!--TAB{name}-BEGIN-->\n", f"<!--TAB{name}-END-->")):
        if not rows.exists():
            continue
        s = path.read_text(encoding="utf-8")
        if beg not in s or end not in s:
            continue
        i, j = s.index(beg) + len(beg), s.index(end)
        s = s[:i] + rows.read_text(encoding="utf-8") + s[j:]
        path.write_text(s, encoding="utf-8")
        print("->", path, name)
