"""Personal scratch script: plot completion dates in data/raw/studies.jsonl.

Not part of the training pipeline. Run from the repo root:

    python scratch/plot_study_dates.py

Writes scratch/study_dates.html and prints a year histogram plus the dates
that would fall at the 80% and 90% marks if you split oldest -> newest.
"""

"""
PERSONAL NOTES: Make train 69.5% (1982 - 2021), val 13.7% (2022-2023), test 16.8% (2024 - 2026)
"""

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "studies.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "scratch" / "study_dates.html"


def parse_date(raw: str) -> date | None:
    """Accept YYYY-MM-DD and YYYY-MM. Month-only dates are pinned to day 1."""
    parts = raw.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2]) if len(parts) > 2 else 1
        return date(year, month, day)
    except (ValueError, IndexError):
        return None


def load_dates(path: Path) -> tuple[list[date], int, int]:
    parsed = []
    missing = 0
    month_only = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line).get("date")
            if not raw:
                missing += 1
                continue
            if len(raw) == 7:
                month_only += 1
            value = parse_date(raw)
            if value is None:
                missing += 1
                continue
            parsed.append(value)
    parsed.sort()
    return parsed, missing, month_only


def month_key(value: date) -> str:
    return f"{value.year:04d}-{value.month:02d}"


def percentile_date(sorted_dates: list[date], fraction: float) -> date:
    """Date at which `fraction` of studies (oldest first) have been included."""
    index = min(len(sorted_dates) - 1, max(0, int(len(sorted_dates) * fraction) - 1))
    return sorted_dates[index]


def text_histogram(year_counts: Counter, width: int = 40) -> str:
    peak = max(year_counts.values())
    lines = []
    for year in range(min(year_counts), max(year_counts) + 1):
        count = year_counts[year]
        bar = "#" * round(width * count / peak) if count else ""
        lines.append(f"{year}  {count:5d}  {bar}")
    return "\n".join(lines)


def render_html(
    dates: list[date],
    missing: int,
    month_only: int,
    year_counts: Counter,
    month_counts: Counter,
) -> str:
    years = list(range(min(year_counts), max(year_counts) + 1))
    year_values = [year_counts[year] for year in years]

    first_month = date(dates[0].year, dates[0].month, 1)
    last_month = date(dates[-1].year, dates[-1].month, 1)
    months = []
    cursor = first_month
    while cursor <= last_month:
        months.append(month_key(cursor))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)
    month_values = [month_counts[key] for key in months]

    running = 0
    cumulative = []
    total = len(dates)
    for count in year_values:
        running += count
        cumulative.append(round(100 * running / total, 2))

    p80 = percentile_date(dates, 0.8).isoformat()
    p90 = percentile_date(dates, 0.9).isoformat()

    payload = {
        "years": years,
        "year_counts": year_values,
        "cumulative_pct": cumulative,
        "months": months,
        "month_counts": month_values,
        "n": total,
        "missing": missing,
        "month_only": month_only,
        "min_date": dates[0].isoformat(),
        "max_date": dates[-1].isoformat(),
        "p80": p80,
        "p90": p90,
    }

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Study completion dates</title>
<style>
  :root {{
    color-scheme: light;
    --ink: #1c1917;
    --muted: #57534e;
    --line: #e7e5e4;
    --bar: #0f766e;
    --bar-hover: #115e59;
    --cum: #b45309;
  }}
  body {{
    margin: 0;
    font: 15px/1.45 "Iowan Old Style", Palatino, Georgia, serif;
    color: var(--ink);
    background: #fafaf9;
  }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }}
  h1 {{ font-weight: 600; font-size: 28px; margin: 0 0 8px; }}
  h2 {{ font-weight: 600; font-size: 18px; margin: 36px 0 8px; }}
  p {{ margin: 0 0 12px; color: var(--muted); }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0 8px; }}
  .stat {{
    background: white;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 10px 14px;
    min-width: 140px;
  }}
  .stat b {{ display: block; font-size: 18px; color: var(--ink); }}
  .stat span {{ color: var(--muted); font-size: 13px; }}
  .chart {{
    background: white;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 16px 12px 8px;
  }}
  .scroll {{ overflow-x: auto; }}
  svg {{ display: block; }}
  .tip {{
    position: fixed;
    pointer-events: none;
    background: #1c1917;
    color: white;
    padding: 6px 8px;
    border-radius: 6px;
    font: 12px/1.3 ui-sans-serif, system-ui, sans-serif;
    opacity: 0;
  }}
  table {{ border-collapse: collapse; width: 100%; background: white; }}
  th, td {{ text-align: right; padding: 4px 10px; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ font-size: 12px; color: var(--muted); font-weight: 600; }}
</style>
</head>
<body>
<main>
  <h1>Study completion dates</h1>
  <p>
    One bar per year of <code>completionDate</code> on the {payload["n"]:,} studies in
    <code>studies.jsonl</code>. The amber line is the cumulative share of studies up to
    that year. Dashed guides mark an 80/10/10 oldest-to-newest cut, only as a reference.
  </p>
  <div class="stats">
    <div class="stat"><b id="n"></b><span>studies with a date</span></div>
    <div class="stat"><b id="span"></b><span>earliest to latest</span></div>
    <div class="stat"><b id="month-only"></b><span>dates with no day (YYYY-MM)</span></div>
    <div class="stat"><b id="p80"></b><span>80% of studies are on or before</span></div>
    <div class="stat"><b id="p90"></b><span>90% of studies are on or before</span></div>
  </div>

  <h2>By year</h2>
  <div class="chart"><svg id="year-chart"></svg></div>

  <h2>By month</h2>
  <p>Scroll sideways. Early years are sparse; the mass sits after 2010.</p>
  <div class="chart scroll"><svg id="month-chart"></svg></div>

  <h2>Cumulative by year</h2>
  <div class="chart"><table id="year-table"></table></div>
</main>
<div class="tip" id="tip"></div>
<script>
const DATA = {json.dumps(payload)};

document.getElementById("n").textContent = DATA.n.toLocaleString();
document.getElementById("span").textContent = DATA.min_date + " → " + DATA.max_date;
document.getElementById("month-only").textContent = DATA.month_only.toLocaleString();
document.getElementById("p80").textContent = DATA.p80;
document.getElementById("p90").textContent = DATA.p90;

const tip = document.getElementById("tip");
function showTip(evt, text) {{
  tip.textContent = text;
  tip.style.opacity = 1;
  tip.style.left = (evt.clientX + 12) + "px";
  tip.style.top = (evt.clientY + 12) + "px";
}}
function hideTip() {{ tip.style.opacity = 0; }}

function barChart(svg, labels, values, {{ width, height, color, rotate }}) {{
  const pad = {{ top: 16, right: 16, bottom: rotate ? 64 : 36, left: 48 }};
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const peak = Math.max(...values, 1);
  const gap = 2;
  const barW = Math.max(1, (innerW - gap * labels.length) / labels.length);
  const ns = "http://www.w3.org/2000/svg";
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.innerHTML = "";

  const yTicks = 4;
  for (let i = 0; i <= yTicks; i++) {{
    const value = Math.round(peak * (1 - i / yTicks));
    const y = pad.top + innerH * (i / yTicks);
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", pad.left);
    line.setAttribute("x2", width - pad.right);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("stroke", "#e7e5e4");
    svg.appendChild(line);
    const text = document.createElementNS(ns, "text");
    text.setAttribute("x", pad.left - 8);
    text.setAttribute("y", y + 4);
    text.setAttribute("text-anchor", "end");
    text.setAttribute("font-size", "11");
    text.setAttribute("fill", "#57534e");
    text.textContent = value.toLocaleString();
    svg.appendChild(text);
  }}

  labels.forEach((label, i) => {{
    const h = innerH * (values[i] / peak);
    const x = pad.left + i * (barW + gap);
    const y = pad.top + innerH - h;
    const rect = document.createElementNS(ns, "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", barW);
    rect.setAttribute("height", Math.max(h, values[i] ? 1 : 0));
    rect.setAttribute("fill", color);
    rect.addEventListener("mousemove", (evt) => showTip(evt, label + ": " + values[i].toLocaleString()));
    rect.addEventListener("mouseleave", hideTip);
    svg.appendChild(rect);

    const showLabel = labels.length <= 50 || i % Math.ceil(labels.length / 24) === 0;
    if (showLabel) {{
      const text = document.createElementNS(ns, "text");
      text.setAttribute("x", x + barW / 2);
      text.setAttribute("y", pad.top + innerH + 14);
      text.setAttribute("text-anchor", rotate ? "end" : "middle");
      text.setAttribute("font-size", "11");
      text.setAttribute("fill", "#57534e");
      if (rotate) {{
        text.setAttribute("transform", `rotate(-55 ${{x + barW / 2}} ${{pad.top + innerH + 14}})`);
      }}
      text.textContent = label;
      svg.appendChild(text);
    }}
  }});
}}

const yearSvg = document.getElementById("year-chart");
barChart(yearSvg, DATA.years.map(String), DATA.year_counts, {{
  width: 1040, height: 360, color: "#0f766e", rotate: false,
}});

const ns = "http://www.w3.org/2000/svg";
const pad = {{ top: 16, right: 16, bottom: 36, left: 48 }};
const width = 1040;
const height = 360;
const innerW = width - pad.left - pad.right;
const innerH = height - pad.top - pad.bottom;
const cum = document.createElementNS(ns, "polyline");
const points = DATA.cumulative_pct.map((pct, i) => {{
  const x = pad.left + ((i + 0.5) / DATA.years.length) * innerW;
  const y = pad.top + innerH * (1 - pct / 100);
  return x + "," + y;
}});
cum.setAttribute("points", points.join(" "));
cum.setAttribute("fill", "none");
cum.setAttribute("stroke", "#b45309");
cum.setAttribute("stroke-width", "2");
yearSvg.appendChild(cum);

for (const [pct, label] of [[80, "80%"], [90, "90%"]]) {{
  const y = pad.top + innerH * (1 - pct / 100);
  const line = document.createElementNS(ns, "line");
  line.setAttribute("x1", pad.left);
  line.setAttribute("x2", width - pad.right);
  line.setAttribute("y1", y);
  line.setAttribute("y2", y);
  line.setAttribute("stroke", "#b45309");
  line.setAttribute("stroke-dasharray", "4 4");
  yearSvg.appendChild(line);
  const text = document.createElementNS(ns, "text");
  text.setAttribute("x", width - pad.right);
  text.setAttribute("y", y - 4);
  text.setAttribute("text-anchor", "end");
  text.setAttribute("font-size", "11");
  text.setAttribute("fill", "#b45309");
  text.textContent = label;
  yearSvg.appendChild(text);
}}

barChart(document.getElementById("month-chart"), DATA.months, DATA.month_counts, {{
  width: Math.max(1040, DATA.months.length * 8),
  height: 320,
  color: "#0f766e",
  rotate: true,
}});

const table = document.getElementById("year-table");
let html = "<tr><th>Year</th><th>Studies</th><th>Cumulative</th><th>Cumulative %</th></tr>";
let running = 0;
DATA.years.forEach((year, i) => {{
  running += DATA.year_counts[i];
  html += `<tr><td>${{year}}</td><td>${{DATA.year_counts[i].toLocaleString()}}</td><td>${{running.toLocaleString()}}</td><td>${{DATA.cumulative_pct[i].toFixed(1)}}%</td></tr>`;
}});
table.innerHTML = html;
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot completion dates in studies.jsonl")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    dates, missing, month_only = load_dates(args.input)
    if not dates:
        raise SystemExit(f"No parseable dates in {args.input}")

    year_counts = Counter(value.year for value in dates)
    month_counts = Counter(month_key(value) for value in dates)
    p80 = percentile_date(dates, 0.8)
    p90 = percentile_date(dates, 0.9)

    print(f"studies with a date: {len(dates)}")
    print(f"unparsed or missing: {missing}")
    print(f"month precision only (YYYY-MM): {month_only}")
    print(f"range: {dates[0].isoformat()} -> {dates[-1].isoformat()}")
    print(f"80% of studies fall on or before {p80.isoformat()}  (newest 20% after this)")
    print(f"90% of studies fall on or before {p90.isoformat()}  (newest 10% after this)")
    print()
    print(text_histogram(year_counts))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_html(dates, missing, month_only, year_counts, month_counts),
        encoding="utf-8",
    )
    print()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
