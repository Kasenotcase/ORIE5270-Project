#!/usr/bin/env python3
"""Generate the Zoom progress HTML report from project artifacts.

The report intentionally reads performance numbers from existing CSV/JSON
outputs instead of hard-coding empirical results in the HTML.
"""

import csv
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
OUT = DOCS / "zoom_progress_report.html"

TABLE_DIR = DOCS / "04_empirical_section" / "tables"
V3_DIR = ROOT / "improvements" / "06_v3_rolling_regime_mpc" / "results"


SOURCES = {
    "data_quality": TABLE_DIR / "tab_01_data_quality.csv",
    "original_perf": TABLE_DIR / "tab_03_original_strategy_performance.csv",
    "full_perf": TABLE_DIR / "tab_04_v3_full_sample_performance.csv",
    "excess_perf": TABLE_DIR / "tab_05_v3_excess_performance.csv",
    "selection": TABLE_DIR / "tab_07_v3_selection_summary.csv",
    "avg_weights": TABLE_DIR / "tab_08_v3_average_weights_by_regime.csv",
    "forecast": TABLE_DIR / "tab_10_forecast_diagnostics.csv",
    "manifest": V3_DIR / "manifest.json",
    "candidate_rebalances": V3_DIR / "tables" / "candidate_rebalance_records.csv",
    "selected_rebalances": V3_DIR / "tables" / "selected_params_over_time.csv",
}


# Formula source: docs/04_empirical_section.md, Sections 5.1-5.4.
MOMENTUM_FORMULA = r"""
m_{i,t} =
\frac{1}{231}
\sum_{j=22}^{252}\log(1+r_{i,t-j})
"""

ALPHA_FORMULA = r"""
\alpha_t =
\operatorname{clip}\left(
\alpha_0
\left[
1 - 0.20q_t^{\text{tr}} - 0.65q_t^{\text{st}}
\right],
0.10,
\alpha_0
\right)
"""

SCORE_FORMULA = r"""
\text{Score}_{k,t}=
\text{Sharpe}_{k,t}
+0.40\,\text{IR}_{k,t}^{\text{EW}}
+2.00\,\text{ActiveReturn}_{k,t}^{\text{EW}}
+0.15\,\text{DDImprove}_{k,t}^{\text{EW}}
"""

MPC_OBJECTIVE = r"""
\max_{\{x_h,b_h,s_h\}_{h=1}^{H}}
\sum_{h=1}^{H}
\left[
\hat{\mu}_t^\top x_h
-\frac{\gamma}{2}x_h^\top\hat{\Sigma}_t x_h
-a_t\mathbf{1}^\top(b_h+s_h)
-\frac{\kappa}{2}\|b_h-s_h\|_2^2
\right]
"""

MPC_CONSTRAINTS = r"""
\mathbf{1}^\top x_h=1,\quad
0 \le x_h \le 0.60,\quad
x_h-x_{h-1}=b_h-s_h,\quad
\frac{1}{2}\mathbf{1}^\top(b_h+s_h)\le \bar{\tau}
"""


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_manifest(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def count_data_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def esc(value):
    return html.escape(str(value), quote=True)


def as_float(value):
    return float(value)


def pct(value, digits=2):
    return f"{as_float(value) * 100:.{digits}f}%"


def num(value, digits=3):
    return f"{as_float(value):.{digits}f}"


def intish(value):
    return f"{int(round(as_float(value))):,}"


def source_badge(label):
    return f'<span class="source">Source: {esc(label)}</span>'


def table(headers, rows, class_name=""):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{cell}</td>" for cell in row)
        body.append(f"<tr>{cells}</tr>")
    cls = f' class="{class_name}"' if class_name else ""
    return f'<div class="table-wrap"><table{cls}><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def image_block(src, alt, caption, source):
    return f"""
    <figure>
      <img src="{esc(src)}" alt="{esc(alt)}" loading="lazy">
      <figcaption>{esc(caption)} {source_badge(source)}</figcaption>
    </figure>
    """


def metric(label, value, note=""):
    note_html = f"<small>{esc(note)}</small>" if note else ""
    return f"""
    <div class="metric">
      <span>{esc(label)}</span>
      <strong>{value}</strong>
      {note_html}
    </div>
    """


def key_value_rows(data_quality):
    labels = [
        ("Asset count", "asset_count", intish),
        ("Raw price start", "raw_price_start", str),
        ("Raw price end", "raw_price_end", str),
        ("Clean return start", "clean_return_start", str),
        ("Clean return end", "clean_return_end", str),
        ("Return observations", "return_observations", intish),
        ("Macro missing after alignment", "macro_missing_after_alignment", intish),
        ("Rolling covariance condition number, median", "rolling_cov_condition_median", lambda x: num(x, 2)),
        ("Rolling covariance condition number, max", "rolling_cov_condition_max", lambda x: num(x, 2)),
    ]
    rows = []
    for label, key, formatter in labels:
        value = data_quality[key]
        rows.append([esc(label), esc(formatter(value))])
    return rows


def performance_rows(rows):
    display_names = {
        "Equal Weight": "Equal Weight",
        "Markowitz": "Markowitz CVXPY",
        "Original HMM-MPC": "Original HMM-MPC",
        "V3 Rolling Regime-MPC": "V3 Rolling Regime-MPC",
    }
    out = []
    for row in rows:
        out.append([
            esc(display_names.get(row["Strategy"], row["Strategy"])),
            esc(pct(row["Annual Return"])),
            esc(pct(row["Annual Volatility"])),
            esc(num(row["Sharpe"], 3)),
            esc(pct(row["Max Drawdown"])),
            esc(num(row["Final Wealth"], 3)),
            esc(pct(row["Avg Rebalance Turnover"])),
            esc(pct(row["Total Cost"])),
        ])
    return out


def compact_performance_rows(rows):
    out = []
    for row in rows:
        out.append([
            esc(row["Strategy"]),
            esc(pct(row["Annual Return"])),
            esc(num(row["Sharpe"], 3)),
            esc(pct(row["Max Drawdown"])),
            esc(num(row["Final Wealth"], 3)),
            esc(pct(row["Avg Rebalance Turnover"])),
        ])
    return out


def excess_rows(rows):
    readable = {
        "V3_Rolling_Regime_MPC minus EqualWeight_weekly": "V3 minus Equal Weight",
        "V3_Rolling_Regime_MPC minus Markowitz_CVXPY_weekly": "V3 minus Markowitz",
        "V3_Rolling_Regime_MPC minus HMM_MPC_CVXPY_weekly": "V3 minus Original HMM-MPC",
    }
    out = []
    for row in rows:
        out.append([
            esc(readable.get(row["comparison"], row["comparison"])),
            esc(pct(row["annualized_active_return"])),
            esc(num(row["information_ratio"], 3)),
            esc(pct(row["annualized_return_difference"])),
            esc(num(row["sharpe_difference"], 3)),
            esc(num(row["final_wealth_ratio"], 3) + "x"),
        ])
    return out


def selection_rows(rows, limit=6):
    out = []
    for row in rows[:limit]:
        out.append([
            esc(row["selected_candidate"]),
            esc(num(row["gamma"], 2)),
            esc(num(row["base_alpha"], 2)),
            esc(pct(row["turnover_cap"], 0)),
            esc(intish(row["selection_count"])),
        ])
    return out


def top_weights_by_regime(rows):
    out = []
    for row in rows:
        regime = row["regime"]
        weights = []
        for key, value in row.items():
            if key == "regime":
                continue
            weights.append((key, as_float(value)))
        weights.sort(key=lambda item: item[1], reverse=True)
        top = ", ".join(f"{ticker} {weight * 100:.1f}%" for ticker, weight in weights[:3])
        out.append([esc(regime.title()), esc(top)])
    return out


def forecast_rows(rows):
    selected = [row for row in rows if row["rolling_window"] in {"252", "504"} and row["future_horizon_days"] in {"1", "21"}]
    out = []
    for row in selected:
        out.append([
            esc(row["rolling_window"]),
            esc(row["future_horizon_days"]),
            esc(num(row["mean_spearman_ic"], 3)),
            esc(num(row["median_spearman_ic"], 3)),
            esc(pct(row["sign_hit_rate"])),
            esc(pct(row["annualized_top_minus_bottom_realized"])),
        ])
    return out


def pct_pp(value, digits=2):
    return f"{as_float(value) * 100:.{digits}f} pp"


def build_report():
    for name, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required source for {name}: {path}")

    data_quality_rows = read_csv_rows(SOURCES["data_quality"])
    data_quality = {row["check"]: row["value"] for row in data_quality_rows}
    original_perf = read_csv_rows(SOURCES["original_perf"])
    full_perf = read_csv_rows(SOURCES["full_perf"])
    excess_perf = read_csv_rows(SOURCES["excess_perf"])
    selection = read_csv_rows(SOURCES["selection"])
    avg_weights = read_csv_rows(SOURCES["avg_weights"])
    forecast = read_csv_rows(SOURCES["forecast"])
    manifest = read_manifest(SOURCES["manifest"])

    selected_rebalances = count_data_rows(SOURCES["selected_rebalances"])
    candidate_rebalances = count_data_rows(SOURCES["candidate_rebalances"])
    v3 = next(row for row in full_perf if row["Strategy"] == "V3 Rolling Regime-MPC")
    ew = next(row for row in full_perf if row["Strategy"] == "Equal Weight")
    hmm = next(row for row in full_perf if row["Strategy"] == "Original HMM-MPC")
    v3_vs_ew = next(row for row in excess_perf if "EqualWeight" in row["comparison"])
    v3_vs_hmm = next(row for row in excess_perf if "HMM_MPC" in row["comparison"])

    data_table = table(["Check", "Value"], key_value_rows(data_quality), "mini-table")
    initial_table = table(
        ["Strategy", "Ann. Return", "Sharpe", "Max Drawdown", "Final Wealth", "Rebalance Turnover"],
        compact_performance_rows(original_perf),
    )
    full_table = table(
        [
            "Strategy",
            "Ann. Return",
            "Ann. Vol.",
            "Sharpe",
            "Max Drawdown",
            "Final Wealth",
            "Rebalance Turnover",
            "Total Cost",
        ],
        performance_rows(full_perf),
    )
    excess_table = table(
        ["Comparison", "Active Return", "IR", "Return Diff.", "Sharpe Diff.", "Final Wealth Ratio"],
        excess_rows(excess_perf),
    )
    selection_table = table(
        ["Candidate", "Gamma", "Base Alpha", "Turnover Cap", "Selections"],
        selection_rows(selection),
        "mini-table",
    )
    regime_weight_table = table(["Regime", "Largest Average Weights"], top_weights_by_regime(avg_weights), "mini-table")
    forecast_table = table(
        ["Window", "Horizon", "Mean IC", "Median IC", "Sign Hit Rate", "Top-Bottom Realized"],
        forecast_rows(forecast),
        "mini-table",
    )

    source_list = [
        "docs/04_empirical_section/tables/tab_01_data_quality.csv",
        "docs/04_empirical_section/tables/tab_03_original_strategy_performance.csv",
        "docs/04_empirical_section/tables/tab_04_v3_full_sample_performance.csv",
        "docs/04_empirical_section/tables/tab_05_v3_excess_performance.csv",
        "docs/04_empirical_section/tables/tab_07_v3_selection_summary.csv",
        "docs/04_empirical_section/tables/tab_08_v3_average_weights_by_regime.csv",
        "docs/04_empirical_section/tables/tab_10_forecast_diagnostics.csv",
        "improvements/06_v3_rolling_regime_mpc/results/manifest.json",
        "improvements/06_v3_rolling_regime_mpc/results/figures/v3_monthly_weights.png",
        "improvements/06_v3_rolling_regime_mpc/results/tables/candidate_rebalance_records.csv",
        "improvements/06_v3_rolling_regime_mpc/results/tables/selected_params_over_time.csv",
    ]
    source_html = "".join(f"<li><code>{esc(path)}</code></li>" for path in source_list)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Regime-Aware MPC Portfolio Optimization | Progress Brief</title>
  <script>
    window.MathJax = {{
      tex: {{inlineMath: [['\\\\(', '\\\\)'], ['$', '$']], displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']]}},
      svg: {{fontCache: 'global'}}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {{
      --ink: #202124;
      --muted: #5f6662;
      --line: #d9dfdc;
      --panel: #ffffff;
      --soft: #f5f7f6;
      --accent: #0f7b6c;
      --accent-dark: #0a4f46;
      --warn: #a15c00;
      --bad: #a33a2b;
    }}

    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--soft);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
      letter-spacing: 0;
    }}

    a {{ color: var(--accent-dark); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}

    .topbar {{
      position: sticky;
      top: 0;
      z-index: 50;
      display: flex;
      gap: 18px;
      align-items: center;
      justify-content: space-between;
      padding: 12px 28px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.96);
      backdrop-filter: blur(10px);
    }}

    .brand {{
      font-size: 13px;
      font-weight: 760;
      letter-spacing: 0;
      color: var(--accent-dark);
      white-space: nowrap;
    }}

    nav {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}

    nav a {{
      padding: 6px 9px;
      border-radius: 6px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 650;
    }}

    nav a:hover {{
      background: #e9f2ef;
      color: var(--accent-dark);
      text-decoration: none;
    }}

    main {{ width: 100%; }}

    section {{
      min-height: 92vh;
      display: flex;
      align-items: center;
      padding: 72px 28px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}

    section:nth-child(even) {{ background: var(--soft); }}

    .wrap {{
      width: min(1180px, 100%);
      margin: 0 auto;
    }}

    .eyebrow {{
      margin: 0 0 8px;
      color: var(--accent-dark);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
      line-height: 1.08;
    }}

    h1 {{
      max-width: 1040px;
      font-size: clamp(38px, 6vw, 74px);
      font-weight: 820;
    }}

    h2 {{
      font-size: clamp(30px, 4vw, 52px);
      font-weight: 800;
    }}

    h3 {{
      font-size: 19px;
      font-weight: 780;
    }}

    .lead {{
      max-width: 850px;
      margin: 18px 0 0;
      color: var(--muted);
      font-size: clamp(18px, 2vw, 23px);
    }}

    .grid {{
      display: grid;
      gap: 20px;
    }}

    .grid > * {{
      min-width: 0;
    }}

    .two {{ grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr); }}
    .three {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .four {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}

    .mt-24 {{ margin-top: 24px; }}
    .mt-32 {{ margin-top: 32px; }}

    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 20px;
      box-shadow: 0 14px 28px rgba(32,33,36,0.06);
      min-width: 0;
      overflow: hidden;
    }}

    .plain {{
      border: 0;
      background: transparent;
      box-shadow: none;
      padding: 0;
    }}

    .pill-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 26px;
    }}

    .pill {{
      padding: 8px 12px;
      border: 1px solid #b9d7cf;
      border-radius: 6px;
      background: #eef7f4;
      color: var(--accent-dark);
      font-size: 13px;
      font-weight: 760;
    }}

    .pipeline {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-top: 36px;
    }}

    .step {{
      min-height: 94px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      font-weight: 760;
      font-size: 14px;
    }}

    .metric-row {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 26px;
    }}

    .metric {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 16px;
    }}

    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}

    .metric strong {{
      display: block;
      margin-top: 7px;
      font-size: clamp(22px, 3.5vw, 36px);
      line-height: 1;
      color: var(--accent-dark);
    }}

    .metric small {{
      display: block;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
    }}

    ul {{
      margin: 18px 0 0;
      padding-left: 21px;
    }}

    li {{ margin: 8px 0; }}

    .short-list li {{
      font-size: 16px;
    }}

    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
      box-shadow: 0 14px 28px rgba(32,33,36,0.06);
    }}

    figure img {{
      width: 100%;
      display: block;
      background: #fff;
    }}

    figcaption {{
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      border-radius: 8px;
      background: #fff;
      font-size: 14px;
    }}

    .table-wrap {{
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      border-radius: 8px;
    }}

    .table-wrap table {{
      min-width: max-content;
    }}

    th, td {{
      padding: 10px 11px;
      border-bottom: 1px solid var(--line);
      text-align: right;
      vertical-align: top;
      white-space: nowrap;
    }}

    th:first-child, td:first-child {{ text-align: left; }}
    th {{
      background: #e9f2ef;
      color: var(--accent-dark);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    tbody tr:last-child td {{ border-bottom: 0; }}

    .mini-table {{
      font-size: 13px;
    }}

    .formula {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 14px 16px;
      margin-top: 16px;
    }}

    .formula .source {{
      margin-top: 10px;
    }}

    .panel .formula {{
      border: 0;
      border-radius: 0;
      background: transparent;
      padding: 0;
    }}

    .source {{
      display: inline-block;
      margin-left: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 650;
    }}

    .note {{
      margin-top: 16px;
      padding: 12px 14px;
      border-left: 4px solid var(--accent);
      background: #eef7f4;
      color: var(--accent-dark);
      border-radius: 6px;
      font-weight: 650;
    }}

    .warning {{
      border-left-color: var(--warn);
      background: #fff7e8;
      color: #6c3e00;
    }}

    .negative {{ color: var(--bad); font-weight: 760; }}
    .positive {{ color: var(--accent-dark); font-weight: 760; }}

    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}

    .figure-grid figure img {{
      aspect-ratio: 1.55 / 1;
      object-fit: contain;
    }}

    code {{
      padding: 2px 5px;
      border-radius: 4px;
      background: #eef1f0;
      color: #303633;
      font-size: 0.92em;
    }}

    .small {{
      color: var(--muted);
      font-size: 13px;
    }}

    @media (max-width: 980px) {{
      .two, .three, .four, .metric-row, .pipeline, .figure-grid {{
        grid-template-columns: 1fr;
      }}

      section {{
        min-height: auto;
        padding: 56px 18px;
      }}

      .topbar {{
        position: relative;
        align-items: flex-start;
        flex-direction: column;
        padding: 12px 18px;
      }}

      nav {{ justify-content: flex-start; }}
      th, td {{ white-space: normal; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand">ORIE 5370 Progress Brief</div>
    <nav>
      <a href="#snapshot">Snapshot</a>
      <a href="#data">Data</a>
      <a href="#regime">Regime</a>
      <a href="#model">Model</a>
      <a href="#framework">Framework</a>
      <a href="#baselines">Baselines</a>
      <a href="#v3">V3</a>
      <a href="#v3-weights">Weights</a>
      <a href="#results">Results</a>
      <a href="#diagnostics">Diagnostics</a>
      <a href="#status">Status</a>
    </nav>
  </header>

  <main>
    <section id="snapshot">
      <div class="wrap">
        <p class="eyebrow">Zoom progress report</p>
        <h1>Regime-aware multi-period portfolio optimization solved with CVXPY.</h1>
        <p class="lead">A rolling empirical study of U.S. sector ETF allocation that combines market regime estimation, convex model predictive trading, realistic transaction costs, and benchmark-driven diagnostics.</p>
        <div class="pill-row">
          <span class="pill">Optimization model</span>
          <span class="pill">Multi-period trading</span>
          <span class="pill">Market regime</span>
          <span class="pill">CVXPY convex program</span>
        </div>
        <div class="pipeline">
          <div class="step">Clean sector ETF data</div>
          <div class="step">Estimate regime probabilities</div>
          <div class="step">Build return and risk inputs</div>
          <div class="step">Solve MPC with CVXPY</div>
          <div class="step">Backtest against baselines</div>
          <div class="step">Diagnose and improve V3</div>
        </div>
        <div class="metric-row">
          {metric("Asset universe", intish(data_quality["asset_count"]), "Select Sector SPDR ETFs")}
          {metric("Backtest period", f'{esc(manifest["period"][0])}<br>to {esc(manifest["period"][1])}', "from V3 manifest")}
          {metric("V3 annual return", pct(v3["Annual Return"]), "full sample")}
          {metric("V3 Sharpe", num(v3["Sharpe"], 3), "net of trading costs")}
        </div>
        <p class="small mt-24">All empirical numbers in this report are generated from the local project CSV/JSON artifacts listed in the final source section.</p>
      </div>
    </section>

    <section id="question">
      <div class="wrap grid two">
        <div>
          <p class="eyebrow">Research question</p>
          <h2>Can regime information improve a realistic convex trading strategy?</h2>
          <p class="lead">The project tests whether a regime-aware multi-period optimizer can outperform both a simple investable benchmark and a classical mean-variance optimizer under rolling, no-look-ahead conditions.</p>
        </div>
        <div class="panel">
          <h3>Meeting-level takeaways</h3>
          <ul class="short-list">
            <li>The initial HMM-MPC model was theoretically aligned with the project, but it underperformed equal weight.</li>
            <li>The diagnosis pointed to weak rolling-mean return forecasts and excessive defensive tilts.</li>
            <li>V3 keeps the convex MPC structure, but improves the return input and selects parameters through a rolling benchmark-aware rule.</li>
            <li>V3 finishes above all baselines in annual return and Sharpe in the full sample.</li>
          </ul>
        </div>
      </div>
    </section>

    <section id="data">
      <div class="wrap grid two">
        <div>
          <p class="eyebrow">Data</p>
          <h2>Moderate, liquid, and reproducible asset universe.</h2>
          <p class="lead">The empirical universe is the nine long-history Select Sector SPDR ETFs: <code>XLB</code>, <code>XLE</code>, <code>XLF</code>, <code>XLI</code>, <code>XLK</code>, <code>XLP</code>, <code>XLU</code>, <code>XLV</code>, and <code>XLY</code>.</p>
          <ul class="short-list">
            <li>Daily adjusted prices, close prices, and volume come from the project Yahoo Finance data pipeline.</li>
            <li>Regime features also use FRED <code>VIXCLS</code> and <code>BAMLH0A0HYM2</code>.</li>
            <li>The ETF universe reduces survivorship issues, keeps covariance estimation stable, and remains directly tradable.</li>
          </ul>
          <div class="panel mt-24">
            {data_table}
            {source_badge("docs/04_empirical_section/tables/tab_01_data_quality.csv")}
          </div>
        </div>
        {image_block("../outputs/figures/asset_cumulative_returns.png", "Asset cumulative returns", "Sector ETF cumulative returns used to understand cross-sectional return dispersion.", "outputs/figures/asset_cumulative_returns.png")}
      </div>
    </section>

    <section id="regime">
      <div class="wrap grid two">
        <div>
          <p class="eyebrow">Market regime</p>
          <h2>HMM probabilities enter the optimizer, not a separate trading rule.</h2>
          <p class="lead">The regime layer estimates latent market states from daily features and passes filtered probabilities into return, covariance, and trading-cost inputs.</p>
          <ul class="short-list">
            <li>Features are built with rolling, past-information-only transformations.</li>
            <li>Regimes are interpreted as calm, transition, and stress states.</li>
            <li>Stress probabilities shrink alpha and can increase risk or cost inputs.</li>
          </ul>
          <div class="formula">
            $$q_{{t+1}} = \\Pr(z_{{t+1}} \\mid \\mathcal{{F}}_t)$$
            {source_badge("docs/03_market_regime_design.md")}
          </div>
        </div>
        {image_block("../outputs/figures/hmm_regime_probabilities.png", "HMM regime probabilities", "Filtered HMM regime probabilities over the empirical sample.", "outputs/figures/hmm_regime_probabilities.png")}
      </div>
    </section>

    <section id="model">
      <div class="wrap">
        <p class="eyebrow">Optimization model</p>
        <h2>Finite-horizon convex MPC, execute the first trade only.</h2>
        <div class="grid two mt-32">
          <div class="formula">
            $${MPC_OBJECTIVE}$$
            {source_badge("docs/04_empirical_section.md, Section 5.4")}
          </div>
          <div>
            <div class="formula">
              $${MPC_CONSTRAINTS}$$
              {source_badge("docs/04_empirical_section.md, Section 5.4")}
            </div>
            <ul class="short-list">
              <li>Long-only and fully invested sector weights.</li>
              <li>Single-asset cap is 60% in the final empirical model.</li>
              <li>Transaction costs are included directly in the objective.</li>
              <li>Problems are solved through CVXPY with OSQP as the first solver.</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <section id="framework">
      <div class="wrap grid two">
        <div>
          <p class="eyebrow">Training and backtest framework</p>
          <h2>No future information is used at the decision date.</h2>
          <p class="lead">At each weekly rebalance, the strategy estimates inputs using information available no later than date <code>t</code>, solves the convex program, and applies the first trade on the next trading day.</p>
          <div class="metric-row">
            {metric("V3 rebalances", f'{selected_rebalances:,}', "rows in selected_params_over_time.csv")}
            {metric("Candidate rebalances", f'{candidate_rebalances:,}', "rows in candidate_rebalance_records.csv")}
            {metric("Candidate count", f'{manifest["candidate_count"]}', "from manifest.json")}
            {metric("Selection window", f'{manifest["selection_window"]} days', "from manifest.json")}
          </div>
          <div class="note mt-24">This structure matches the real investment chronology: estimate, choose, trade, then observe realized performance.</div>
        </div>
        <div class="panel">
          <h3>Leakage controls from the V3 manifest</h3>
          <ul class="short-list">
            {"".join(f"<li>{esc(item.capitalize())}.</li>" for item in manifest["no_future_leakage_controls"])}
          </ul>
          {source_badge("improvements/06_v3_rolling_regime_mpc/results/manifest.json")}
        </div>
      </div>
    </section>

    <section id="baselines">
      <div class="wrap">
        <p class="eyebrow">Baselines and initial model</p>
        <h2>The first HMM-MPC was useful, but not enough.</h2>
        <p class="lead">Equal weight is the simple investable benchmark. Markowitz CVXPY is the classical optimization benchmark. The original HMM-MPC satisfies the regime-aware multi-period convex optimization requirement, but it does not beat equal weight.</p>
        <div class="panel mt-32">
          {initial_table}
          {source_badge("docs/04_empirical_section/tables/tab_03_original_strategy_performance.csv")}
        </div>
        <div class="note warning">Interpretation for the meeting: the initial model improved over Markowitz in return and Sharpe, but lagged equal weight because its return forecasts were too weak and its allocation became too defensive.</div>
      </div>
    </section>

    <section id="diagnosis">
      <div class="wrap grid two">
        <div>
          <p class="eyebrow">Diagnosis</p>
          <h2>The problem was the input design, not the convex program itself.</h2>
          <ul class="short-list">
            <li>Rolling sample means had low cross-sectional forecasting power.</li>
            <li>Small expected-return errors created large portfolio tilts.</li>
            <li>The initial HMM-MPC concentrated in defensive sectors and missed several growth-led or cyclical rallies.</li>
            <li>A fixed parameter choice can lock in a market style from one historical period.</li>
          </ul>
          <div class="panel mt-24">
            {forecast_table}
            {source_badge("docs/04_empirical_section/tables/tab_10_forecast_diagnostics.csv")}
          </div>
        </div>
        {image_block("../outputs/figures/hmm_mpc_weights.png", "Original HMM-MPC weights", "Initial model weights, showing the defensive allocation pattern that motivated the improvement stage.", "outputs/figures/hmm_mpc_weights.png")}
      </div>
    </section>

    <section id="v3">
      <div class="wrap">
        <p class="eyebrow">Final model</p>
        <h2>V3 rolling regime-MPC keeps the optimization structure and improves the inputs.</h2>
        <div class="grid three mt-32">
          <div class="panel">
            <h3>Momentum return input</h3>
            <div class="formula">$$ {MOMENTUM_FORMULA} $$</div>
            <p class="small">12-1 style sector momentum replaces noisy raw rolling means.</p>
          </div>
          <div class="panel">
            <h3>Regime-dependent alpha</h3>
            <div class="formula">$$ {ALPHA_FORMULA} $$</div>
            <p class="small">Calm regimes retain more alpha; stress regimes shrink it.</p>
          </div>
          <div class="panel">
            <h3>Rolling candidate selection</h3>
            <div class="formula">$$ {SCORE_FORMULA} $$</div>
            <p class="small">The score rewards total performance and performance relative to equal weight.</p>
          </div>
        </div>
        <div class="panel mt-24">
          {selection_table}
          {source_badge("docs/04_empirical_section/tables/tab_07_v3_selection_summary.csv")}
        </div>
      </div>
    </section>

    <section id="v3-weights">
      <div class="wrap grid two">
        <div>
          <p class="eyebrow">V3 weight dynamics</p>
          <h2>The final model changes sector exposure through time.</h2>
          <p class="lead">This figure shows the monthly V3 sector weights generated by the rolling regime-MPC experiment.</p>
          <ul class="short-list">
            <li>The weights are produced by the V3 optimization run, not manually reconstructed.</li>
            <li>The plot helps explain how the strategy moves across sector exposures rather than only reporting final performance.</li>
            <li>It complements the regime average-weight diagnostics shown later in the report.</li>
          </ul>
        </div>
        {image_block("../improvements/06_v3_rolling_regime_mpc/results/figures/v3_monthly_weights.png", "V3 monthly weights", "Monthly sector weights from the V3 rolling regime-MPC experiment.", "improvements/06_v3_rolling_regime_mpc/results/figures/v3_monthly_weights.png")}
      </div>
    </section>

    <section id="results">
      <div class="wrap">
        <p class="eyebrow">Main results</p>
        <h2>V3 improves return and Sharpe relative to all three references.</h2>
        <div class="metric-row">
          {metric("V3 final wealth", num(v3["Final Wealth"], 3), f'Equal weight: {num(ew["Final Wealth"], 3)}')}
          {metric("V3 annual return", pct(v3["Annual Return"]), f'Original HMM-MPC: {pct(hmm["Annual Return"])}')}
          {metric("Return diff. vs EW", pct_pp(v3_vs_ew["annualized_return_difference"]), "annualized")}
          {metric("Return diff. vs HMM", pct_pp(v3_vs_hmm["annualized_return_difference"]), "annualized")}
        </div>
        <div class="grid two mt-32">
          {image_block("04_empirical_section/figures/fig_01_cumulative_wealth.png", "Cumulative wealth", "Full-sample cumulative wealth for V3 and reference strategies.", "docs/04_empirical_section/figures/fig_01_cumulative_wealth.png")}
          <div class="panel">
            {full_table}
            {source_badge("docs/04_empirical_section/tables/tab_04_v3_full_sample_performance.csv")}
          </div>
        </div>
      </div>
    </section>

    <section id="excess">
      <div class="wrap grid two">
        <div>
          <p class="eyebrow">Excess performance</p>
          <h2>The improvement is visible against every reference strategy.</h2>
          <div class="panel mt-24">
            {excess_table}
            {source_badge("docs/04_empirical_section/tables/tab_05_v3_excess_performance.csv")}
          </div>
          <div class="note">Against equal weight, V3 improves final wealth by a factor of {esc(num(v3_vs_ew["final_wealth_ratio"], 3))}x and has a higher Sharpe ratio by {esc(num(v3_vs_ew["sharpe_difference"], 3))}.</div>
        </div>
        {image_block("04_empirical_section/figures/fig_03_v3_excess_cumulative_pnl.png", "V3 excess cumulative PnL", "Cumulative V3 PnL relative to the reference strategies.", "docs/04_empirical_section/figures/fig_03_v3_excess_cumulative_pnl.png")}
      </div>
    </section>

    <section id="diagnostics">
      <div class="wrap">
        <p class="eyebrow">Diagnostics</p>
        <h2>Result checks for yearly performance, drawdown, selection, and regimes.</h2>
        <div class="figure-grid mt-32">
          {image_block("04_empirical_section/figures/fig_02_drawdowns.png", "Drawdowns", "Drawdown paths for V3 and baselines.", "docs/04_empirical_section/figures/fig_02_drawdowns.png")}
          {image_block("04_empirical_section/figures/fig_04_yearly_active_return.png", "Yearly active return", "Annual active return of V3 against reference strategies.", "docs/04_empirical_section/figures/fig_04_yearly_active_return.png")}
          {image_block("04_empirical_section/figures/fig_05_selected_parameters_over_time.png", "Selected parameters", "Rolling parameter choices selected by V3 over time.", "docs/04_empirical_section/figures/fig_05_selected_parameters_over_time.png")}
          {image_block("04_empirical_section/figures/fig_06_average_weights_by_regime.png", "Average weights by regime", "Average V3 allocations conditional on inferred market regime.", "docs/04_empirical_section/figures/fig_06_average_weights_by_regime.png")}
        </div>
        <div class="panel mt-24">
          {regime_weight_table}
          {source_badge("docs/04_empirical_section/tables/tab_08_v3_average_weights_by_regime.csv")}
        </div>
      </div>
    </section>

    <section id="status">
      <div class="wrap grid two">
        <div>
          <p class="eyebrow">Current status</p>
          <h2>The project is ready for a progress meeting.</h2>
          <ul class="short-list">
            <li>Literature review, theoretical model, and regime design are drafted.</li>
            <li>Data acquisition, cleaning, quality checks, and visualization are complete.</li>
            <li>Equal weight and Markowitz CVXPY baselines are implemented.</li>
            <li>The original HMM-MPC model has been implemented and diagnosed.</li>
            <li>V3 rolling regime-MPC is implemented with reproducible tables and figures.</li>
          </ul>
          <div class="note">Suggested discussion focus: robustness to transaction costs, alternative regime specifications, and how much of V3's performance is linked to sector momentum persistence.</div>
        </div>
        <div class="panel">
          <h3>Traceability checklist</h3>
          <ul class="short-list">
            <li>Performance values come from CSV outputs.</li>
            <li>V3 design settings come from the V3 manifest and experiment outputs.</li>
            <li>Mathematical formulas match the project empirical/theoretical documents.</li>
            <li>Figures are linked from existing project output directories.</li>
          </ul>
          <h3 class="mt-24">Empirical source files</h3>
          <ul class="short-list small">
            {source_html}
          </ul>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""
    OUT.write_text(html_text, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    out = build_report()
    print(out)
