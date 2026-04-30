from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mpl_cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DOC_DIR = ROOT / "docs" / "04_empirical_section"
FIGURE_DIR = DOC_DIR / "figures"
TABLE_DIR = DOC_DIR / "tables"

V3_TABLE_DIR = ROOT / "improvements" / "06_v3_rolling_regime_mpc" / "results" / "tables"
OUTPUT_TABLE_DIR = ROOT / "outputs" / "tables"

STRATEGY_NAME = "V3_Rolling_Regime_MPC"
REFERENCES = ["EqualWeight_weekly", "Markowitz_CVXPY_weekly", "HMM_MPC_CVXPY_weekly"]
ASSETS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "font.size": 10,
            "legend.frameon": False,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )


def strategy_label(name: str) -> str:
    mapping = {
        "EqualWeight_weekly": "Equal Weight",
        "Markowitz_CVXPY_weekly": "Markowitz",
        "HMM_MPC_CVXPY_weekly": "Original HMM-MPC",
        "V3_Rolling_Regime_MPC": "V3 Rolling Regime-MPC",
    }
    return mapping.get(name, name)


def format_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "strategy",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "final_wealth",
        "avg_rebalance_turnover",
        "total_cost",
    ]
    out = df[cols].copy()
    out["strategy"] = out["strategy"].map(strategy_label)
    rename = {
        "strategy": "Strategy",
        "annualized_return": "Annual Return",
        "annualized_volatility": "Annual Volatility",
        "sharpe": "Sharpe",
        "max_drawdown": "Max Drawdown",
        "final_wealth": "Final Wealth",
        "avg_rebalance_turnover": "Avg Rebalance Turnover",
        "total_cost": "Total Cost",
    }
    return out.rename(columns=rename)


def save_table(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(TABLE_DIR / filename, index=False)


def wealth_frame(returns: pd.DataFrame) -> pd.DataFrame:
    return (1.0 + returns).cumprod()


def drawdown_frame(returns: pd.DataFrame) -> pd.DataFrame:
    wealth = wealth_frame(returns)
    return wealth / wealth.cummax() - 1.0


def plot_cumulative_wealth(returns: pd.DataFrame) -> None:
    wealth = wealth_frame(returns[[*REFERENCES, STRATEGY_NAME]])
    colors = {
        "EqualWeight_weekly": "#5f6c72",
        "Markowitz_CVXPY_weekly": "#b07c2b",
        "HMM_MPC_CVXPY_weekly": "#7a6fb0",
        STRATEGY_NAME: "#0b6e4f",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for col in wealth.columns:
        ax.plot(
            wealth.index,
            wealth[col],
            label=strategy_label(col),
            color=colors[col],
            linewidth=2.4 if col == STRATEGY_NAME else 1.5,
            alpha=0.98 if col == STRATEGY_NAME else 0.82,
        )
    ax.set_title("Cumulative Wealth of Backtested Strategies")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(FIGURE_DIR / "fig_01_cumulative_wealth.png", dpi=220)
    plt.close(fig)


def plot_drawdowns(returns: pd.DataFrame) -> None:
    drawdowns = drawdown_frame(returns[[*REFERENCES, STRATEGY_NAME]])
    colors = {
        "EqualWeight_weekly": "#5f6c72",
        "Markowitz_CVXPY_weekly": "#b07c2b",
        "HMM_MPC_CVXPY_weekly": "#7a6fb0",
        STRATEGY_NAME: "#0b6e4f",
    }
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for col in drawdowns.columns:
        ax.plot(
            drawdowns.index,
            drawdowns[col],
            label=strategy_label(col),
            color=colors[col],
            linewidth=2.2 if col == STRATEGY_NAME else 1.3,
            alpha=0.96 if col == STRATEGY_NAME else 0.78,
        )
    ax.set_title("Strategy Drawdowns")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="lower left", ncol=2)
    fig.savefig(FIGURE_DIR / "fig_02_drawdowns.png", dpi=220)
    plt.close(fig)


def plot_excess_cum_pnl(returns: pd.DataFrame) -> pd.DataFrame:
    wealth = wealth_frame(returns[[STRATEGY_NAME, *REFERENCES]])
    cum_pnl = wealth - 1.0
    excess = pd.DataFrame(index=returns.index)
    for ref in REFERENCES:
        excess[f"V3 minus {strategy_label(ref)}"] = cum_pnl[STRATEGY_NAME] - cum_pnl[ref]

    colors = ["#0b6e4f", "#2f6fbb", "#c44e52"]
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for color, col in zip(colors, excess.columns):
        ax.plot(excess.index, excess[col], label=col, color=color, linewidth=1.9)
    ax.axhline(0.0, color="#222222", linewidth=0.9, alpha=0.6)
    ax.set_title("V3 Cumulative PnL Minus Reference Strategies")
    ax.set_ylabel("Cumulative PnL difference")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="upper left")
    fig.savefig(FIGURE_DIR / "fig_03_v3_excess_cumulative_pnl.png", dpi=220)
    plt.close(fig)
    excess.to_csv(TABLE_DIR / "v3_excess_cumulative_pnl.csv")
    return excess


def plot_yearly_active(yearly: pd.DataFrame) -> None:
    plot_df = yearly.copy()
    rename = {
        "active_vs_EqualWeight_weekly": "vs Equal Weight",
        "active_vs_Markowitz_CVXPY_weekly": "vs Markowitz",
        "active_vs_HMM_MPC_CVXPY_weekly": "vs Original HMM-MPC",
    }
    columns = list(rename.keys())
    x = np.arange(len(plot_df))
    width = 0.24
    colors = ["#0b6e4f", "#2f6fbb", "#c44e52"]
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    for i, col in enumerate(columns):
        ax.bar(x + (i - 1) * width, plot_df[col], width=width, label=rename[col], color=colors[i], alpha=0.86)
    ax.axhline(0.0, color="#222222", linewidth=0.9, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["year"].astype(str), rotation=0)
    ax.set_title("Annual Active Return of V3")
    ax.set_ylabel("Annualized active return")
    ax.grid(axis="y", alpha=0.55)
    ax.legend(loc="upper right", ncol=3)
    fig.savefig(FIGURE_DIR / "fig_04_yearly_active_return.png", dpi=220)
    plt.close(fig)


def plot_selected_parameters(selection: pd.DataFrame) -> None:
    frame = selection.copy()
    frame["effective_date"] = pd.to_datetime(frame["effective_date"])
    frame = frame.set_index("effective_date")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.0), sharex=True)

    axes[0].step(frame.index, frame["base_alpha"], where="post", color="#0b6e4f", linewidth=1.8)
    axes[0].plot(frame.index, frame["effective_alpha"], color="#2f6fbb", linewidth=1.2, alpha=0.85)
    axes[0].set_ylabel("Alpha")
    axes[0].set_title("Rolling Parameter Selection in V3")
    axes[0].legend(["Base alpha", "Regime-adjusted alpha"], loc="upper left")
    axes[0].grid(True, alpha=0.5)

    axes[1].step(frame.index, frame["gamma"], where="post", color="#7a6fb0", linewidth=1.6)
    axes[1].set_ylabel("Risk aversion")
    axes[1].grid(True, alpha=0.5)

    axes[2].step(frame.index, frame["turnover_cap"], where="post", color="#b07c2b", linewidth=1.6)
    axes[2].set_ylabel("Turnover cap")
    axes[2].grid(True, alpha=0.5)

    fig.savefig(FIGURE_DIR / "fig_05_selected_parameters_over_time.png", dpi=220)
    plt.close(fig)


def plot_average_weights_by_regime(avg_weights: pd.DataFrame) -> None:
    frame = avg_weights.set_index("regime")[ASSETS].copy()
    order = [x for x in ["calm", "transition", "stress"] if x in frame.index]
    frame = frame.loc[order]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    image = ax.imshow(frame.values, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.30, frame.values.max()))
    ax.set_xticks(np.arange(len(ASSETS)))
    ax.set_xticklabels(ASSETS)
    ax.set_yticks(np.arange(len(frame.index)))
    ax.set_yticklabels([idx.capitalize() for idx in frame.index])
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            ax.text(j, i, f"{frame.iloc[i, j]:.0%}", ha="center", va="center", color="#111111", fontsize=8)
    ax.set_title("Average V3 Portfolio Weights by Dominant Regime")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Average weight")
    fig.savefig(FIGURE_DIR / "fig_06_average_weights_by_regime.png", dpi=220)
    plt.close(fig)


def plot_selection_counts(selection_summary: pd.DataFrame) -> None:
    top = selection_summary.sort_values("selection_count", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.barh(top["selected_candidate"], top["selection_count"], color="#0b6e4f", alpha=0.82)
    ax.set_title("Frequency of Selected V3 Candidate Parameters")
    ax.set_xlabel("Number of rebalances")
    ax.grid(axis="x", alpha=0.55)
    fig.savefig(FIGURE_DIR / "fig_07_selection_counts.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    set_style()

    full_returns = pd.read_csv(V3_TABLE_DIR / "full_strategy_daily_returns.csv", index_col=0, parse_dates=True)
    full_perf = pd.read_csv(V3_TABLE_DIR / "full_sample_performance.csv")
    full_excess = pd.read_csv(V3_TABLE_DIR / "full_sample_excess_vs_references.csv")
    candidate_perf = pd.read_csv(V3_TABLE_DIR / "candidate_performance_full.csv")
    yearly = pd.read_csv(V3_TABLE_DIR / "yearly_active_returns.csv")
    selection = pd.read_csv(V3_TABLE_DIR / "selected_params_over_time.csv")
    selection_summary = pd.read_csv(V3_TABLE_DIR / "selection_summary.csv")
    avg_weights = pd.read_csv(V3_TABLE_DIR / "average_weights_by_regime.csv")

    data_quality = pd.read_csv(OUTPUT_TABLE_DIR / "data_quality_summary.csv")
    asset_summary = pd.read_csv(OUTPUT_TABLE_DIR / "asset_return_summary.csv")
    original_perf = pd.read_csv(OUTPUT_TABLE_DIR / "strategy_performance.csv")
    forecast_quality = pd.read_csv(OUTPUT_TABLE_DIR / "diagnostic_forecast_quality.csv")
    average_original_weights = pd.read_csv(OUTPUT_TABLE_DIR / "diagnostic_average_weights.csv")
    yearly_strategy_returns = pd.read_csv(OUTPUT_TABLE_DIR / "diagnostic_yearly_strategy_returns.csv")

    save_table(data_quality, "tab_01_data_quality.csv")
    save_table(asset_summary, "tab_02_asset_return_summary.csv")
    save_table(format_performance_table(original_perf), "tab_03_original_strategy_performance.csv")
    save_table(format_performance_table(full_perf), "tab_04_v3_full_sample_performance.csv")
    save_table(full_excess, "tab_05_v3_excess_performance.csv")
    save_table(candidate_perf, "tab_06_v3_candidate_performance.csv")
    save_table(selection_summary, "tab_07_v3_selection_summary.csv")
    save_table(avg_weights, "tab_08_v3_average_weights_by_regime.csv")
    save_table(yearly, "tab_09_v3_yearly_active_returns.csv")
    save_table(forecast_quality, "tab_10_forecast_diagnostics.csv")
    save_table(average_original_weights, "tab_11_original_average_weights.csv")
    save_table(yearly_strategy_returns, "tab_12_original_yearly_returns.csv")

    plot_cumulative_wealth(full_returns)
    plot_drawdowns(full_returns)
    plot_excess_cum_pnl(full_returns)
    plot_yearly_active(yearly)
    plot_selected_parameters(selection)
    plot_average_weights_by_regime(avg_weights)
    plot_selection_counts(selection_summary)

    print(f"Wrote empirical paper assets to {DOC_DIR}")


if __name__ == "__main__":
    main()
