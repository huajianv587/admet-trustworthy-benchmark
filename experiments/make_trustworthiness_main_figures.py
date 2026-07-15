from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# Defaults preserve the audited traditional-only preview. Formal five-model
# artwork is enabled only when every input table passes the exact matrix gate.
ALIGNMENT_DEFAULT = ROOT / "results" / "tables" / "trustworthiness_alignment_local_prelim"
CONDITIONAL_DEFAULT = ROOT / "results" / "tables" / "conditional_reliability_local_prelim"
CALIBRATION_DEFAULT = ROOT / "results" / "tables" / "calibration_summary_local_prelim.csv"
OUT_DEFAULT = ROOT / "results" / "figures" / "trustworthiness_submission_partial"

TRADITIONAL_MODELS = {"rf", "xgboost", "lightgbm"}
FINAL_MODELS = TRADITIONAL_MODELS | {"gin", "chemprop_dmpnn"}
DEPRECATED_MODEL_IDS = {"dmpnn", "legacy_node_mpnn"}
PROVISIONAL_BANNER = (
    "PROVISIONAL · TRADITIONAL-ONLY (RF / XGB / LGBM) · "
    "FINAL FIVE-MODEL ANALYSIS PENDING"
)
FINAL_BANNER = "COMPLETE FIVE-MODEL MATRIX (5 SEEDS × 22 ENDPOINTS)"


# Okabe-Ito-derived, color-vision-deficiency-friendly palette.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GOLD = "#E69F00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
DARK = "#252525"
MID_GREY = "#777777"
LIGHT_GREY = "#E6E6E6"
GRID_GREY = "#D0D0D0"


TASK_ORDER = [
    "ames",
    "bbb_martins",
    "bioavailability_ma",
    "cyp2c9_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith",
    "dili",
    "herg",
    "hia_hou",
    "pgp_broccatelli",
    "caco2_wang",
    "clearance_hepatocyte_az",
    "clearance_microsome_az",
    "half_life_obach",
    "ld50_zhu",
    "lipophilicity_astrazeneca",
    "ppbr_az",
    "solubility_aqsoldb",
    "vdss_lombardo",
]

TASK_LABELS = {
    "ames": "AMES",
    "bbb_martins": "BBB Martins",
    "bioavailability_ma": "Bioavailability",
    "caco2_wang": "Caco-2",
    "clearance_hepatocyte_az": "Hepatocyte clearance",
    "clearance_microsome_az": "Microsomal clearance",
    "cyp2c9_substrate_carbonmangels": "CYP2C9 substrate",
    "cyp2c9_veith": "CYP2C9 inhibition",
    "cyp2d6_substrate_carbonmangels": "CYP2D6 substrate",
    "cyp2d6_veith": "CYP2D6 inhibition",
    "cyp3a4_substrate_carbonmangels": "CYP3A4 substrate",
    "cyp3a4_veith": "CYP3A4 inhibition",
    "dili": "DILI",
    "half_life_obach": "Half-life",
    "herg": "hERG",
    "hia_hou": "HIA",
    "ld50_zhu": "LD50",
    "lipophilicity_astrazeneca": "Lipophilicity",
    "pgp_broccatelli": "P-gp inhibition",
    "ppbr_az": "PPBR",
    "solubility_aqsoldb": "Solubility",
    "vdss_lombardo": "VDss",
}

DIMENSION_ORDER = [
    "uncalibrated_ece",
    "uncalibrated_brier",
    "conformal_coverage_error",
    "selective_gain_30pct",
    "applicability_alignment",
]

DIMENSION_LABELS = {
    "uncalibrated_ece": "ECE",
    "uncalibrated_brier": "Brier score",
    "conformal_coverage_error": "Validation-residual\ncoverage error",
    "selective_gain_30pct": "Selective gain\n(30% rejection)",
    "applicability_alignment": "Applicability-\ndomain alignment",
}

MODEL_LABELS = {
    "rf": "RF",
    "xgboost": "XGB",
    "lightgbm": "LGBM",
    "gin": "GIN",
    "chemprop_dmpnn": "Chemprop D-MPNN",
    "dmpnn": "Legacy node-MPNN (excluded)",
    "legacy_node_mpnn": "Legacy node-MPNN (excluded)",
}

CONDITIONAL_LABELS = {
    "log_error_ratio_outside_inside": "Mean absolute error",
    "uq_correlation_advantage_inside": "Uncertainty-error correlation",
    "selective_gain_advantage_inside": "Selective-prediction gain",
    "ece_gap_outside_inside": "ECE",
    "brier_gap_outside_inside": "Brier score",
    "coverage_error_gap_outside_inside": "Validation-residual coverage error",
}


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.3,
            "axes.titlesize": 11.2,
            "axes.labelsize": 9.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 9.3,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, out_dir: Path, name: str, title: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    svg = out_dir / f"{name}.svg"
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title, "Creator": "Matplotlib"},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title, "Creator": "Matplotlib"},
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": title, "Creator": "Matplotlib"},
    )
    return [png, pdf, svg]


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def model_coverage(profile: pd.DataFrame) -> tuple[dict[str, int], bool, str]:
    require_columns(profile, {"task", "model", "n_seeds"}, Path("model profile"))
    coverage = {
        str(model): int(count)
        for model, count in profile.groupby("model")["task"].nunique().sort_index().items()
    }
    complete = bool(
        set(coverage) == FINAL_MODELS
        and all(coverage[model] == 22 for model in FINAL_MODELS)
        and int(profile["n_seeds"].min()) >= 5
    )
    parts = [
        f"{MODEL_LABELS.get(model, model)} {count}/22"
        for model, count in sorted(coverage.items())
    ]
    if complete:
        note = "Complete five-seed model matrix: " + ", ".join(parts) + "."
    elif set(coverage) == TRADITIONAL_MODELS and all(
        coverage[model] == 22 for model in TRADITIONAL_MODELS
    ) and int(profile["n_seeds"].min()) >= 5:
        note = (
            "PROVISIONAL traditional-only matrix: "
            + ", ".join(parts)
            + ". GIN and Chemprop D-MPNN are intentionally excluded until their "
            "locked five-seed endpoint matrix is complete."
        )
    else:
        note = (
            "INCOMPLETE matrix: "
            + ", ".join(parts)
            + ". This source is not eligible for submission artwork."
        )
    return coverage, complete, note


def add_provisional_banner(fig, x: float, y: float = 0.915) -> None:
    fig.text(
        x,
        y,
        PROVISIONAL_BANNER,
        ha="left",
        va="bottom",
        fontsize=10.0,
        fontweight="bold",
        color=ORANGE,
    )


def endpoint_order(frame: pd.DataFrame) -> list[str]:
    available = set(frame["task"].dropna().astype(str))
    ordered = [task for task in TASK_ORDER if task in available]
    ordered.extend(sorted(available.difference(ordered)))
    return ordered


def add_panel_label(ax, label: str, y: float = 1.04) -> None:
    ax.text(
        -0.08,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        ha="right",
        va="bottom",
    )


def figure_winner_misalignment(
    rank_by_task: pd.DataFrame,
    rank_summary: pd.DataFrame,
    coverage_note: str,
    out_dir: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    require_columns(
        rank_by_task,
        {
            "dimension",
            "task",
            "performance_winner",
            "reliability_winner",
            "same_winner",
        },
        Path("rank_alignment_by_task.csv"),
    )
    require_columns(
        rank_summary,
        {
            "dimension",
            "eligible_tasks",
            "same_winner_count",
            "same_winner_ci_low",
            "same_winner_ci_high",
            "mismatch_fraction",
        },
        Path("rank_alignment_summary.csv"),
    )

    tasks = endpoint_order(rank_by_task)
    lookup = rank_by_task.set_index(["task", "dimension"])
    values = np.full((len(tasks), len(DIMENSION_ORDER)), 2.0)
    annotations: list[list[str]] = [["N/A"] * len(DIMENSION_ORDER) for _ in tasks]
    for row_index, task in enumerate(tasks):
        for column_index, dimension in enumerate(DIMENSION_ORDER):
            if (task, dimension) not in lookup.index:
                continue
            row = lookup.loc[(task, dimension)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            same_raw = row["same_winner"]
            same = (
                bool(same_raw)
                if isinstance(same_raw, (bool, np.bool_))
                else str(same_raw).strip().lower() in {"true", "1", "yes"}
            )
            values[row_index, column_index] = 1.0 if same else 0.0
            performance = MODEL_LABELS.get(str(row["performance_winner"]), str(row["performance_winner"]))
            reliability = MODEL_LABELS.get(str(row["reliability_winner"]), str(row["reliability_winner"]))
            annotations[row_index][column_index] = (
                performance if same else f"{performance}→{reliability}"
            )

    fig = plt.figure(figsize=(12.0, 9.3), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.15, 1.25],
        left=0.12,
        right=0.985,
        top=0.82,
        bottom=0.12,
        wspace=0.28,
    )
    ax = fig.add_subplot(grid[0, 0])
    ax_summary = fig.add_subplot(grid[0, 1])

    cmap = ListedColormap(["#F2C9B9", "#B8D8E8", LIGHT_GREY])
    ax.imshow(values, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(len(DIMENSION_ORDER)))
    ax.set_xticklabels([DIMENSION_LABELS[value] for value in DIMENSION_ORDER])
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels([TASK_LABELS.get(task, task) for task in tasks])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xticks(np.arange(-0.5, len(DIMENSION_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(len(tasks)):
        for column_index in range(len(DIMENSION_ORDER)):
            text = annotations[row_index][column_index]
            color = MID_GREY if text == "N/A" else DARK
            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=5.6 if text != "N/A" else 5.2,
                color=color,
            )
    if "caco2_wang" in tasks:
        split = tasks.index("caco2_wang") - 0.5
        ax.axhline(split, color=DARK, linewidth=1.1)
    add_panel_label(ax, "A")

    summary = rank_summary.set_index("dimension").reindex(DIMENSION_ORDER).dropna(how="all")
    y = np.arange(len(summary))
    mismatch = summary["mismatch_fraction"].to_numpy(dtype=float)
    mismatch_low = 1.0 - summary["same_winner_ci_high"].to_numpy(dtype=float)
    mismatch_high = 1.0 - summary["same_winner_ci_low"].to_numpy(dtype=float)
    xerr = np.vstack([mismatch - mismatch_low, mismatch_high - mismatch])
    ax_summary.errorbar(
        mismatch,
        y,
        xerr=xerr,
        fmt="o",
        markersize=6,
        color=ORANGE,
        ecolor=DARK,
        elinewidth=1.0,
        capsize=3,
    )
    ax_summary.axvline(0.5, color=MID_GREY, linestyle="--", linewidth=0.9)
    ax_summary.set_xlim(-0.03, 1.04)
    ax_summary.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_summary.set_xticklabels(["0", "25", "50", "75", "100"])
    ax_summary.set_xlabel("Endpoints with a different winner (%)")
    ax_summary.set_yticks(y)
    ax_summary.set_yticklabels(
        [DIMENSION_LABELS[dimension].replace("\n", " ") for dimension in summary.index]
    )
    ax_summary.invert_yaxis()
    ax_summary.grid(axis="x", color=GRID_GREY, linewidth=0.6)
    ax_summary.set_axisbelow(True)
    for index, (_, row) in enumerate(summary.iterrows()):
        mismatches = int(row["eligible_tasks"] - row["same_winner_count"])
        eligible = int(row["eligible_tasks"])
        ax_summary.text(
            min(float(row["mismatch_fraction"]) + 0.035, 0.96),
            index - 0.16,
            f"{mismatches}/{eligible}",
            ha="left",
            va="center",
            fontsize=7,
        )
    ax_summary.set_title("Mismatch frequency", loc="left", pad=25)
    ax_summary.text(
        0.0,
        1.005,
        "Point = endpoint fraction; bar = 95% endpoint bootstrap interval",
        transform=ax_summary.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=MID_GREY,
    )
    add_panel_label(ax_summary, "B")

    fig.suptitle(
        "Official performance and operational reliability often select different ADMET models",
        x=0.12,
        y=0.98,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.12,
        0.915,
        "Endpoint-level performance winner → reliability winner",
        ha="left",
        va="bottom",
        fontsize=10,
    )
    fig.legend(
        handles=[
            Patch(facecolor="#B8D8E8", label="Same winner"),
            Patch(facecolor="#F2C9B9", label="Different winner"),
            Patch(facecolor=LIGHT_GREY, label="Not applicable"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.12, 0.89),
        ncol=3,
        frameon=False,
        borderaxespad=0,
        handlelength=1.2,
        columnspacing=1.3,
    )
    fig.text(
        0.12,
        0.045,
        coverage_note
        + " Cell text shows the winner, or performance→reliability when winners differ.",
        ha="left",
        va="bottom",
        fontsize=7.2,
        color=MID_GREY,
        wrap=True,
    )
    return save_figure(
        fig,
        out_dir,
        "fig01_winner_misalignment",
        "Performance-versus-reliability winner misalignment",
    )


def eligible_threshold_rows(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    require_columns(
        frame,
        {"task", "model", "task_type", "threshold", "eligible"},
        Path("conditional_reliability_by_task_model.csv"),
    )
    eligible = frame["eligible"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().isin({"true", "1", "yes"})
    return frame[np.isclose(frame["threshold"].astype(float), threshold) & eligible].copy()


def conditional_endpoint_plot_data(
    conditional_rows: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    rows = eligible_threshold_rows(conditional_rows, threshold)
    metrics = list(CONDITIONAL_LABELS)
    require_columns(rows, set(metrics), Path("conditional_reliability_by_task_model.csv"))
    endpoint = (
        rows.groupby(["task", "task_type"], as_index=False)[metrics]
        .mean()
        .sort_values(["task_type", "task"])
        .reset_index(drop=True)
    )
    endpoint["error_change_percent"] = 100.0 * (
        np.exp(endpoint["log_error_ratio_outside_inside"]) - 1.0
    )
    endpoint.insert(0, "threshold", float(threshold))
    return endpoint


def figure_conditional_reliability(
    conditional_rows: pd.DataFrame,
    conditional_summary: pd.DataFrame,
    coverage_note: str,
    threshold: float,
    out_dir: Path,
    provisional: bool = True,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    endpoint_plot = conditional_endpoint_plot_data(conditional_rows, threshold)
    summary = conditional_summary[np.isclose(conditional_summary["threshold"], threshold)].copy()
    require_columns(
        summary,
        {
            "metric",
            "n_endpoints",
            "positive_count",
            "mean",
            "mean_ci_low",
            "mean_ci_high",
            "holm_p_within_threshold",
        },
        Path("conditional_reliability_summary.csv"),
    )
    summary_index = summary.set_index("metric")

    # Portrait two-column layout. Panel A uses the full left-column height so
    # its 21 endpoint labels do not overlap after full-line manuscript
    # inclusion; Panels B and C share the right column.
    fig = plt.figure(figsize=(8.2, 9.2), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.25, 1.0],
        width_ratios=[1.05, 1.0],
        left=0.17,
        right=0.975,
        top=0.80,
        bottom=0.15,
        hspace=0.78,
        wspace=1.03,
    )
    ax_error = fig.add_subplot(grid[:, 0])
    ax_scores = fig.add_subplot(grid[0, 1])
    ax_direction = fig.add_subplot(grid[1, 1])

    endpoint_error = endpoint_plot.dropna(
        subset=["log_error_ratio_outside_inside"]
    ).copy()
    endpoint_error = endpoint_error.sort_values("error_change_percent", ascending=True)
    y = np.arange(len(endpoint_error))
    colors = endpoint_error["task_type"].map(
        {"classification": BLUE, "regression": GOLD}
    ).fillna(MID_GREY)
    ax_error.hlines(y, 0, endpoint_error["error_change_percent"], color=colors, linewidth=1.4)
    ax_error.scatter(
        endpoint_error["error_change_percent"],
        y,
        c=colors,
        s=28,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax_error.axvline(0, color=DARK, linewidth=0.9)
    ax_error.set_yticks(y)
    ax_error.set_yticklabels(
        [TASK_LABELS.get(task, task) for task in endpoint_error["task"]]
    )
    ax_error.set_xlabel("Outside-vs-inside mean absolute error change (%)")
    ax_error.set_title(
        "Chemical-support shift increases prediction error\n"
        f"at Tanimoto threshold {threshold:.1f}",
        loc="left",
        pad=29,
    )
    ax_error.grid(axis="x", color=GRID_GREY, linewidth=0.6)
    ax_error.set_axisbelow(True)
    error_stats = summary_index.loc["log_error_ratio_outside_inside"]
    ax_error.text(
        0.0,
        1.012,
        (
            f"{int(error_stats['positive_count'])}/{int(error_stats['n_endpoints'])} eligible endpoints > 0; "
            f"mean log ratio {error_stats['mean']:.3f}\n"
            f"[{error_stats['mean_ci_low']:.3f}, {error_stats['mean_ci_high']:.3f}]; "
            f"Descriptive Holm p={format_p(float(error_stats['holm_p_within_threshold']))}"
        ),
        transform=ax_error.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=DARK,
    )
    ax_error.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE, label="Classification", linewidth=1.4),
            Line2D([0], [0], marker="o", color=GOLD, label="Regression", linewidth=1.4),
        ],
        loc="lower right",
        ncol=1,
        frameon=False,
    )
    add_panel_label(ax_error, "A")

    score_columns = ["brier_gap_outside_inside", "ece_gap_outside_inside"]
    score_data = endpoint_plot.dropna(subset=score_columns).copy()
    score_data = score_data.sort_values("brier_gap_outside_inside", ascending=True)
    y_score = np.arange(len(score_data))
    ax_scores.axvline(0, color=DARK, linewidth=0.9)
    for index, row in score_data.reset_index(drop=True).iterrows():
        ax_scores.plot(
            [row["ece_gap_outside_inside"], row["brier_gap_outside_inside"]],
            [index, index],
            color=GRID_GREY,
            linewidth=0.9,
            zorder=1,
        )
    ax_scores.scatter(
        score_data["brier_gap_outside_inside"],
        y_score,
        s=29,
        color=BLUE,
        marker="o",
        label="Brier gap",
        zorder=3,
    )
    ax_scores.scatter(
        score_data["ece_gap_outside_inside"],
        y_score,
        s=28,
        color=ORANGE,
        marker="s",
        label="ECE gap",
        zorder=3,
    )
    ax_scores.set_yticks(y_score)
    ax_scores.set_yticklabels(
        [TASK_LABELS.get(task, task) for task in score_data["task"]]
    )
    ax_scores.set_xlabel("Outside − inside score\n(positive = worse outside)")
    ax_scores.set_title(
        "Brier degradation is consistent;\nECE is mixed", loc="left", pad=30
    )
    ax_scores.grid(axis="x", color=GRID_GREY, linewidth=0.6)
    ax_scores.set_axisbelow(True)
    ax_scores.legend(frameon=False, loc="lower right")
    brier_stats = summary_index.loc["brier_gap_outside_inside"]
    ece_stats = summary_index.loc["ece_gap_outside_inside"]
    ax_scores.text(
        0.0,
        1.012,
        (
            f"Brier: {int(brier_stats['positive_count'])}/{int(brier_stats['n_endpoints'])}, "
            f"descriptive Holm p={format_p(float(brier_stats['holm_p_within_threshold']))};\n"
            f"ECE: {int(ece_stats['positive_count'])}/{int(ece_stats['n_endpoints'])}, "
            f"descriptive Holm p={format_p(float(ece_stats['holm_p_within_threshold']))}"
        ),
        transform=ax_scores.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=MID_GREY,
    )
    add_panel_label(ax_scores, "B", y=1.12)

    metric_order = list(CONDITIONAL_LABELS)
    direction = summary_index.reindex(metric_order).dropna(how="all")
    y_direction = np.arange(len(direction))
    positive_fraction = direction["positive_count"] / direction["n_endpoints"]
    opposite_fraction = 1.0 - positive_fraction
    ax_direction.barh(
        y_direction,
        positive_fraction,
        color="#F2C9B9",
        edgecolor=ORANGE,
        linewidth=0.7,
        label="Expected outside\ndegradation",
    )
    ax_direction.barh(
        y_direction,
        opposite_fraction,
        left=positive_fraction,
        color=LIGHT_GREY,
        edgecolor=MID_GREY,
        linewidth=0.7,
        hatch="///",
        label="Opposite / no\ndegradation",
    )
    ax_direction.axvline(0.5, color=MID_GREY, linestyle="--", linewidth=0.8)
    ax_direction.set_xlim(0, 1.0)
    ax_direction.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_direction.set_xticklabels(["0", "25", "50", "75", "100"])
    ax_direction.set_xlabel("Eligible endpoints (%)")
    ax_direction.set_yticks(y_direction)
    ax_direction.set_yticklabels(
        [CONDITIONAL_LABELS[metric] for metric in direction.index]
    )
    ax_direction.invert_yaxis()
    for index, (_, row) in enumerate(direction.iterrows()):
        ax_direction.text(
            0.98,
            index,
            (
                f"{int(row['positive_count'])}/{int(row['n_endpoints'])}; "
                f"descriptive p_H={format_p(float(row['holm_p_within_threshold']))}"
            ),
            ha="right",
            va="center",
            fontsize=9.3,
            color=DARK,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.5},
        )
    ax_direction.set_title(
        "Reliability metrics are not\ninterchangeable", loc="left", pad=32
    )
    ax_direction.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.012),
        ncol=2,
        borderaxespad=0,
        handlelength=1.3,
        columnspacing=1.0,
    )
    add_panel_label(ax_direction, "C", y=1.14)

    fig.suptitle(
        "Outside-support prediction error rises,\nwhile reliability shifts are metric-dependent",
        x=0.08,
        y=0.99,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    if provisional:
        add_provisional_banner(fig, 0.08, y=0.875)
    else:
        fig.text(
            0.08,
            0.875,
            FINAL_BANNER,
            ha="left",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color=GREEN,
        )
    total_endpoints = int(conditional_rows["task"].nunique())
    excluded = total_endpoints - int(error_stats["n_endpoints"])
    exclusion_note = (
        f" {excluded} endpoint was ineligible at this threshold because a domain subset "
        "did not meet the predefined minimum sample count."
        if excluded
        else ""
    )
    matrix_note = (
        coverage_note
        if provisional
        else "Complete matrix: five model families × 22 endpoints × five seeds."
    )
    figure_note = textwrap.fill(
        (
            "Endpoint is the comparison unit; task-model values are averaged within endpoint. "
            "Positive direction denotes worse outside-domain reliability or stronger inside-domain utility. "
            + matrix_note
            + exclusion_note
        ),
        width=96,
    )
    fig.text(
        0.08,
        0.025,
        figure_note,
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=MID_GREY,
        wrap=False,
    )
    return save_figure(
        fig,
        out_dir,
        "fig02_conditional_reliability",
        "Inside-versus-outside conditional reliability is metric-dependent",
    )


def calibration_endpoint_gains(calibration: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        calibration,
        {
            "task",
            "task_type",
            "model",
            "seed",
            "method",
            "test_ece_before",
            "test_ece_after",
            "test_brier_before",
            "test_brier_after",
        },
        CALIBRATION_DEFAULT,
    )
    frame = calibration[
        (calibration["task_type"] == "classification")
        & calibration["method"].isin(["isotonic", "platt"])
    ].copy()
    frame["ece_gain"] = frame["test_ece_before"] - frame["test_ece_after"]
    frame["brier_gain"] = frame["test_brier_before"] - frame["test_brier_after"]
    return (
        frame.groupby(["task", "method"], as_index=False)[["ece_gain", "brier_gain"]]
        .mean()
        .sort_values(["task", "method"])
    )


def figure_calibration_gains(
    calibration: pd.DataFrame,
    effects: pd.DataFrame,
    out_dir: Path,
    included_models: set[str] = TRADITIONAL_MODELS,
    provisional: bool = True,
) -> list[Path]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    endpoint = calibration_endpoint_gains(calibration)
    require_columns(
        effects,
        {
            "effect",
            "n_endpoints",
            "positive_count",
            "mean",
            "mean_ci_low",
            "mean_ci_high",
            "holm_p",
        },
        Path("method_effects_summary.csv"),
    )
    effect_index = effects.set_index("effect")
    methods = ["isotonic", "platt"]
    method_labels = ["Isotonic", "Logistic-on-p"]
    method_colors = [BLUE, GOLD]

    fig, axes = plt.subplots(
        2,
        1,
        # The vertical stack gives each annotation a full-width panel; 9.3-pt
        # text remains comfortably above 7 pt after manuscript inclusion.
        figsize=(7.2, 9.0),
        constrained_layout=False,
        gridspec_kw={"left": 0.13, "right": 0.985, "top": 0.80, "bottom": 0.18, "hspace": 0.52},
    )
    rng = np.random.default_rng(20260714)
    for panel_index, (ax, metric, title) in enumerate(
        zip(axes, ["ece_gain", "brier_gain"], ["ECE gain", "Brier-score gain"])
    ):
        arrays = [
            endpoint.loc[endpoint["method"] == method, metric].dropna().to_numpy(dtype=float)
            for method in methods
        ]
        violins = ax.violinplot(
            arrays,
            positions=np.arange(len(methods)),
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body, color in zip(violins["bodies"], method_colors):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.16)

        for method_index, (method, color) in enumerate(zip(methods, method_colors)):
            method_data = endpoint[endpoint["method"] == method].dropna(subset=[metric])
            jitter = rng.uniform(-0.11, 0.11, len(method_data))
            values = method_data[metric].to_numpy(dtype=float)
            improved = values > 0
            ax.scatter(
                np.full(np.sum(improved), method_index) + jitter[improved],
                values[improved],
                s=25,
                color=color,
                marker="o",
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            ax.scatter(
                np.full(np.sum(~improved), method_index) + jitter[~improved],
                values[~improved],
                s=29,
                color=ORANGE,
                marker="x",
                linewidth=1.1,
                zorder=4,
            )
            effect_name = f"{method}_{metric}"
            stats = effect_index.loc[effect_name]
            mean = float(stats["mean"])
            low = float(stats["mean_ci_low"])
            high = float(stats["mean_ci_high"])
            ax.errorbar(
                method_index,
                mean,
                yerr=np.array([[mean - low], [high - mean]]),
                fmt="D",
                markersize=5,
                color=DARK,
                ecolor=DARK,
                capsize=3,
                linewidth=1.0,
                zorder=5,
            )
            ax.text(
                method_index,
                0.975,
                (
                    f"{int(stats['positive_count'])}/{int(stats['n_endpoints'])} improve\n"
                    f"mean {mean:.3f} [{low:.3f}, {high:.3f}]\n"
                    f"Descriptive Holm p={format_p(float(stats['holm_p']))}"
                ),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9.3,
                color=DARK,
            )

        lower, upper = ax.get_ylim()
        ax.set_ylim(lower, upper + 0.28 * (upper - lower))
        ax.axhline(0, color=DARK, linewidth=0.9)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(method_labels)
        ax.set_ylabel(f"{title} (before − after)")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID_GREY, linewidth=0.6)
        ax.set_axisbelow(True)
        add_panel_label(ax, chr(ord("A") + panel_index))

    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", label="Endpoint improved"),
            Line2D([0], [0], marker="x", color=ORANGE, linestyle="none", label="Endpoint failed to improve"),
            Line2D([0], [0], marker="D", color=DARK, linestyle="none", label="Endpoint mean ± 95% bootstrap interval"),
        ],
        loc="lower left",
        frameon=False,
    )
    fig.suptitle(
        "Post-hoc calibration gains are metric- and endpoint-dependent",
        x=0.08,
        y=0.982,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    if provisional:
        add_provisional_banner(fig, 0.08, y=0.88)
    else:
        fig.text(
            0.08,
            0.88,
            FINAL_BANNER,
            ha="left",
            va="bottom",
            fontsize=10.0,
            fontweight="bold",
            color=GREEN,
        )
    model_description = (
        "five model families"
        if included_models == FINAL_MODELS
        else "three traditional models"
    )
    fig.text(
        0.08,
        0.04,
        (
            f"Each point is an endpoint mean across {model_description} and five seeds "
            "(13 classification endpoints). Positive gain indicates improvement; crosses mark "
            "zero or negative gain. Calibration mappings were fitted on validation predictions only."
        ),
        ha="left",
        va="bottom",
        fontsize=9.3,
        color=MID_GREY,
        wrap=True,
    )
    return save_figure(
        fig,
        out_dir,
        "fig03_calibration_gains_failures",
        "Calibration gains and failures across classification endpoints",
    )


def write_figure_stats(
    out_dir: Path,
    rank_summary: pd.DataFrame | None,
    conditional_summary: pd.DataFrame,
    effects: pd.DataFrame,
    threshold: float,
) -> Path:
    rows: list[pd.DataFrame] = []
    if rank_summary is not None:
        alignment = rank_summary.copy()
        alignment.insert(0, "figure", "fig01")
        alignment.insert(1, "panel", "B")
        alignment.insert(2, "analysis_unit", "endpoint")
        rows.append(alignment)

    conditional = conditional_summary[
        np.isclose(conditional_summary["threshold"].astype(float), threshold)
    ].copy()
    conditional.insert(0, "figure", "fig02")
    conditional.insert(1, "panel", "A-C")
    conditional.insert(2, "analysis_unit", "endpoint")
    rows.append(conditional)

    calibration = effects[
        effects["effect"].isin(
            ["isotonic_ece_gain", "platt_ece_gain", "isotonic_brier_gain", "platt_brier_gain"]
        )
    ].copy()
    calibration.insert(0, "figure", "fig03")
    calibration.insert(1, "panel", "A-B")
    calibration.insert(2, "analysis_unit", "endpoint")
    rows.append(calibration)

    combined = pd.concat(rows, ignore_index=True, sort=False)
    path = out_dir / "figure_source_statistics.csv"
    combined.to_csv(path, index=False)
    return path


def validate_final_five_model_sources(
    profile: pd.DataFrame,
    conditional_rows: pd.DataFrame,
    calibration: pd.DataFrame,
    threshold: float = 0.5,
) -> None:
    """Hard gate the exact source matrix used by final Figures 2--3."""
    require_columns(
        profile,
        {"task", "task_type", "model", "n_seeds"},
        Path("model_reliability_profiles.csv"),
    )
    profile_keys = profile[["task", "model"]]
    if (
        len(profile) != 22 * len(FINAL_MODELS)
        or profile_keys.duplicated().any()
        or set(profile["task"].astype(str)) != set(TASK_ORDER)
        or set(profile["model"].astype(str)) != FINAL_MODELS
        or set(pd.to_numeric(profile["n_seeds"], errors="raise").astype(int)) != {5}
        or any(
            int(count) != 22
            for count in profile.groupby("model")["task"].nunique()
        )
    ):
        raise ValueError(
            "Final profile gate failed: each of RF, XGB, LGBM, GIN, and Chemprop "
            "D-MPNN must have exactly 22 endpoint rows and exactly five seeds."
        )

    require_columns(
        conditional_rows,
        {"task", "model", "threshold", "eligible"},
        Path("conditional_reliability_by_task_model.csv"),
    )
    if set(conditional_rows["model"].astype(str)) != FINAL_MODELS:
        raise ValueError("Final conditional-reliability model set is incomplete")
    if set(conditional_rows["task"].astype(str)) != set(TASK_ORDER):
        raise ValueError("Final conditional-reliability endpoint set is incomplete")
    primary_conditional = conditional_rows[
        np.isclose(
            pd.to_numeric(conditional_rows["threshold"], errors="raise"), threshold
        )
    ]
    conditional_keys = primary_conditional[["task", "model"]]
    if (
        len(conditional_keys) != 22 * len(FINAL_MODELS)
        or conditional_keys.duplicated().any()
    ):
        raise ValueError(
            "Final conditional-reliability table must contain exactly one row for "
            f"every endpoint-model pair at threshold {threshold}"
        )

    require_columns(
        calibration,
        {
            "task",
            "task_type",
            "model",
            "seed",
            "method",
            "test_ece_before",
            "test_ece_after",
            "test_brier_before",
            "test_brier_after",
        },
        Path("calibration_summary.csv"),
    )
    classification_tasks = sorted(
        set(profile.loc[profile["task_type"].eq("classification"), "task"].astype(str))
    )
    methods = ("isotonic", "platt")
    calibration_core = calibration[
        calibration["task_type"].eq("classification")
        & calibration["method"].isin(methods)
    ].copy()
    key_columns = ["task", "model", "seed", "method"]
    if calibration_core.duplicated(key_columns).any():
        raise ValueError("Final calibration table contains duplicate run-method rows")
    observed = {
        (str(row.task), str(row.model), int(row.seed), str(row.method))
        for row in calibration_core[key_columns].itertuples(index=False)
    }
    expected = {
        (task, model, seed, method)
        for task in classification_tasks
        for model in FINAL_MODELS
        for seed in range(1, 6)
        for method in methods
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "Final calibration matrix mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    numeric = calibration_core[
        [
            "test_ece_before",
            "test_ece_after",
            "test_brier_before",
            "test_brier_after",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(
            "Final calibration matrix contains a failed or non-finite calibration run"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate submission-oriented trustworthiness figures from analysis tables."
    )
    parser.add_argument("--alignment-dir", default=str(ALIGNMENT_DEFAULT))
    parser.add_argument("--conditional-dir", default=str(CONDITIONAL_DEFAULT))
    parser.add_argument("--calibration-summary", default=str(CALIBRATION_DEFAULT))
    parser.add_argument("--out-dir", default=str(OUT_DEFAULT))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--include-superseded-fig01",
        action="store_true",
        help=(
            "Also regenerate the superseded winner-misalignment figure. The locked "
            "reliability-regret script is the submission source for Figure 1."
        ),
    )
    args = parser.parse_args()

    configure_matplotlib()
    import matplotlib.pyplot as plt

    alignment_dir = Path(args.alignment_dir)
    conditional_dir = Path(args.conditional_dir)
    calibration_path = Path(args.calibration_summary)
    out_dir = Path(args.out_dir)

    rank_by_task = pd.read_csv(alignment_dir / "rank_alignment_by_task.csv")
    rank_summary = pd.read_csv(alignment_dir / "rank_alignment_summary.csv")
    effects = pd.read_csv(alignment_dir / "method_effects_summary.csv")
    conditional_rows = pd.read_csv(conditional_dir / "conditional_reliability_by_task_model.csv")
    conditional_summary = pd.read_csv(conditional_dir / "conditional_reliability_summary.csv")
    profile_path = conditional_dir / "model_reliability_profiles.csv"
    if not profile_path.exists():
        profile_path = alignment_dir / "model_reliability_profiles.csv"
    profile = pd.read_csv(profile_path)
    calibration = pd.read_csv(calibration_path)

    coverage, complete, coverage_note = model_coverage(profile)
    profile_models = set(profile["model"].astype(str))
    conditional_models = set(conditional_rows["model"].astype(str))
    calibration_models = set(calibration["model"].astype(str))
    deprecated = sorted(
        (profile_models | conditional_models | calibration_models) & DEPRECATED_MODEL_IDS
    )
    if deprecated:
        raise ValueError(
            "Deprecated legacy node-MPNN identifiers are forbidden in Figures 2--3: "
            + ", ".join(deprecated)
        )
    provisional = bool(
        profile_models == TRADITIONAL_MODELS
        and conditional_models == TRADITIONAL_MODELS
        and calibration_models == TRADITIONAL_MODELS
    )
    final_five_model = bool(
        profile_models == FINAL_MODELS
        and conditional_models == FINAL_MODELS
        and calibration_models == FINAL_MODELS
        and complete
    )
    if provisional:
        traditional_gate = bool(
            coverage == {model: 22 for model in sorted(TRADITIONAL_MODELS)}
            and int(profile["n_seeds"].min()) >= 5
            and int(profile["n_seeds"].max()) == 5
            and int(profile[["task", "model"]].drop_duplicates().shape[0]) == 66
        )
        if not traditional_gate:
            raise ValueError(
                "Traditional-only figure gate failed: RF, XGB, and LGBM must each "
                "cover 22 endpoints with exactly five seeds."
            )
    elif final_five_model:
        validate_final_five_model_sources(
            profile, conditional_rows, calibration, threshold=args.threshold
        )
    else:
        raise ValueError(
            "Figure source model matrix is neither the audited traditional preview nor "
            "the exact final five-model matrix. Observed model sets: "
            f"profile={sorted(profile_models)}, conditional={sorted(conditional_models)}, "
            f"calibration={sorted(calibration_models)}"
        )

    generated: list[Path] = []
    if args.include_superseded_fig01:
        generated.extend(
            figure_winner_misalignment(
                rank_by_task, rank_summary, coverage_note, out_dir
            )
        )
        plt.close("all")
    generated.extend(
        figure_conditional_reliability(
            conditional_rows,
            conditional_summary,
            coverage_note,
            args.threshold,
            out_dir,
            provisional=provisional,
        )
    )
    plt.close("all")
    generated.extend(
        figure_calibration_gains(
            calibration,
            effects,
            out_dir,
            included_models=profile_models,
            provisional=provisional,
        )
    )
    plt.close("all")

    fig02_source = out_dir / "fig02_source_data.csv"
    conditional_endpoint_plot_data(conditional_rows, args.threshold).to_csv(
        fig02_source, index=False
    )
    generated.append(fig02_source)
    fig03_source = out_dir / "fig03_source_data.csv"
    calibration_endpoint_gains(calibration).to_csv(fig03_source, index=False)
    generated.append(fig03_source)

    stats_path = write_figure_stats(
        out_dir,
        rank_summary if args.include_superseded_fig01 else None,
        conditional_summary,
        effects,
        args.threshold,
    )
    generated.append(stats_path)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "provisional_traditional_only" if provisional else "final_five_model"
        ),
        "visible_label": PROVISIONAL_BANNER if provisional else FINAL_BANNER,
        "coverage_by_model": coverage,
        "conditional_threshold": args.threshold,
        "inference_unit": "endpoint",
        "models_included": sorted(profile_models),
        "neural_results_included": not provisional,
        "gate": {
            "traditional_only_provisional_figures": "PASS" if provisional else "NOT_APPLICABLE",
            "final_five_model_submission_figures": "PASS" if final_five_model else "HOLD",
            "reason": (
                "RF, XGB, and LGBM each have 22 endpoints x 5 seeds; unfinished "
                "neural results and the deprecated node-MPNN are excluded. Final "
                "regeneration remains blocked on complete GIN and Chemprop D-MPNN."
                if provisional
                else "All five formal model families pass exact 22-endpoint x 5-seed "
                "profile, conditional-reliability, and calibration source gates."
            ),
        },
        "sources": {
            "alignment_dir": str(alignment_dir.resolve()),
            "conditional_dir": str(conditional_dir.resolve()),
            "calibration_summary": str(calibration_path.resolve()),
        },
        "superseded_artifacts": {
            "old_combined_partial_fig01": str(
                (
                    ROOT
                    / "results"
                    / "debug"
                    / "superseded_combined_partial_fig01_20260714"
                ).resolve()
            ),
            "submission_eligible": False,
        },
        "outputs": [str(path.resolve()) for path in generated],
        "output_sha256": {
            str(path.resolve()): file_sha256(path) for path in generated
        },
        "limitations": (
            [
                "These are deliberately provisional traditional-only figures.",
                "GIN and Chemprop D-MPNN are excluded rather than mixed at incomplete endpoint coverage.",
                "The final five-model artwork must be regenerated after the locked neural matrix passes its gate.",
            ]
            if provisional
            else [
                "Reliability summaries use endpoint as the comparison unit.",
                "The final analysis is internally held-out but not an independent external validation study.",
            ]
        ),
    }
    manifest_path = out_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"status: {manifest['status']}")
    print(f"coverage: {coverage}")
    print(f"wrote {len(generated) + 1} files to {out_dir}")


if __name__ == "__main__":
    main()
