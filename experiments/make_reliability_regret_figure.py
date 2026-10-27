#!/usr/bin/env python3
"""Create the validation-selected reliability-regret submission heatmap.

The figure is intentionally independent of the older winner-mismatch plotting
workflow.  It reads one reliability-regret result directory, validates its gate
and model-pool metadata, and writes a deterministic vector SVG plus 300-dpi PNG
and vector PDF renderings.  No plotting package is required: the SVG is built
directly and rendered with a local Chromium/Chrome executable.

The same command works for the locked traditional-only analysis and, once
available, a complete five-model final-scope directory.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import signal
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "tables" / "reliability_regret_local_prelim"
DEFAULT_OUTPUT = ROOT / "results" / "figures" / "reliability_regret_traditional_provisional"

FIGURE_STEM = "fig01_validation_selected_reliability_regret"
# The final artwork is intentionally close to the manuscript column width.
# A 33-unit visible font at a 6.5-inch inclusion width renders at 7.35 pt:
# 33 * (6.5 * 72) / 2100.  Keeping the coordinate system compact avoids the
# unreadable 3--4 pt effective labels produced by the former 3900-unit canvas.
WIDTH = 2100
HEIGHT = 2020
WIDTH_IN = 7.0
DPI = 300
HEIGHT_IN = HEIGHT / DPI


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

MODEL_LABELS = {
    "rf": "RF",
    "xgboost": "XGB",
    "lightgbm": "LGBM",
    "gin": "GIN",
    "chemprop_dmpnn": "Chemprop D-MPNN",
}

# The secondary log-loss AURC is intentionally retained in source tables but
# omitted from the primary heatmap to avoid duplicating the selective-prediction
# construct. The core selective column uses Brier loss for classification and
# absolute error for regression.
DIMENSIONS = [
    {
        "id": "ece",
        "label": ["ECE"],
        "detail": "classification",
    },
    {
        "id": "brier",
        "label": ["Brier score"],
        "detail": "classification",
    },
    {
        "id": "conformal_coverage_error",
        "label": ["Validation-residual", "coverage error"],
        "detail": "regression",
    },
    {
        "id": "selective_relative_excess_aurc",
        "label": ["Selective", "excess AURC"],
        "detail": "Brier / MAE risk",
    },
    {
        "id": "applicability_alignment",
        "label": ["Morgan support/error", "rank association"],
        "detail": "Spearman",
    },
]

# Cividis-like sequential palette: perceptually ordered and readable under
# common forms of color-vision deficiency.
COLOR_STOPS = [
    (0.00, "#00204C"),
    (0.25, "#354E77"),
    (0.50, "#77766D"),
    (0.75, "#B8A85A"),
    (1.00, "#FDE737"),
]

INK = "#20242B"
MUTED = "#626B75"
GRID = "#D6DADE"
NA_FILL = "#F1F3F4"
CLASS_BLUE = "#0072B2"
REG_ORANGE = "#E69F00"
PRACTICAL_ORANGE = "#D55E00"
TIE_BLUE = "#56B4E9"
SIGNAL_GREEN = "#007A5E"
WARNING_FILL = "#FFF2CC"
WARNING_STROKE = "#B58100"


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def plotted_practical_signals(gate: dict[str, Any], core_ids: list[str]) -> list[str]:
    """Return gate signals represented in the primary heatmap.

    The locked gate may also contain the secondary log-loss selective-risk
    dimension.  It is intentionally disclosed in source tables and the figure
    manifest but omitted from the primary heatmap, so it must not be treated as
    a disagreement with the plotted core-dimension summary.
    """

    gate_signals = gate.get("effect_gate", {}).get(
        "dimensions_with_practical_effect_signal", []
    )
    return sorted(set(map(str, gate_signals)).intersection(core_ids))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(value: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(round(max(0, min(255, part)))):02X}" for part in value)


def regret_color(value: float) -> str:
    value = float(np.clip(value, 0.0, 1.0))
    for (left_x, left_color), (right_x, right_color) in zip(
        COLOR_STOPS[:-1], COLOR_STOPS[1:]
    ):
        if value <= right_x:
            fraction = (value - left_x) / (right_x - left_x)
            left = _hex_to_rgb(left_color)
            right = _hex_to_rgb(right_color)
            return _rgb_to_hex(
                tuple(a + fraction * (b - a) for a, b in zip(left, right))
            )
    return COLOR_STOPS[-1][1]


def contrast_text(fill: str) -> str:
    rgb = np.asarray(_hex_to_rgb(fill), dtype=float) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = float(np.dot(linear, [0.2126, 0.7152, 0.0722]))
    return "#FFFFFF" if luminance < 0.39 else INK


def ordered_tasks(regrets: pd.DataFrame) -> list[str]:
    available = set(regrets["task"].astype(str))
    result = [task for task in TASK_ORDER if task in available]
    result.extend(sorted(available.difference(result)))
    return result


def prepare_figure_data(
    result_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    regret_path = result_dir / "reliability_regret_by_endpoint.csv"
    summary_path = result_dir / "reliability_regret_summary.csv"
    gate_path = result_dir / "reliability_regret_gate.json"
    selection_path = result_dir / "validation_selection_by_task.csv"
    for path in (regret_path, summary_path, gate_path, selection_path):
        if not path.exists():
            raise FileNotFoundError(path)

    regrets = pd.read_csv(regret_path)
    summary = pd.read_csv(summary_path)
    selections = pd.read_csv(selection_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    require_columns(
        regrets,
        {
            "task",
            "task_type",
            "official_metric",
            "dimension",
            "model_pool",
            "validation_selected_model",
            "test_reliability_oracle_model",
            "reliability_regret",
            "normalized_regret",
            "practical_tolerance",
            "meaningful_regret",
            "tie_robust_meaningful_regret",
            "selected_exact_bootstrap_probability",
            "selected_tie_bootstrap_probability",
            "random_selection_expected_regret",
            "regret_fraction_of_random",
        },
        regret_path,
    )
    require_columns(
        summary,
        {
            "dimension",
            "n_endpoints",
            "mean_regret",
            "mean_normalized_regret",
            "meaningful_regret_count",
            "tie_robust_meaningful_count",
            "practical_effect_signal",
            "tie_robust_effect_signal",
            "random_baseline_retention_signal",
        },
        summary_path,
    )
    require_columns(
        selections,
        {
            "task",
            "validation_selected_model",
            "model_pool",
            "selected_exact_bootstrap_probability",
        },
        selection_path,
    )

    core_ids = [dimension["id"] for dimension in DIMENSIONS]
    plotted = regrets[regrets["dimension"].isin(core_ids)].copy()
    plotted["meaningful_regret"] = plotted["meaningful_regret"].map(parse_bool)
    plotted["tie_robust_meaningful_regret"] = plotted[
        "tie_robust_meaningful_regret"
    ].map(parse_bool)
    plotted["normalized_regret"] = pd.to_numeric(
        plotted["normalized_regret"], errors="raise"
    )
    if plotted.duplicated(["task", "dimension"]).any():
        duplicates = plotted.loc[
            plotted.duplicated(["task", "dimension"], keep=False),
            ["task", "dimension"],
        ]
        raise ValueError(f"Duplicate endpoint-dimension rows:\n{duplicates}")
    tasks = ordered_tasks(plotted)
    if len(tasks) != 22:
        raise ValueError(f"Expected 22 endpoints, found {len(tasks)}")
    if not np.all(
        (plotted["normalized_regret"] >= -1e-12)
        & (plotted["normalized_regret"] <= 1 + 1e-12)
    ):
        raise ValueError("normalized_regret must lie in [0, 1]")
    if np.any(
        plotted["tie_robust_meaningful_regret"] & ~plotted["meaningful_regret"]
    ):
        raise ValueError("Tie-robust practical regret must imply practical regret")
    expected_practical = (
        plotted["reliability_regret"] > plotted["practical_tolerance"]
    )
    if not np.array_equal(
        expected_practical.to_numpy(), plotted["meaningful_regret"].to_numpy()
    ):
        raise ValueError("Practical markers disagree with locked tolerances")

    selected_per_task = plotted.groupby("task")["validation_selected_model"].nunique()
    if int(selected_per_task.max()) != 1:
        raise ValueError("Validation-selected model changes across dimensions")
    selection_lookup = selections.set_index("task")["validation_selected_model"]
    plotted_lookup = plotted.drop_duplicates("task").set_index("task")[
        "validation_selected_model"
    ]
    if not plotted_lookup.equals(selection_lookup.reindex(plotted_lookup.index)):
        raise ValueError("Regret table and validation selection table disagree")

    # Produce a complete 22 x 5 source grid, including explicit N/A rows.
    grid_index = pd.MultiIndex.from_product(
        [tasks, core_ids], names=["task", "dimension"]
    ).to_frame(index=False)
    source_grid = grid_index.merge(plotted, on=["task", "dimension"], how="left")
    source_grid["applicable"] = source_grid["normalized_regret"].notna()
    source_grid["task_label"] = source_grid["task"].map(TASK_LABELS).fillna(
        source_grid["task"]
    )

    summary_core = summary[summary["dimension"].isin(core_ids)].copy()
    summary_core["dimension"] = pd.Categorical(
        summary_core["dimension"], categories=core_ids, ordered=True
    )
    summary_core = summary_core.sort_values("dimension").reset_index(drop=True)
    if list(summary_core["dimension"].astype(str)) != core_ids:
        raise ValueError("Regret summary is missing one or more core dimensions")

    gate_signals = plotted_practical_signals(gate, core_ids)
    csv_signals = sorted(
        summary_core.loc[
            summary_core["practical_effect_signal"].map(parse_bool), "dimension"
        ].astype(str)
    )
    if gate_signals != csv_signals:
        raise ValueError(
            f"Gate/summary practical signals disagree: {gate_signals} vs {csv_signals}"
        )
    return source_grid, summary_core, gate, selections


def model_pool_text(gate: dict[str, Any], selections: pd.DataFrame) -> str:
    signatures = gate.get("coverage", {}).get("selection_pool_signatures", [])
    if len(signatures) == 1:
        models = [MODEL_LABELS.get(model, model) for model in signatures[0].split(";")]
        return "Model pool on every endpoint: " + " + ".join(models)
    readable = [
        "+".join(MODEL_LABELS.get(model, model) for model in signature.split(";"))
        for signature in signatures
    ]
    return "MODEL POOLS VARY BY ENDPOINT: " + " | ".join(readable)


def scope_banner(gate: dict[str, Any]) -> tuple[str, str]:
    scope = str(gate.get("scope", "unknown"))
    coverage = gate.get("coverage", {})
    complete = bool(coverage.get("complete_expected_model_endpoint_matrix", False))
    if scope == "traditional_only":
        return (
            "TRADITIONAL-MODEL EXPLORATORY MATRIX · NOT THE FINAL FIVE-MODEL ANALYSIS",
            WARNING_FILL,
        )
    if scope == "final" and complete:
        return (
            "COMPLETE FIVE-MODEL MATRIX · DESCRIPTIVE BENCHMARK AUDIT",
            "#E5F4EF",
        )
    return (
        "INCOMPLETE FIVE-MODEL MATRIX · MODEL COVERAGE IS PARTIAL",
        "#FCE8E6",
    )


def _text(
    x: float,
    y: float,
    value: str,
    size: float,
    *,
    anchor: str = "start",
    fill: str = INK,
    weight: int = 400,
    family: str = "Arial, Helvetica, 'DejaVu Sans', sans-serif",
    letter_spacing: float | None = None,
) -> str:
    spacing = (
        f' letter-spacing="{letter_spacing}"' if letter_spacing is not None else ""
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'fill="{fill}" font-family="{family}" font-size="{size:.1f}" '
        f'font-weight="{weight}"{spacing}>{html.escape(str(value))}</text>'
    )


def _diamond(cx: float, cy: float, radius: float, fill: str, stroke: str) -> str:
    points = [
        (cx, cy - radius),
        (cx + radius, cy),
        (cx, cy + radius),
        (cx - radius, cy),
    ]
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return (
        f'<polygon points="{encoded}" fill="{fill}" stroke="{stroke}" '
        'stroke-width="4"/>'
    )


def build_svg(
    source_grid: pd.DataFrame,
    summary: pd.DataFrame,
    gate: dict[str, Any],
    selections: pd.DataFrame,
    source_digest: str,
) -> str:
    tasks = list(dict.fromkeys(source_grid["task"].astype(str)))
    dimensions = [dimension["id"] for dimension in DIMENSIONS]
    lookup = source_grid.set_index(["task", "dimension"])
    summary_lookup = summary.set_index("dimension")
    selection_lookup = selections.set_index("task")

    # Compact, manuscript-width-first layout.  Every visible label is at least
    # 33 SVG units (7.35 pt at the 6.5-inch manuscript inclusion width).
    heat_x = 650
    heat_y = 410
    cell_w = 280
    cell_h = 58
    heat_w = cell_w * len(dimensions)
    heat_h = cell_h * len(tasks)
    heat_right = heat_x + heat_w
    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_IN}in" '
        f'height="{HEIGHT_IN}in" viewBox="0 0 {WIDTH} {HEIGHT}" '
        'role="img" aria-labelledby="figure-title figure-desc">',
        '<title id="figure-title">Validation-selected reliability regret across 22 ADMET endpoints</title>',
        '<desc id="figure-desc">Heatmap of normalized dimension-specific candidate-pool regret after validation-only model selection. Circles mark heuristic-threshold exceedance and diamonds mark effects robust to validation-score ties.</desc>',
        f'<metadata id="source-digest">SHA-256 {source_digest}</metadata>',
        "<defs>",
        '<pattern id="na-hatch" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">',
        f'<rect width="12" height="12" fill="{NA_FILL}"/>',
        f'<line x1="0" y1="0" x2="0" y2="12" stroke="{GRID}" stroke-width="3"/>',
        "</pattern>",
        "</defs>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#FFFFFF"/>',
    ]

    lines.append(
        _text(
            20,
            52,
            "Reliability regret after validation-only model selection",
            46,
            weight=500,
        )
    )
    lines.append(
        _text(
            20,
            98,
            "Dimension-specific candidate-pool comparison; endpoint is the comparison unit",
            33,
            fill=MUTED,
        )
    )
    banner_text, banner_fill = scope_banner(gate)
    lines.append(
        f'<rect x="20" y="118" width="{WIDTH - 40}" height="52" rx="7" '
        f'fill="{banner_fill}" stroke="{WARNING_STROKE}" stroke-width="2"/>'
    )
    lines.append(_text(38, 155, banner_text, 33, weight=500, letter_spacing=0.3))
    pool_text = model_pool_text(gate, selections)
    lines.append(_text(20, 211, pool_text, 33, weight=500))
    coverage = gate.get("coverage", {})
    coverage_by_model = coverage.get("model_task_coverage", {})
    coverage_order = ["rf", "xgboost", "lightgbm", "gin", "chemprop_dmpnn"]
    coverage_parts = [
        f"{MODEL_LABELS.get(model, model)} {coverage_by_model[model]}/22"
        for model in coverage_order
        if model in coverage_by_model
    ]
    coverage_parts.extend(
        f"{MODEL_LABELS.get(model, model)} {count}/22"
        for model, count in sorted(coverage_by_model.items())
        if model not in coverage_order
    )
    lines.append(
        _text(
            20,
            254,
            "Coverage: " + " / ".join(coverage_parts) + "; ≥5 seeds per included task-model",
            33,
            fill=MUTED,
        )
    )

    lines.append(_text(74, 392, "Endpoint", 33, weight=500))
    lines.append(
        _text(630, 392, "Validation choice", 33, anchor="end", weight=500)
    )
    practical_signals = set(
        gate.get("effect_gate", {}).get(
            "dimensions_with_practical_effect_signal", []
        )
    )
    for column, dimension in enumerate(DIMENSIONS):
        center = heat_x + column * cell_w + cell_w / 2
        label_lines = dimension["label"]
        base_y = 281 if len(label_lines) == 2 else 300
        for line_index, label in enumerate(label_lines):
            lines.append(
                _text(
                    center,
                    base_y + line_index * 37,
                    label,
                    33,
                    anchor="middle",
                    weight=500,
                )
            )
        lines.append(
            _text(
                center,
                355,
                dimension["detail"],
                33,
                anchor="middle",
                fill=MUTED,
            )
        )
        stats = summary_lookup.loc[dimension["id"]]
        practical = int(stats["meaningful_regret_count"])
        tie = int(stats["tie_robust_meaningful_count"])
        eligible = int(stats["n_endpoints"])
        lines.append(
            _text(
                center,
                392,
                f"● {practical}   ◆ {tie}   / {eligible}",
                33,
                anchor="middle",
                fill=MUTED,
            )
        )
        if dimension["id"] in practical_signals:
            lines.append(
                f'<line x1="{center - 105:.1f}" y1="404" x2="{center + 105:.1f}" '
                f'y2="404" stroke="{SIGNAL_GREEN}" stroke-width="6"/>'
            )

    # Endpoint rows and cells.
    classification_count = 0
    for row_index, task in enumerate(tasks):
        y = heat_y + row_index * cell_h
        available = source_grid[source_grid["task"] == task]
        task_type_values = available["task_type"].dropna().astype(str).unique()
        if len(task_type_values) != 1:
            raise ValueError(f"Could not determine a unique task type for {task}")
        task_type = task_type_values[0]
        if task_type == "classification":
            classification_count += 1
        stripe_color = CLASS_BLUE if task_type == "classification" else REG_ORANGE
        type_letter = "C" if task_type == "classification" else "R"
        lines.append(
            f'<rect x="20" y="{y + 4:.1f}" width="7" height="{cell_h - 8}" '
            f'fill="{stripe_color}"/>'
        )
        lines.append(
            _text(38, y + 42, type_letter, 33, fill=stripe_color, weight=500)
        )
        lines.append(
            _text(74, y + 42, TASK_LABELS.get(task, task), 33, weight=400)
        )
        selected = str(selection_lookup.loc[task, "validation_selected_model"])
        lines.append(
            _text(
                630,
                y + 42,
                MODEL_LABELS.get(selected, selected),
                33,
                anchor="end",
                fill=MUTED,
                weight=500,
            )
        )
        for column, dimension in enumerate(dimensions):
            x = heat_x + column * cell_w
            cell = lookup.loc[(task, dimension)]
            value = cell["normalized_regret"]
            if pd.isna(value):
                lines.append(
                    f'<rect x="{x + 4:.1f}" y="{y + 4:.1f}" '
                    f'width="{cell_w - 8:.1f}" height="{cell_h - 8:.1f}" '
                    'fill="url(#na-hatch)"/>'
                )
                lines.append(
                    _text(
                        x + cell_w / 2,
                        y + 42,
                        "N/A",
                        33,
                        anchor="middle",
                        fill="#8A929A",
                    )
                )
                continue
            fill = regret_color(float(value))
            text_fill = contrast_text(fill)
            lines.append(
                f'<rect x="{x + 4:.1f}" y="{y + 4:.1f}" '
                f'width="{cell_w - 8:.1f}" height="{cell_h - 8:.1f}" '
                f'rx="4" fill="{fill}"/>'
            )
            lines.append(
                _text(
                    x + cell_w / 2,
                    y + 42,
                    f"{float(value):.2f}",
                    33,
                    anchor="middle",
                    fill=text_fill,
                    weight=500,
                )
            )
            if parse_bool(cell["meaningful_regret"]):
                lines.append(
                    f'<circle cx="{x + cell_w - 17:.1f}" cy="{y + 15:.1f}" r="7" '
                    f'fill="{PRACTICAL_ORANGE}" stroke="#FFFFFF" stroke-width="2"/>'
                )
            if parse_bool(cell["tie_robust_meaningful_regret"]):
                lines.append(
                    _diamond(
                        x + cell_w - 17,
                        y + cell_h - 14,
                        8,
                        TIE_BLUE,
                        "#FFFFFF",
                    )
                )

    # Classification/regression separator and fine grid.
    split_y = heat_y + classification_count * cell_h
    lines.append(
        f'<line x1="20" y1="{split_y:.1f}" x2="{heat_right:.1f}" y2="{split_y:.1f}" '
        f'stroke="{INK}" stroke-width="4"/>'
    )
    for column in range(len(dimensions) + 1):
        x = heat_x + column * cell_w
        lines.append(
            f'<line x1="{x:.1f}" y1="{heat_y:.1f}" x2="{x:.1f}" '
            f'y2="{heat_y + heat_h:.1f}" stroke="#FFFFFF" stroke-width="2"/>'
        )

    # Color scale.
    legend_y = heat_y + heat_h + 28
    segments = 100
    for index in range(segments):
        x = heat_x + heat_w * index / segments
        segment_width = heat_w / segments + 1
        lines.append(
            f'<rect x="{x:.1f}" y="{legend_y:.1f}" width="{segment_width:.1f}" '
            f'height="22" fill="{regret_color(index / (segments - 1))}"/>'
        )
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = heat_x + tick * heat_w
        lines.append(
            f'<line x1="{x:.1f}" y1="{legend_y + 22:.1f}" x2="{x:.1f}" '
            f'y2="{legend_y + 34:.1f}" stroke="{INK}" stroke-width="2"/>'
        )
        lines.append(
            _text(x, legend_y + 72, f"{tick:.2f}", 33, anchor="middle", fill=MUTED)
        )
    lines.append(
        _text(
            heat_x + heat_w / 2,
            legend_y + 116,
            "Normalized dimension-specific regret (0 = pool reference; 1 = worst candidate)",
            33,
            anchor="middle",
            fill=MUTED,
        )
    )

    # Marker and type legends; shape and text accompany all color encodings.
    footer_y = legend_y + 171
    lines.append(
        f'<circle cx="38" cy="{footer_y - 10:.1f}" r="8" fill="{PRACTICAL_ORANGE}" '
        'stroke="#FFFFFF" stroke-width="2"/>'
    )
    lines.append(
        _text(
            61,
            footer_y,
            "Magnitude marker: raw regret exceeds the analysis-locked threshold",
            33,
        )
    )
    lines.append(_diamond(38, footer_y + 43, 9, TIE_BLUE, "#FFFFFF"))
    lines.append(
        _text(
            61,
            footer_y + 53,
            "Tie-robust: remains above threshold for the best validation-tied model",
            33,
        )
    )
    lines.append(
        f'<circle cx="38" cy="{footer_y + 87:.1f}" r="8" fill="{CLASS_BLUE}"/>'
    )
    lines.append(_text(61, footer_y + 97, "Classification", 33))
    lines.append(
        f'<circle cx="342" cy="{footer_y + 87:.1f}" r="8" fill="{REG_ORANGE}"/>'
    )
    lines.append(_text(365, footer_y + 97, "Regression", 33))
    lines.append(
        f'<line x1="660" y1="{footer_y + 87:.1f}" x2="760" y2="{footer_y + 87:.1f}" '
        f'stroke="{SIGNAL_GREEN}" stroke-width="6"/>'
    )
    lines.append(
        _text(
            782,
            footer_y + 97,
            "Dimension meets the secondary heuristic magnitude screen",
            33,
        )
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def locate_renderer(explicit: str | None = None) -> Path:
    """Locate a browser renderer, with ImageMagick as a headless fallback.

    The analysis runner may be on a compute node without Chrome/Chromium.  The
    SVG is self-contained, so ImageMagick can still produce the required raster
    and PDF artifacts without changing the figure source or analysis tables.
    """
    candidates = [
        explicit,
        os.environ.get("CHROME_BIN"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    imagemagick = shutil.which("magick") or shutil.which("convert")
    if imagemagick:
        return Path(imagemagick)
    raise FileNotFoundError(
        "Chrome/Chromium or ImageMagick is required to render PNG/PDF. "
        "Set CHROME_BIN or pass --chrome."
    )


def render_svg(svg_path: Path, png_path: Path, pdf_path: Path, chrome: Path) -> None:
    if chrome.name in {"convert", "magick"}:
        # ImageMagick reads the SVG's explicit width/height.  Density is set so
        # the PNG has the same 300-dpi physical size as the browser path.
        common = [str(chrome), "-background", "white", "-density", str(DPI)]
        subprocess.run(
            [*common, str(svg_path), "-alpha", "remove", "-alpha", "off", str(png_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        subprocess.run(
            [*common, str(svg_path), "-alpha", "remove", "-alpha", "off", str(pdf_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return

    svg = svg_path.read_text(encoding="utf-8")
    wrapper = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>Validation-selected reliability regret</title><style>"
        f"@page{{size:{WIDTH_IN}in {HEIGHT_IN}in;margin:0}}"
        "html,body{margin:0;padding:0;overflow:hidden;background:#fff}"
        f"@media screen{{html,body{{width:{WIDTH}px;height:{HEIGHT}px}}"
        f"svg{{display:block;width:{WIDTH}px;height:{HEIGHT}px}}}}"
        f"@media print{{html,body{{width:{WIDTH_IN}in;height:{HEIGHT_IN}in}}"
        f"svg{{display:block;width:{WIDTH_IN}in;height:{HEIGHT_IN}in}}}}"
        "</style></head><body>"
        + svg
        + "</body></html>"
    )
    with tempfile.TemporaryDirectory(prefix="regret_figure_chrome_") as directory:
        directory_path = Path(directory)
        html_path = directory_path / "figure.html"
        html_path.write_text(wrapper, encoding="utf-8")

        def run_until_written(command: list[str], output: Path) -> None:
            """Accept a stable renderer output even if macOS Chrome stays resident."""
            output.unlink(missing_ok=True)
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            deadline = time.monotonic() + 90
            previous_size = -1
            stable_polls = 0
            try:
                while time.monotonic() < deadline:
                    return_code = process.poll()
                    if output.exists() and output.stat().st_size > 1024:
                        size = output.stat().st_size
                        stable_polls = stable_polls + 1 if size == previous_size else 0
                        previous_size = size
                        if stable_polls >= 4:
                            return
                    if return_code is not None:
                        if return_code != 0:
                            raise subprocess.CalledProcessError(return_code, command)
                        if output.exists() and output.stat().st_size > 1024:
                            return
                        raise RuntimeError(f"Chrome exited without writing {output}")
                    time.sleep(0.25)
                raise TimeoutError(f"Chrome did not finish {output} within 90 seconds")
            finally:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=5)

        common = [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-default-apps",
            "--disable-extensions",
            "--hide-scrollbars",
            "--metrics-recording-only",
            "--mute-audio",
            "--no-first-run",
            "--no-default-browser-check",
            "--no-sandbox",
        ]
        run_until_written(
            [
                *common,
                f"--user-data-dir={directory_path / 'profile_png'}",
                f"--window-size={WIDTH},{HEIGHT}",
                "--force-device-scale-factor=1",
                f"--screenshot={png_path.resolve()}",
                html_path.resolve().as_uri(),
            ],
            png_path,
        )
        run_until_written(
            [
                *common,
                f"--user-data-dir={directory_path / 'profile_pdf'}",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path.resolve()}",
                html_path.resolve().as_uri(),
            ],
            pdf_path,
        )

    # Record the intended 300-dpi physical resolution in the PNG metadata.
    try:
        from PIL import Image

        with Image.open(png_path) as image:
            if image.size != (WIDTH, HEIGHT):
                raise ValueError(
                    f"Unexpected PNG dimensions {image.size}; expected {(WIDTH, HEIGHT)}"
                )
            image.save(png_path, dpi=(DPI, DPI), optimize=True)
    except ImportError:
        pass


def write_manifest(
    output_dir: Path,
    result_dir: Path,
    gate: dict[str, Any],
    source_grid: pd.DataFrame,
    summary: pd.DataFrame,
    output_paths: list[Path],
) -> Path:
    regret_path = result_dir / "reliability_regret_by_endpoint.csv"
    gate_path = result_dir / "reliability_regret_gate.json"
    applicable = source_grid[source_grid["applicable"]].copy()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure": "Validation-selected reliability-regret heatmap",
        "figure_stem": FIGURE_STEM,
        "source_result_dir": str(result_dir.resolve()),
        "source_files": {
            "reliability_regret_by_endpoint.csv": file_sha256(regret_path),
            "reliability_regret_gate.json": file_sha256(gate_path),
        },
        "scope": gate.get("scope"),
        "gate_status": gate.get("status"),
        "final_submission_gate": gate.get("final_submission_gate"),
        "model_coverage": gate.get("coverage"),
        "analysis_timeline_policy": gate.get("analysis_timeline_policy"),
        "encoding": {
            "color": "normalized_regret, fixed 0-1 cividis-like sequential scale",
            "circle": "raw reliability_regret > locked heuristic threshold",
            "diamond": "tie_best_case_regret > locked heuristic threshold",
            "green_header_rule": (
                "dimension meets the secondary heuristic magnitude screen"
            ),
            "na_hatch": "dimension is inapplicable to endpoint task type",
        },
        "dimensions": [dimension["id"] for dimension in DIMENSIONS],
        "excluded_secondary_dimensions": [
            "selective_relative_excess_aurc_log_loss"
        ],
        "counts": {
            "endpoints": int(source_grid["task"].nunique()),
            "grid_cells": int(len(source_grid)),
            "applicable_cells": int(source_grid["applicable"].sum()),
            "na_cells": int((~source_grid["applicable"]).sum()),
            "practical_cells": int(applicable["meaningful_regret"].map(parse_bool).sum()),
            "tie_robust_cells": int(
                applicable["tie_robust_meaningful_regret"].map(parse_bool).sum()
            ),
        },
        "dimension_statistics": [
            {
                "dimension": str(row["dimension"]),
                "n_endpoints": int(row["n_endpoints"]),
                "mean_regret": float(row["mean_regret"]),
                "mean_normalized_regret": float(row["mean_normalized_regret"]),
                "practical_count": int(row["meaningful_regret_count"]),
                "tie_robust_count": int(row["tie_robust_meaningful_count"]),
                "practical_effect_signal": parse_bool(row["practical_effect_signal"]),
                "tie_robust_effect_signal": parse_bool(
                    row["tie_robust_effect_signal"]
                ),
                "random_baseline_retention_signal": parse_bool(
                    row["random_baseline_retention_signal"]
                ),
            }
            for _, row in summary.iterrows()
        ],
        "render": {
            "width_pixels": WIDTH,
            "height_pixels": HEIGHT,
            "width_inches": WIDTH_IN,
            "height_inches": HEIGHT_IN,
            "png_dpi": DPI,
            "palette": "cividis-like; color-vision-deficiency-safe sequential",
            "font": "Arial/Helvetica/DejaVu Sans fallback",
        },
        "validation_checks": [
            "22 unique endpoints",
            "one validation-selected model per endpoint across dimensions",
            "normalized regret bounded in [0,1]",
            "magnitude markers exactly reproduce locked raw heuristic thresholds",
            "tie-robust markers imply magnitude markers",
            "gate and summary magnitude-signal lists agree",
            "scope, model pool, and coverage are printed on the figure",
        ],
        "outputs": {
            path.name: {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in output_paths
        },
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--chrome", default=None)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_grid, summary, gate, selections = prepare_figure_data(result_dir)
    regret_path = result_dir / "reliability_regret_by_endpoint.csv"
    source_digest = file_sha256(regret_path)
    svg = build_svg(source_grid, summary, gate, selections, source_digest)

    svg_path = output_dir / f"{FIGURE_STEM}.svg"
    png_path = output_dir / f"{FIGURE_STEM}.png"
    pdf_path = output_dir / f"{FIGURE_STEM}.pdf"
    source_path = output_dir / "figure_source_statistics.csv"
    dimension_path = output_dir / "figure_dimension_statistics.csv"
    svg_path.write_text(svg, encoding="utf-8")
    source_grid.to_csv(source_path, index=False)
    summary.to_csv(dimension_path, index=False)
    render_svg(svg_path, png_path, pdf_path, locate_renderer(args.chrome))
    manifest_path = write_manifest(
        output_dir,
        result_dir,
        gate,
        source_grid,
        summary,
        [svg_path, png_path, pdf_path, source_path, dimension_path],
    )
    print(f"scope: {gate.get('scope')}")
    print(f"gate status: {gate.get('status')}")
    print(f"endpoints: {source_grid['task'].nunique()}")
    print(f"applicable cells: {int(source_grid['applicable'].sum())}/{len(source_grid)}")
    print(f"wrote: {svg_path}")
    print(f"wrote: {png_path}")
    print(f"wrote: {pdf_path}")
    print(f"wrote: {manifest_path}")


if __name__ == "__main__":
    main()
