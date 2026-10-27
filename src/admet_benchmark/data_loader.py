from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd


TDC_TASK_META: dict[str, dict[str, str]] = {
    "Caco2_Wang": {"task_type": "regression", "metric": "mae", "category": "absorption"},
    "HIA_Hou": {"task_type": "classification", "metric": "auroc", "category": "absorption"},
    "Pgp_Broccatelli": {"task_type": "classification", "metric": "auroc", "category": "absorption"},
    "Bioavailability_Ma": {"task_type": "classification", "metric": "auroc", "category": "absorption"},
    "Lipophilicity_AstraZeneca": {"task_type": "regression", "metric": "mae", "category": "absorption"},
    "Solubility_AqSolDB": {"task_type": "regression", "metric": "mae", "category": "absorption"},
    "BBB_Martins": {"task_type": "classification", "metric": "auroc", "category": "distribution"},
    "PPBR_AZ": {"task_type": "regression", "metric": "mae", "category": "distribution"},
    "VDss_Lombardo": {"task_type": "regression", "metric": "spearman", "category": "distribution"},
    "CYP2C9_Veith": {"task_type": "classification", "metric": "auprc", "category": "metabolism"},
    "CYP2D6_Veith": {"task_type": "classification", "metric": "auprc", "category": "metabolism"},
    "CYP3A4_Veith": {"task_type": "classification", "metric": "auprc", "category": "metabolism"},
    "CYP2C9_Substrate_CarbonMangels": {"task_type": "classification", "metric": "auprc", "category": "metabolism"},
    "CYP2D6_Substrate_CarbonMangels": {"task_type": "classification", "metric": "auprc", "category": "metabolism"},
    "CYP3A4_Substrate_CarbonMangels": {"task_type": "classification", "metric": "auroc", "category": "metabolism"},
    "Half_Life_Obach": {"task_type": "regression", "metric": "spearman", "category": "excretion"},
    "Clearance_Hepatocyte_AZ": {"task_type": "regression", "metric": "spearman", "category": "excretion"},
    "Clearance_Microsome_AZ": {"task_type": "regression", "metric": "spearman", "category": "excretion"},
    "LD50_Zhu": {"task_type": "regression", "metric": "mae", "category": "toxicity"},
    "hERG": {"task_type": "classification", "metric": "auroc", "category": "toxicity"},
    "AMES": {"task_type": "classification", "metric": "auroc", "category": "toxicity"},
    "DILI": {"task_type": "classification", "metric": "auroc", "category": "toxicity"},
}

TDC_TASKS = list(TDC_TASK_META)

# Keep the pytdc and portable cached protocols explicit. The formal remote
# neural artifacts use the cached protocol; callers must select it directly.
PYTDC_SPLIT_SOURCE = "pytdc"
LOCKED_CACHE_SPLIT_SOURCE = "locked_cache"
SPLIT_SOURCE_CHOICES = (PYTDC_SPLIT_SOURCE, LOCKED_CACHE_SPLIT_SOURCE)
SplitSource = Literal["pytdc", "locked_cache"]

# This is public provenance for the cached split contract. A protocol change
# requires a new identifier so existing artifacts cannot be relabeled silently.
LOCKED_CACHE_SPLIT_ALGORITHM = (
    "sklearn_train_test_split_test_size_0.125_random_state_task_seed_"
    "classification_stratified_v1"
)


@dataclass(frozen=True)
class OfficialSplit:
    name: str
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    task_type: str
    metric: str
    category: str


def list_tdc_tasks() -> list[str]:
    return list(TDC_TASKS)


def _make_admet_group(path: str | Path):
    try:
        from tdc.benchmark_group import admet_group

        return admet_group(path=str(path))
    except ImportError:
        try:
            from tdc import BenchmarkGroup

            return BenchmarkGroup(name="ADMET_Group", path=str(path))
        except ImportError as exc:
            raise ImportError(
                "Missing pytdc. Install the project environment before loading TDC data."
            ) from exc


def _task_slug(task_name: str) -> str:
    return task_name.lower()


def _load_locked_cache_split(
    task_name: str, data_dir: str | Path, seed: int
) -> OfficialSplit:
    """Load the versioned portable split used by formal GPU artifacts.

    Only the cached train_val frame is partitioned. This is not silently labeled
    as pytdc's default split: the cached protocol is an explicit 7:1 split with
    classification stratification and task-seed random state.
    """
    from sklearn.model_selection import train_test_split

    meta = TDC_TASK_META[task_name]
    task_dir = Path(data_dir) / "admet_group" / _task_slug(task_name)
    train_val_path = task_dir / "train_val.csv"
    test_path = task_dir / "test.csv"
    if not train_val_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Missing cached TDC CSV files for {task_name}: {train_val_path}, {test_path}"
        )

    train_val = pd.read_csv(train_val_path)
    test = pd.read_csv(test_path)
    stratify = train_val["Y"] if meta["task_type"] == "classification" else None
    train, valid = train_test_split(
        train_val,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return OfficialSplit(
        name=task_name,
        train=train.reset_index(drop=True),
        valid=valid.reset_index(drop=True),
        test=test.reset_index(drop=True),
        task_type=meta["task_type"],
        metric=meta["metric"],
        category=meta["category"],
    )


def load_tdc_official(
    task_name: str,
    data_dir: str | Path = "data/tdc_official",
    seed: int = 42,
    split_source: SplitSource = PYTDC_SPLIT_SOURCE,
) -> OfficialSplit:
    """Load one TDC ADMET task using the official test split.

    The returned frames are not cleaned or canonicalized. This is deliberate:
    main benchmark tables must remain comparable with TDC official results.
    """
    if task_name not in TDC_TASK_META:
        valid = ", ".join(TDC_TASKS)
        raise ValueError(f"Unknown TDC ADMET task '{task_name}'. Valid tasks: {valid}")

    if split_source not in SPLIT_SOURCE_CHOICES:
        choices = ", ".join(SPLIT_SOURCE_CHOICES)
        raise ValueError(
            f"Unknown split_source '{split_source}'. Valid choices: {choices}"
        )
    if split_source == LOCKED_CACHE_SPLIT_SOURCE:
        return _load_locked_cache_split(task_name, data_dir, seed)

    meta = TDC_TASK_META[task_name]
    try:
        group = _make_admet_group(data_dir)
    except ImportError as exc:
        raise ImportError(
            "split_source='pytdc' was requested, but pytdc is unavailable. "
            "Install pytdc or explicitly request split_source='locked_cache'; "
            "the loader never changes split protocols implicitly."
        ) from exc
    benchmark = group.get(task_name)
    name = benchmark.get("name", task_name)
    train, valid = group.get_train_valid_split(
        benchmark=name,
        split_type="default",
        seed=seed,
    )
    return OfficialSplit(
        name=name,
        train=train,
        valid=valid,
        test=benchmark["test"],
        task_type=meta["task_type"],
        metric=meta["metric"],
        category=meta["category"],
    )


def infer_smiles_column(frame: pd.DataFrame) -> str:
    candidates = ["Drug", "SMILES", "smiles", "Drug_ID", "Molecule"]
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not infer SMILES column from: {list(frame.columns)}")


def infer_label_column(frame: pd.DataFrame) -> str:
    candidates = ["Y", "Label", "label", "target", "Target"]
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Could not infer label column from: {list(frame.columns)}")


def extract_xy(
    frame: pd.DataFrame,
    smiles_column: str | None = None,
    label_column: str | None = None,
) -> tuple[list[str], pd.Series]:
    smiles_column = smiles_column or infer_smiles_column(frame)
    label_column = label_column or infer_label_column(frame)
    return frame[smiles_column].astype(str).tolist(), frame[label_column]


def filter_tasks(
    tasks: Iterable[str] | None,
    categories: Iterable[str] | None = None,
) -> list[str]:
    selected = list(tasks) if tasks else list(TDC_TASKS)
    if categories:
        category_set = set(categories)
        selected = [
            task for task in selected if TDC_TASK_META[task]["category"] in category_set
        ]
    return selected
