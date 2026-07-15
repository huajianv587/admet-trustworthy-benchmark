from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from admet_benchmark import data_loader


def _write_cache(root: Path, task: str, *, classification: bool) -> pd.DataFrame:
    task_dir = root / "admet_group" / task.lower()
    task_dir.mkdir(parents=True)
    labels = [index % 2 for index in range(16)] if classification else list(range(16))
    train_val = pd.DataFrame(
        {"Drug": [f"C{index}" for index in range(16)], "Y": labels}
    )
    test = pd.DataFrame({"Drug": ["N0", "N1", "N2"], "Y": [0, 1, 0]})
    train_val.to_csv(task_dir / "train_val.csv", index=False)
    test.to_csv(task_dir / "test.csv", index=False)
    return train_val


def test_locked_cache_classification_membership_and_order_are_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(tmp_path, "AMES", classification=True)
    monkeypatch.setattr(
        data_loader,
        "_make_admet_group",
        lambda *_: (_ for _ in ()).throw(AssertionError("pytdc must not be called")),
    )

    first = data_loader.load_tdc_official(
        "AMES", data_dir=tmp_path, seed=3, split_source="locked_cache"
    )
    second = data_loader.load_tdc_official(
        "AMES", data_dir=tmp_path, seed=3, split_source="locked_cache"
    )

    # These indices pin the exact historical sklearn split used on the remote
    # neural workers, including the order written into prediction artifacts.
    assert first.train["Drug"].tolist() == [
        "C12", "C15", "C5", "C14", "C8", "C3", "C7",
        "C9", "C6", "C10", "C0", "C1", "C2", "C13",
    ]
    assert first.valid["Drug"].tolist() == ["C11", "C4"]
    assert first.test["Drug"].tolist() == ["N0", "N1", "N2"]
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.valid, second.valid)


def test_locked_cache_regression_is_unstratified_and_seeded(tmp_path: Path) -> None:
    _write_cache(tmp_path, "Caco2_Wang", classification=False)
    split = data_loader.load_tdc_official(
        "Caco2_Wang", data_dir=tmp_path, seed=3, split_source="locked_cache"
    )
    assert split.valid["Drug"].tolist() == ["C7", "C13"]
    assert split.train["Drug"].tolist() == [
        "C4", "C1", "C6", "C2", "C12", "C15", "C5",
        "C0", "C11", "C14", "C3", "C9", "C8", "C10",
    ]


def test_default_pytdc_source_fails_closed_instead_of_falling_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_cache(tmp_path, "AMES", classification=True)
    monkeypatch.setattr(
        data_loader,
        "_make_admet_group",
        lambda *_: (_ for _ in ()).throw(ImportError("not installed")),
    )

    with pytest.raises(ImportError, match="explicitly request split_source='locked_cache'"):
        data_loader.load_tdc_official("AMES", data_dir=tmp_path, seed=3)


def test_unknown_split_source_is_rejected_before_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown split_source"):
        data_loader.load_tdc_official(
            "AMES", data_dir=tmp_path, seed=3, split_source="automatic"  # type: ignore[arg-type]
        )


def test_formal_configs_and_launchers_pin_locked_cache() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("tdc_official.yaml", "gin.yaml", "chemprop_dmpnn.yaml"):
        payload = yaml.safe_load((root / "experiments" / "config" / name).read_text())
        assert payload["data"]["split_source"] == data_loader.LOCKED_CACHE_SPLIT_SOURCE
        assert (
            payload["data"]["split_algorithm"]
            == data_loader.LOCKED_CACHE_SPLIT_ALGORITHM
        )

    for name in (
        "slurm_gin_resume_12h.sh",
        "slurm_chemprop_dmpnn_resume_12h.sh",
        "run_locked_cache_traditional_full.sh",
    ):
        launcher = (root / "scripts" / name).read_text()
        assert "--split-source locked_cache" in launcher
