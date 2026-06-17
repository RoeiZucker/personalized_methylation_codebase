from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import yaml


DEFAULT_UBER_SCRIPT_PATH = (
    "/sci/nosnap/michall/roeizucker/jupyter_notebooks/"
    "Tom_Hope_Project/refactored_code/src/uber_project_creator_script.py"
)
DEFAULT_BIGWIG_BASE_PATH = "/sci/archive/michall/roeizucker/downloaded_datasets"

_FILES_NAMES_RE = re.compile(
    r"FILES_NAMES\s*=\s*'''''(?P<body>.*?)'''\.split\((?:\"(?:\\n|\n)\"|'(?:\\n|\n)')\)",
    re.DOTALL,
)
_BASE_FILE_PATH_RE = re.compile(r'^BASE_FILE_PATH\s*=\s*"(?P<path>[^"]+)"', re.MULTILINE)
_ANALYSIS_NAME_RE = re.compile(
    r"^(?P<sample>[^_]+)_atlas_eval_(?P<group>.+?)(?:_seq_\d+)?$"
)


def _sample_name_from_file_name(file_name: str) -> str:
    return file_name.replace(".hg38.bigwig", "").split("-")[-1]


def _normalize_group_name(group_name: str) -> str:
    if group_name.endswith("_kmer"):
        return group_name[: -len("_kmer")]
    if group_name.endswith("_window"):
        return group_name[: -len("_window")]
    return group_name


def _extract_files_names_block(script_path: str | os.PathLike[str]) -> List[str]:
    text = Path(script_path).read_text(encoding="utf-8")
    match = _FILES_NAMES_RE.search(text)
    if match is None:
        raise ValueError(f"Could not parse FILES_NAMES block from {script_path}")
    return [line for line in match.group("body").split("\n") if line.strip()]


def load_grouped_tissue_files(
    script_path: str | os.PathLike[str],
    min_group_size: int = 1,
) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for file_name in _extract_files_names_block(script_path):
        suffix_source = "_".join(file_name.split("_")[1:])
        group_name = "-".join(suffix_source.split("-")[:-1])
        grouped.setdefault(group_name, []).append(file_name)
    return {
        group_name: sorted(file_names)
        for group_name, file_names in grouped.items()
        if len(file_names) >= min_group_size
    }


def extract_base_file_path(script_path: str | os.PathLike[str]) -> str | None:
    text = Path(script_path).read_text(encoding="utf-8")
    match = _BASE_FILE_PATH_RE.search(text)
    if match is None:
        return None
    return match.group("path")


def resolve_group_bigwig_paths(
    group_name: str,
    held_out_sample: str,
    uber_script_path: str | os.PathLike[str] = DEFAULT_UBER_SCRIPT_PATH,
    base_file_path: str | os.PathLike[str] | None = None,
) -> Dict[str, object]:
    grouped = load_grouped_tissue_files(uber_script_path, min_group_size=1)
    normalized_group_name = _normalize_group_name(group_name)
    if normalized_group_name not in grouped:
        available = ", ".join(sorted(grouped.keys()))
        raise KeyError(f"Unknown tissue group '{group_name}'. Available groups: {available}")

    file_names = grouped[normalized_group_name]
    target_file_name = None
    atlas_file_names: List[str] = []
    for file_name in file_names:
        if _sample_name_from_file_name(file_name) == held_out_sample:
            target_file_name = file_name
        else:
            atlas_file_names.append(file_name)

    if target_file_name is None:
        sample_names = ", ".join(_sample_name_from_file_name(file_name) for file_name in file_names)
        raise KeyError(
            f"Held-out sample '{held_out_sample}' was not found in group '{group_name}'. "
            f"Available samples: {sample_names}"
        )

    resolved_base_path = (
        str(base_file_path)
        if base_file_path is not None
        else extract_base_file_path(uber_script_path) or DEFAULT_BIGWIG_BASE_PATH
    )
    return {
        "group_name": normalized_group_name,
        "held_out_sample": held_out_sample,
        "target_bigwig_path": os.path.join(resolved_base_path, target_file_name),
        "atlas_bigwig_paths": [os.path.join(resolved_base_path, file_name) for file_name in atlas_file_names],
    }


def _parse_legacy_config_identifiers(cfg: dict) -> tuple[str | None, str | None]:
    task = cfg.get("task", {})
    analysis_name = task.get("analysis_name", "")
    if analysis_name:
        match = _ANALYSIS_NAME_RE.match(analysis_name)
        if match is not None:
            return match.group("sample"), match.group("group")
        if "_atlas_eval_" in analysis_name:
            sample_name, remainder = analysis_name.split("_atlas_eval_", 1)
            group_name = remainder.rsplit("_seq_", 1)[0]
            return sample_name, group_name
    base_suffix = str(task.get("base_suffix", "")).lstrip("_") or None
    return None, base_suffix


def resolve_atlas_job_inputs(
    atlas_config_path: str | os.PathLike[str],
    uber_script_path: str | os.PathLike[str] = DEFAULT_UBER_SCRIPT_PATH,
    base_file_path: str | os.PathLike[str] | None = None,
) -> Dict[str, object]:
    cfg = yaml.safe_load(Path(atlas_config_path).read_text(encoding="utf-8"))
    paths = cfg.get("paths", {})
    task = cfg.get("task", {})
    testing_params = cfg.get("testing_params", {})

    target_bigwig_path = paths.get("target_bigwig_path")
    atlas_bigwig_paths = paths.get("atlas_bigwig_paths")
    held_out_sample = None
    group_name = str(task.get("base_suffix", "")).lstrip("_") or None

    if target_bigwig_path and atlas_bigwig_paths:
        target_name = Path(str(target_bigwig_path)).name
        if target_name.endswith(".hg38.bigwig"):
            held_out_sample = _sample_name_from_file_name(target_name)
    else:
        held_out_sample, fallback_group_name = _parse_legacy_config_identifiers(cfg)
        if group_name is None:
            group_name = fallback_group_name
        if held_out_sample is None or group_name is None:
            raise ValueError(
                "Could not resolve BigWig paths from atlas config. "
                "Provide target_bigwig_path/atlas_bigwig_paths or use a config with "
                "analysis_name/base_suffix that maps back to uber_project_creator_script.py."
            )
        resolved = resolve_group_bigwig_paths(
            group_name=group_name,
            held_out_sample=held_out_sample,
            uber_script_path=uber_script_path,
            base_file_path=base_file_path,
        )
        target_bigwig_path = resolved["target_bigwig_path"]
        atlas_bigwig_paths = resolved["atlas_bigwig_paths"]

    return {
        "group_name": group_name,
        "held_out_sample": held_out_sample,
        "target_bigwig_path": str(target_bigwig_path),
        "atlas_bigwig_paths": [str(path) for path in atlas_bigwig_paths],
        "variant_file_path": paths.get("variant_file_path"),
        "chromosomes": task.get("chromosomes"),
        "number_of_bins": int(task.get("number_of_bins", 5)),
        "top_rows": int(task.get("top_rows", -1)),
        "test_mode": bool(testing_params.get("test_mode", False)),
        "jump_sample": int(testing_params.get("jump_sample", -1)),
    }


def _empty_distribution_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "measure",
            "count",
            "mean",
            "std",
            "min",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "max",
        ]
    )


def summarize_distribution_values(values: Iterable[float], measure_name: str) -> pd.DataFrame:
    values_array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=np.float64)
    if values_array.size == 0:
        return _empty_distribution_summary()
    return pd.DataFrame(
        [
            {
                "measure": measure_name,
                "count": int(values_array.size),
                "mean": float(values_array.mean()),
                "std": float(values_array.std(ddof=1)) if values_array.size > 1 else 0.0,
                "min": float(values_array.min()),
                "p05": float(np.quantile(values_array, 0.05)),
                "p25": float(np.quantile(values_array, 0.25)),
                "p50": float(np.quantile(values_array, 0.50)),
                "p75": float(np.quantile(values_array, 0.75)),
                "p95": float(np.quantile(values_array, 0.95)),
                "max": float(values_array.max()),
            }
        ]
    )


def summarize_atlas_distribution(matched_df: pd.DataFrame) -> pd.DataFrame:
    if matched_df.empty:
        return _empty_distribution_summary()

    summary_frames = []
    for measure_name, column_name in (
        ("reference_std", "std"),
        ("atlas_mean_prediction", "atlas_mean"),
        ("held_out_target_rate", "target_value"),
    ):
        if column_name not in matched_df.columns:
            continue
        summary_frames.append(summarize_distribution_values(matched_df[column_name], measure_name))
    if not summary_frames:
        return _empty_distribution_summary()
    return pd.concat(summary_frames, ignore_index=True)


def load_variability_std_dataframe(
    variability_path: str | os.PathLike[str],
    top_rows: int = -1,
    test_mode: bool = False,
    jump_sample: int = -1,
) -> pd.DataFrame:
    df = pd.read_csv(variability_path)
    if "std" not in df.columns:
        raise ValueError(f"Variability CSV at {variability_path} does not contain a 'std' column.")

    keep_columns = [column for column in ("full_position", "window_id", "std", "high_diff") if column in df.columns]
    df = df[keep_columns].copy() if keep_columns else df.copy()
    df["std"] = pd.to_numeric(df["std"], errors="coerce")
    df = df.dropna(subset=["std"]).reset_index(drop=True)
    if top_rows is not None and top_rows > 0:
        df = df.head(top_rows).copy()
    if test_mode and jump_sample is not None and jump_sample > 0:
        df = df.iloc[::jump_sample].reset_index(drop=True)
    return df


def build_histogram_table(
    values: Iterable[float],
    bins: int,
    hist_range: tuple[float, float] | None = None,
    count_column_name: str = "count",
) -> pd.DataFrame:
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        return pd.DataFrame(columns=["bin_left", "bin_right", count_column_name])
    counts, edges = np.histogram(values_array, bins=bins, range=hist_range)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            count_column_name: counts.astype(int),
        }
    )
