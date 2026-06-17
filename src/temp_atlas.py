# At the very top of notebooks/1_data_extraction.ipynb
print("script started")
import sys, os
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
# Insert project_root (one level up) onto the import path
import numpy as np
import yaml
# from utils.dataset_utils import create_dataset, create_dataset_dict, create_tokenizer
from data_extractor import extract_data, save_huggingface_dataset, load_preprocessed_encode_cpg_dfs


var_df = pd.read_csv("/sci/labs/michall/roeizucker/huggingface_datasets_dir/huggingface_datasets_dir/_Liver-Hepatocytes_kmer/Z000000R3_per_varaint_variability_Liver-Hepatocytes_kmer_seq_5400_datasets.csv")

paths = '''/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652190_Liver-Endothelium-Z000000RB.hg38.bigwig
/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652233_Liver-Hepatocytes-Z000000R3.hg38.bigwig
/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652234_Liver-Hepatocytes-Z000000T3.hg38.bigwig
/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652235_Liver-Hepatocytes-Z0000043Q.hg38.bigwig
/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652236_Liver-Hepatocytes-Z0000044H.hg38.bigwig
/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652237_Liver-Hepatocytes-Z0000044M.hg38.bigwig
/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652238_Liver-Hepatocytes-Z00000431.hg38.bigwig
/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652307_Liver-Macrophages-Z0000043P.hg38.bigwig'''.split("\n")
dfs = load_preprocessed_encode_cpg_dfs(paths,[
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr22",
    "chr21",
    "chr20",
    "chr19",
    "chr18",
],"full_position",True)

target_path = "/sci/archive/michall/roeizucker/downloaded_datasets/GSM5652233_Liver-Hepatocytes-Z000000R3.hg38.bigwig"
target_idx = paths.index(target_path)
target_df = dfs[target_idx][["full_position", "methyl_rate"]]

other_methylation_df = pd.concat(
    [
        df.set_index("full_position")["methyl_rate"].rename(f"methyl_rate_{i}")
        for i, df in enumerate(dfs)
        if i != target_idx
    ],
    axis=1,
    copy=False,
)

methylation_comparison_df = pd.DataFrame(
    {
        "full_position": target_df["full_position"].to_numpy(copy=False),
        "target_methylation_label": target_df["methyl_rate"].to_numpy(copy=False),
        "average_other_methylation_label": other_methylation_df.reindex(
            target_df["full_position"]
        ).mean(axis=1).to_numpy(),
    }
)

valid_methylation_comparison_df = methylation_comparison_df.dropna(
    subset=["target_methylation_label", "average_other_methylation_label"]
).copy()
target_values = valid_methylation_comparison_df["target_methylation_label"].to_numpy(
    dtype=np.float32, copy=False
)
average_other_values = valid_methylation_comparison_df[
    "average_other_methylation_label"
].to_numpy(dtype=np.float32, copy=False)

methylation_comparison_metrics = {
    "mse": np.mean((target_values - average_other_values) ** 2),
    "mae": np.mean(np.abs(target_values - average_other_values)),
    "pearson_r": np.corrcoef(target_values, average_other_values)[0, 1]
    if len(target_values) > 1
    else np.nan,
}

comparison_with_variability_df = valid_methylation_comparison_df.merge(
    var_df[["full_position", "std"]],
    on="full_position",
    how="inner",
).rename(columns={"std": "variability_score"})
comparison_with_variability_df["error"] = (
    comparison_with_variability_df["target_methylation_label"]
    - comparison_with_variability_df["average_other_methylation_label"]
)
comparison_with_variability_df["absolute_error"] = comparison_with_variability_df[
    "error"
].abs()
comparison_with_variability_df["squared_error"] = (
    comparison_with_variability_df["error"] ** 2
)
comparison_with_variability_df["target_squared"] = (
    comparison_with_variability_df["target_methylation_label"] ** 2
)
comparison_with_variability_df["average_other_squared"] = (
    comparison_with_variability_df["average_other_methylation_label"] ** 2
)
comparison_with_variability_df["target_average_other_product"] = (
    comparison_with_variability_df["target_methylation_label"]
    * comparison_with_variability_df["average_other_methylation_label"]
)


def create_group_metrics(df, group_column_name):
    group_metrics = df.groupby(group_column_name, observed=False).agg(
        count=("variability_score", "size"),
        variability_min=("variability_score", "min"),
        variability_max=("variability_score", "max"),
        mse=("squared_error", "mean"),
        mae=("absolute_error", "mean"),
        sum_x=("target_methylation_label", "sum"),
        sum_y=("average_other_methylation_label", "sum"),
        sum_x2=("target_squared", "sum"),
        sum_y2=("average_other_squared", "sum"),
        sum_xy=("target_average_other_product", "sum"),
    ).reset_index()

    group_counts = group_metrics["count"].to_numpy(dtype=np.float64)
    pearson_numerator = (
        group_counts * group_metrics["sum_xy"].to_numpy(dtype=np.float64)
        - group_metrics["sum_x"].to_numpy(dtype=np.float64)
        * group_metrics["sum_y"].to_numpy(dtype=np.float64)
    )
    pearson_denominator = np.sqrt(
        (
            group_counts * group_metrics["sum_x2"].to_numpy(dtype=np.float64)
            - group_metrics["sum_x"].to_numpy(dtype=np.float64) ** 2
        )
        * (
            group_counts * group_metrics["sum_y2"].to_numpy(dtype=np.float64)
            - group_metrics["sum_y"].to_numpy(dtype=np.float64) ** 2
        )
    )
    group_metrics["pearson_r"] = np.where(
        (group_counts > 1) & (pearson_denominator > 0),
        pearson_numerator / pearson_denominator,
        np.nan,
    )
    return group_metrics.rename(columns={group_column_name: "variability_group"})[
        [
            "variability_group",
            "count",
            "variability_min",
            "variability_max",
            "mse",
            "mae",
            "pearson_r",
        ]
    ]


comparison_with_variability_df["variability_group"] = pd.qcut(
    comparison_with_variability_df["variability_score"].rank(method="first"),
    q=5,
    labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
)
methylation_group_metrics = create_group_metrics(
    comparison_with_variability_df,
    "variability_group",
)

max_variability = comparison_with_variability_df["variability_score"].max()
if max_variability > 0:
    equal_width_bin_edges = np.linspace(0.0, max_variability, 6)
    equal_width_labels = [
        f"{equal_width_bin_edges[i]:.5f}-{equal_width_bin_edges[i + 1]:.5f}"
        for i in range(len(equal_width_bin_edges) - 1)
    ]
    comparison_with_variability_df["equal_width_variability_group"] = pd.cut(
        comparison_with_variability_df["variability_score"],
        bins=equal_width_bin_edges,
        labels=equal_width_labels,
        include_lowest=True,
    )
else:
    comparison_with_variability_df["equal_width_variability_group"] = "0.00000-0.00000"

methylation_equal_width_group_metrics = create_group_metrics(
    comparison_with_variability_df,
    "equal_width_variability_group",
)

print("overall_metrics")
print(methylation_comparison_metrics)
print("percentile_group_metrics")
print(methylation_group_metrics)
print("equal_width_group_metrics")
print(methylation_equal_width_group_metrics)
