# At the very top of notebooks/1_data_extraction.ipynb
import sys, os
import pandas as pd
# Insert project_root (one level up) onto the import path
sys.path.insert(0, os.path.abspath("/cs/usr/roeizucker/new_storage/jupyter_notebooks/Tom_Hope_Project/refactored_code"))


import textwrap
import pytest
import yaml

from src.data_extractor import *
from src.utils.formatting import convert_string_column_to_list

@pytest.fixture(scope="module")
def cfg():
    config_path = os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    with open(config_path) as f:
        return yaml.safe_load(f)

def test_load_fasta(tmp_path):
    # 1. Create a temporary FASTA file
    fasta_content = textwrap.dedent("""\
        >seq1
        ACTG
        >seq2
        GGTTAAC
    """)
    fasta_file = tmp_path / "sample.fa"
    fasta_file.write_text(fasta_content)

    # 2. Call the function
    result = load_fasta(str(fasta_file))

    # 3. Assert we got exactly what we put in
    assert isinstance(result, dict)
    assert set(result.keys()) == {"seq1", "seq2"}
    assert result["seq1"] == "ACTG"
    assert result["seq2"] == "GGTTAAC"

    # 4. Also check that no extra whitespace sneaks in
    for seq in result.values():
        assert seq.isupper() and all(c in "ACGT" for c in seq)

def test_load_real_fasta(cfg):
    """
    Verify that load_fasta can read the real FASTA file specified
    by cfg['task']['assembly'] → cfg['paths']['assemblies'].
    """
    assemblies = cfg["paths"].get("assemblies", {})
    assert assemblies, "Missing 'assemblies' under paths in config"

    asm = cfg["task"]["assembly"]
    assert isinstance(asm, str), "'assembly' must be a string"
    assert asm in assemblies, f"task.assembly '{asm}' not defined in paths.assemblies"

    fasta_path = assemblies[asm]
    assert os.path.exists(fasta_path), f"FASTA file not found at {fasta_path}"

    sequences = load_fasta(fasta_path)
    assert isinstance(sequences, dict), "load_fasta should return a dict"
    assert sequences, "No sequences were loaded from the FASTA"
    # Sanity check: expect at least one standard chromosome key
    assert any(k.startswith("chr1") or k == "1" for k in sequences), \
        "Expected key like 'chr1' (or '1') in the loaded FASTA dict"

# def test_compare_to_old_pipeline(cfg):
#     # 1. Read expected output paths from config
#     train_path = cfg["paths"]["train_path"]
#     test_path  = cfg["paths"]["test_path"]

#     # 2. Make sure those files actually exist
#     assert os.path.exists(train_path), f"Expected train CSV not found at {train_path}"
#     assert os.path.exists(test_path),  f"Expected test  CSV not found at {test_path}"

#     # 3. Load the “ground truth” DataFrames
#     baseline_train = pd.read_csv(train_path)
#     baseline_test  = pd.read_csv(test_path)

#     # 4. Run your extractor on the same config (which now points at the small input CSV)
#     new_train, new_test = encode_cpg_extraction(cfg)

#     # 5. Compare the first N rows (e.g. 5 000) of each
#     N = 5000
#     pd.testing.assert_frame_equal(
#         new_train.head(N).reset_index(drop=True),
#         baseline_train.head(N).reset_index(drop=True),
#         check_dtype=False
#     )
#     pd.testing.assert_frame_equal(
#         new_test.head(N).reset_index(drop=True),
#         baseline_test.head(N).reset_index(drop=True),
#         check_dtype=False
#     )


def test_add_borders_to_multiple_instance_format_df(cfg):
    task = cfg["task"]
    testing_params = cfg["testing_params"]
    columns_to_sort_by = [ "starts", "ends"]
    df = pd.read_csv(task["input_csv"])
    df["value"] = df["methyl_rate"]
    new_df = combine_rows_to_multiple_instances_format(df,task["seq_size"]).reset_index(drop=True)
    add_borders_to_multiple_instance_format_df(new_df,task["seq_size"],True)
    new_df["starts"] = new_df["starts"].astype(str)
    new_df["ends"] = new_df["ends"].astype(str)
    new_df["values"] = new_df["values"].astype(str)
    # new_df["seq"] = new_df["seq"].astype(str)

    # Load the expected output DataFrame
    assert os.path.exists(testing_params["add_borders_to_multiple_instance_format_df_path"]), \
        f"Expected output CSV not found at {testing_params['add_borders_to_multiple_instance_format_df_path']}"
    other_df = pd.read_csv(testing_params["add_borders_to_multiple_instance_format_df_path"]).drop("Unnamed: 0",axis=1).reset_index(drop=True)
    other_df["starts"] = other_df["starts"].astype(str)
    other_df["ends"] = other_df["ends"].astype(str)
    other_df["values"] = other_df["values"].astype(str)
    # other_df["seq"] = other_df["seq"].astype(str)
    # Sort both DataFrames by the specified columns
    new_sorted = new_df.sort_values(by=columns_to_sort_by).reset_index(drop=True)  
    oter_sorted = other_df.sort_values(by=columns_to_sort_by).reset_index(drop=True)
    pd.testing.assert_frame_equal(  
        new_sorted,
        oter_sorted,
        check_dtype=False
    )

def test_combine_rows_to_multiple_instances_format(cfg):
    task = cfg["task"]
    testing_params = cfg["testing_params"]
    columns_to_sort_by = [ "starts", "ends"]
    df = pd.read_csv(task["input_csv"])
    df["value"] = df["methyl_rate"]
    new_df = combine_rows_to_multiple_instances_format(df,task["seq_size"]).reset_index(drop=True)
    new_df["starts"] = new_df["starts"].astype(str)
    new_df["ends"] = new_df["ends"].astype(str)
    new_df["values"] = new_df["values"].astype(str)

    # Load the expected output DataFrame
    assert os.path.exists(testing_params["instcombine_rows_to_multiple_instances_format_path"]), \
        f"Expected output CSV not found at {testing_params['instcombine_rows_to_multiple_instances_format_path']}"
    other_df = pd.read_csv(testing_params["instcombine_rows_to_multiple_instances_format_path"]).drop("Unnamed: 0",axis=1).reset_index(drop=True)
    other_df["starts"] = other_df["starts"].astype(str)
    other_df["ends"] = other_df["ends"].astype(str)
    other_df["values"] = other_df["values"].astype(str)
    # Sort both DataFrames by the specified columns
    new_sorted = new_df.sort_values(by=columns_to_sort_by).reset_index(drop=True)  
    oter_sorted = other_df.sort_values(by=columns_to_sort_by).reset_index(drop=True)
    pd.testing.assert_frame_equal(  
    new_sorted,
    oter_sorted,
    check_dtype=False
    )

def test_create_labels_from_multiple_instance_format_df(cfg):
    task = cfg["task"]
    testing_params = cfg["testing_params"]
    columns_to_sort_by = [ "starts", "ends"]
    df = pd.read_csv(task["input_csv"])
    df["value"] = df["methyl_rate"]
    new_df = combine_rows_to_multiple_instances_format(df,task["seq_size"]).reset_index(drop=True)
    add_borders_to_multiple_instance_format_df(new_df,task["seq_size"],True)
    create_labels_formultiple_instance_format_df(new_df,task["blank_label"])
    new_df["starts"] = new_df["starts"].astype(str)
    new_df["ends"] = new_df["ends"].astype(str)
    new_df["values"] = new_df["values"].astype(str)
    new_df["labels"] = new_df["labels"].astype(str)

    # Load the expected output DataFrame
    assert os.path.exists(testing_params["create_labels_formultiple_instance_format_df_path"]), \
        f"Expected output CSV not found at {testing_params['create_labels_formultiple_instance_format_df_path']}"
    other_df = pd.read_csv(testing_params["create_labels_formultiple_instance_format_df_path"]).drop("Unnamed: 0",axis=1).reset_index(drop=True)
    other_df["starts"] = other_df["starts"].astype(str)
    other_df["ends"] = other_df["ends"].astype(str)
    other_df["values"] = other_df["values"].astype(str)
    other_df["labels"] = other_df["labels"].astype(str)
    # Sort both DataFrames by the specified columns
    new_sorted = new_df.sort_values(by=columns_to_sort_by).reset_index(drop=True)  
    oter_sorted = other_df.sort_values(by=columns_to_sort_by).reset_index(drop=True)
    pd.testing.assert_frame_equal(  
        new_sorted,
        oter_sorted,
        check_dtype=False
    )

def test_compare_combined_baseline(cfg):
    # 1) Load the “old” train/test paths from the config
    train_path = cfg["paths"]["train_path"]
    test_path  = cfg["paths"]["test_path"]

    columns_to_fix = ['starts', 'ends', 'values']
    columns_to_sort_by = [ "start", "end"]

    assert os.path.exists(train_path), f"Baseline train CSV not found: {train_path}"
    assert os.path.exists(test_path),  f"Baseline test  CSV not found: {test_path}"

    baseline_train = pd.read_csv(train_path).drop("Unnamed: 0",axis=1)
    baseline_test  = pd.read_csv(test_path).drop("Unnamed: 0",axis=1)

    # 2) Build the full baseline set
    baseline_full = pd.concat([baseline_train, baseline_test], ignore_index=True)
    for col in columns_to_fix:
        baseline_full[col] = convert_string_column_to_list(baseline_full,col,float,"[]")
    baseline_full = baseline_full[["seq","labels","start", "end"]]
    # 3) Run your extractor to get new splits
    new_train, new_test = encode_cpg_extraction_from_intermidiate_paths(cfg)
    new_full = pd.concat([new_train, new_test], ignore_index=True)[["seq","labels","start", "end"]]
    # 4) Sort both on all columns to ensure consistent ordering
    
    baseline_sorted = baseline_full.sort_values(by=columns_to_sort_by).reset_index(drop=True)
    new_sorted = new_full.sort_values(by=columns_to_sort_by).reset_index(drop=True)

    baseline_sorted["seq"] = baseline_sorted["seq"].astype(str)
    new_sorted["seq"] = new_sorted["seq"].astype(str)
    baseline_sorted["labels"] = baseline_sorted["labels"].astype(str)
    new_sorted["labels"] = new_sorted["labels"].astype(str)
    # 5) Compare the first 5000 rows
    # N = 5000
    pd.testing.assert_frame_equal(
        new_sorted,
        baseline_sorted,
        check_dtype=False
    )
