import pytest
import os
import yaml

@pytest.fixture(scope="module")
def cfg():
    config_path = os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    with open(config_path) as f:
        return yaml.safe_load(f)

def test_paths_assemblies_defined(cfg):
    assert "paths" in cfg, "Missing 'paths' section in config"
    paths = cfg["paths"]
    assert "assemblies" in paths, "Missing 'assemblies' under paths"
    assemblies = paths["assemblies"]
    assert isinstance(assemblies, dict), "`paths.assemblies` should be a dict"
    assert assemblies, "`paths.assemblies` must define at least one assembly"
    # Check that each assembly name and path is a string
    for asm_name, asm_path in assemblies.items():
        assert isinstance(asm_name, str), f"Assembly name {asm_name!r} is not a string"
        assert isinstance(asm_path, str), f"Assembly path for {asm_name!r} is not a string"
        # Optionally check existence (uncomment if FASTA exists in test environment)
        # assert os.path.exists(asm_path), f"FASTA file for {asm_name} not found at {asm_path}"

def test_task_section(cfg):
    assert "task" in cfg, "Missing 'task' section in config"
    task = cfg["task"]
    required_keys = {
        "assembly",
        "input_csv",
        "value_column",
        "seq_size",
        "test_size",
        "blank_label",
        "use_fasta",
        "shuffle"
    }
    missing = required_keys - set(task.keys())
    assert not missing, f"Missing keys in 'task': {missing}"
    # Validate types
    assert isinstance(task["assembly"], str), "'assembly' must be a string"
    assert task["assembly"] in cfg["paths"]["assemblies"], \
        "'assembly' must be one of the keys in paths.assemblies"
    assert isinstance(task["input_csv"], str), "'input_csv' must be a string"
    assert isinstance(task["value_column"], str), "'value_column' must be a string"
    assert isinstance(task["seq_size"], int), "'seq_size' must be an integer"
    assert isinstance(task["test_size"], float), "'test_size' must be a float"
    assert isinstance(task["blank_label"], int), "'blank_label' must be an integer"
    assert isinstance(task["use_fasta"], bool), "'use_fasta' must be a boolean"
    assert isinstance(task["shuffle"], bool), "'shuffle' must be a boolean"

def test_random_state(cfg):
    # random_state is optional but if present must be int
    if "random_state" in cfg:
        assert isinstance(cfg["random_state"], int), "'random_state' must be an integer"


def test_task_assembly_defined(cfg):
    """
    Ensure that cfg['task']['assembly'] is a valid key under cfg['paths']['assemblies'].
    """
    assemblies = cfg["paths"].get("assemblies")
    assert assemblies, "Missing 'assemblies' under paths in config"

    asm = cfg["task"]["assembly"]
    assert isinstance(asm, str), "task.assembly must be a string"
    assert asm in assemblies, f"task.assembly '{asm}' not found in paths.assemblies"

