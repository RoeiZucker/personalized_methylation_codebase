import pandas as pd

try:
    from constants import DEFAULT_FULL_POSITION_COLUMN_NAME
except ImportError:
    from ..constants import DEFAULT_FULL_POSITION_COLUMN_NAME


def load_variability_dict(variant_file_path, use_variant_filtering, verbose=False):
    if not use_variant_filtering:
        return None
    if not variant_file_path:
        raise ValueError("variant_file_path is required when variant filtering is enabled")

    df = pd.read_csv(variant_file_path)
    variability_dict = (
        df[[DEFAULT_FULL_POSITION_COLUMN_NAME, "std"]]
        .set_index(DEFAULT_FULL_POSITION_COLUMN_NAME)
        .to_dict(orient="index")
    )
    variability_dict = {
        key: value["std"]
        for key, value in variability_dict.items()
    }
    if verbose:
        print("finished loading variability dict", flush=True)
    return variability_dict
