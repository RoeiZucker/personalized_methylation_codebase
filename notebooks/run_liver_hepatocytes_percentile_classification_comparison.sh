#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOK_PATH="${SCRIPT_DIR}/10_liver_hepatocytes_percentile_classification_comparison.ipynb"
RESULTS_ROOT="${SCRIPT_DIR}/results/liver_hepatocytes_percentile_classification_comparison"

usage() {
  cat <<'EOF'
Usage:
  ./run_liver_hepatocytes_percentile_classification_comparison.sh [--clean]

Options:
  --clean    Remove the existing results directory before running.
  -h, --help Show this help message.
EOF
}

clean_results=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      clean_results=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${NOTEBOOK_PATH}" ]]; then
  echo "Notebook not found: ${NOTEBOOK_PATH}" >&2
  exit 1
fi

if [[ "${clean_results}" == "true" ]]; then
  echo "Removing previous results at ${RESULTS_ROOT}"
  rm -rf "${RESULTS_ROOT}"
fi

mkdir -p "${RESULTS_ROOT}"

echo "Executing notebook:"
echo "  ${NOTEBOOK_PATH}"

NOTEBOOK_PATH_ENV="${NOTEBOOK_PATH}" python3 - <<'PY'
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

notebook_path = Path(os.environ["NOTEBOOK_PATH_ENV"])
nb = nbformat.read(notebook_path, as_version=4)

client = NotebookClient(
    nb,
    timeout=None,
    kernel_name="python3",
    resources={"metadata": {"path": str(notebook_path.parent)}},
)

client.execute()
nbformat.write(nb, notebook_path)
print(f"Executed notebook successfully: {notebook_path}")
PY

echo
echo "Verifying outputs under:"
echo "  ${RESULTS_ROOT}"

RESULTS_ROOT_ENV="${RESULTS_ROOT}" python3 - <<'PY'
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["RESULTS_ROOT_ENV"])
summary_files = [
    "run_manifest.csv",
    "metrics_by_run.csv",
    "per_class_metrics.csv",
    "confusion_counts.csv",
    "group_summary.csv",
]

for file_name in summary_files:
    path = root / file_name
    size = path.stat().st_size if path.exists() else "missing"
    print(f"{file_name}: exists={path.exists()} size={size}")

atlas_count = len(list((root / "atlas_reference").glob("*.csv.gz")))
classified_count = len(list((root / "classified_rows").glob("*.csv.gz")))
figure_names = sorted(path.name for path in (root / "figures").glob("*.png"))

print(f"atlas refs: {atlas_count}")
print(f"classified rows: {classified_count}")
print(f"figures: {figure_names}")

group_summary_path = root / "group_summary.csv"
if group_summary_path.exists():
    print()
    print("group_summary.csv preview:")
    print(pd.read_csv(group_summary_path).head().to_string(index=False))
PY
