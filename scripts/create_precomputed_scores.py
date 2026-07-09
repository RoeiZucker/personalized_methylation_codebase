#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import ast
import pickle


DEFAULT_TYPES = ["mean_label", "all_two"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-type",
        action="append",
        dest="score_types",
        default=None,
        help="Score type to extract. Can be passed multiple times. Defaults to mean_label and all_two.",
    )
    parser.add_argument(
        "--result-index",
        type=int,
        default=0,
        help="Which result file per sample to use as the source of fixed scores.",
    )
    return parser.parse_args()


def extract_scores(eval_object, score_types):
    scores = {}
    for score_type in score_types:
        matrix_key = f"{score_type}_confusion_matrix"
        if score_type in eval_object:
            scores[score_type] = eval_object[score_type]
        if matrix_key in eval_object:
            scores[matrix_key] = eval_object[matrix_key]
    return scores


def main():
    args = parse_args()
    score_types = args.score_types or DEFAULT_TYPES
    data = ast.literal_eval(args.input_file.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample_name, sample_results in sorted(data.items()):
        result_items = list(sample_results.items())
        if not result_items:
            continue
        result_path, result_by_section = result_items[args.result_index]
        if "all_results" not in result_by_section:
            raise ValueError(f"{sample_name} result {result_path} does not contain all_results")
        scores = extract_scores(result_by_section["all_results"], score_types)
        if not scores:
            raise ValueError(f"No requested score types found for {sample_name}: {score_types}")
        output_path = args.output_dir / f"{sample_name}_precomputed_scores.pkl"
        with output_path.open("wb") as handle:
            pickle.dump(
                {
                    "sample_name": sample_name,
                    "source_result_path": result_path,
                    "score_types": score_types,
                    "scores": scores,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
