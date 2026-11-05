from __future__ import annotations

import argparse
from pathlib import Path

from .data import MODEL, read_table, write_table


def assignments(values):
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if not separator or not key or not path or key in result:
            raise ValueError("Use unique name=path arguments")
        result[key] = path
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Personalized methylation prediction")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Build reference, target-donor or unseen-chromosome datasets")
    p.add_argument("--track", required=True); p.add_argument("--fasta", required=True)
    p.add_argument("--donor", required=True); p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["reference", "personalized", "chromosomes"], default="personalized")
    p.add_argument("--chromosomes", nargs="+"); p.add_argument("--tokenizer", default=MODEL)
    p = sub.add_parser("train", help="Train a donor-excluded or personalized model")
    p.add_argument("--datasets", nargs="+", required=True); p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=["donor-excluded", "personalized"], required=True)
    p.add_argument("--donor", required=True); p.add_argument("--model")
    p.add_argument("--batch-size", type=int, default=14); p.add_argument("--epochs", type=int)
    p.add_argument("--resume"); p.add_argument("--workers", type=int, default=0)
    p = sub.add_parser("predict", help="Write token predictions from a saved model")
    p.add_argument("--dataset", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--donor", required=True); p.add_argument("--tissue", required=True)
    p.add_argument("--output", required=True); p.add_argument("--batch-size", type=int, default=14)
    p = sub.add_parser("training-sites", help="Export available TRAIN class labels for local prediction")
    p.add_argument("--dataset", required=True); p.add_argument("--donor", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("join", help="Align predictions by donor, window and position")
    p.add_argument("--model", action="append", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("reference", help="Build a reference-donor CpG mean and standard deviation")
    p.add_argument("--tracks", nargs="+", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("attach-reference", help="Attach atlas predictions and variability bins")
    p.add_argument("--rows", required=True); p.add_argument("--reference", required=True)
    p.add_argument("--frozen-bins"); p.add_argument("--output", required=True)
    p = sub.add_parser("local", help="Predict from nearby TRAIN class labels")
    p.add_argument("--rows", required=True); p.add_argument("--training", required=True)
    p.add_argument("--radius", type=int, default=300); p.add_argument("--output", required=True)
    p = sub.add_parser("annotate", help="Apply BED annotations and CpG contexts")
    p.add_argument("--rows", required=True); p.add_argument("--bed", action="append", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("bimodal", help="Count same-tissue reference-donor bimodal support")
    p.add_argument("--rows", required=True); p.add_argument("--regions", required=True); p.add_argument("--cohort", required=True)
    p.add_argument("--chain"); p.add_argument("--output", required=True)
    p = sub.add_parser("report", help="Score annotated predictions and calculate donor-level tests")
    p.add_argument("--rows", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("matching", help="Compare matched and mismatched models on shared sites")
    p.add_argument("--rows", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("chromosome-checks", help="Recompute chromosome and cell-type omission summaries")
    p.add_argument("--scores", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("figures", help="Reproduce figures and tables from the included donor scores")
    p.add_argument("--data", default="data"); p.add_argument("--output", default="results")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            from .data import prepare
            prepare(args.track, args.fasta, args.output, args.donor, args.mode, args.chromosomes, args.tokenizer)
        elif args.command == "train":
            from .model import train
            train(args.datasets, args.output, args.stage, args.donor, args.model, args.batch_size, args.epochs, args.resume, args.workers)
        elif args.command == "predict":
            from .model import predict
            predict(args.dataset, args.checkpoint, args.donor, args.tissue, args.output, args.batch_size)
        elif args.command == "training-sites":
            from .data import export_training_sites
            export_training_sites(args.dataset, args.donor, args.output)
        elif args.command == "join":
            from .analysis import join_predictions
            write_table(join_predictions(assignments(args.model)), args.output)
        elif args.command == "reference":
            from .baselines import reference_cpgs
            from .data import read_track
            write_table(reference_cpgs([read_track(p) for p in args.tracks]), args.output)
        elif args.command == "attach-reference":
            from .baselines import attach_reference
            write_table(attach_reference(read_table(args.rows), read_table(args.reference), read_table(args.frozen_bins) if args.frozen_bins else None), args.output)
        elif args.command == "local":
            from .baselines import local_class_average
            rows = read_table(args.rows)
            rows["local_class_average"] = local_class_average(rows, read_table(args.training), args.radius)
            write_table(rows, args.output)
        elif args.command == "annotate":
            from .annotations import annotate
            write_table(annotate(read_table(args.rows), assignments(args.bed)), args.output)
        elif args.command == "bimodal":
            from .annotations import map_to_hg19, bimodal_support
            rows = read_table(args.rows)
            if args.chain:
                rows = map_to_hg19(rows, args.chain)
            write_table(bimodal_support(rows, read_table(args.regions), read_table(args.cohort)), args.output)
        elif args.command == "report":
            from .analysis import run_report
            run_report(read_table(args.rows), args.output)
        elif args.command == "matching":
            from .analysis import donor_matching
            donors, summary = donor_matching(read_table(args.rows))
            write_table(donors, Path(args.output) / "matching_donors.csv")
            write_table(summary, Path(args.output) / "matching_summary.csv")
        elif args.command == "chromosome-checks":
            from .analysis import chromosome_exclusions
            write_table(chromosome_exclusions(read_table(args.scores)), args.output)
        elif args.command == "figures":
            from .figures import reproduce
            reproduce(Path(args.data), Path(args.output))
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
