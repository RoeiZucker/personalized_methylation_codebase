from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import KEYS
from .data import read_table, write_table
from .statistics import confusion, f1_from_confusion, random_f1, bootstrap, sign_test, bh_adjust

MODELS = ["atlas", "donor_excluded", "personalized", "local_class_average", "random"]


def join_predictions(files):
    result = None
    for model, path in files.items():
        frame = read_table(path)
        if frame.duplicated(KEYS).any():
            raise ValueError(f"Duplicate prediction keys: {model}")
        frame = frame.rename(columns={"predicted_class":model})
        if result is None:
            result = frame
            continue
        extra = frame[KEYS + ["label", model]]
        result = result.merge(extra, on=KEYS, how="outer", validate="one_to_one", indicator=True, suffixes=("", "_other"))
        if not result._merge.eq("both").all() or not result.label.eq(result.label_other).all():
            raise ValueError(f"Prediction keys or true labels differ: {model}")
        result = result.drop(columns=["_merge", "label_other"])
    return result


def subsets(rows):
    yield "all", np.ones(len(rows), dtype=bool), False
    if "bimodal_support" in rows:
        yield "inside", rows.bimodal_support.gt(0).to_numpy(), False
        yield "outside", rows.bimodal_support.eq(0).to_numpy(), False
        for label, mask in [("support_0", rows.bimodal_support.eq(0)), ("support_1", rows.bimodal_support.eq(1)), ("support_2plus", rows.bimodal_support.ge(2))]:
            yield label, mask.to_numpy(), False
    for col in rows.filter(regex="^annotation_"):
        if rows[col].dtype != bool and not rows[col].isin([0, 1]).all():
            raise ValueError(f"Annotation must contain booleans or 0/1: {col}")
        yield col[11:], rows[col].to_numpy(bool), True


def donor_scores(rows):
    required = {"donor", "tissue", "label", "bin", "donor_excluded", "personalized"}
    if not required.issubset(rows.columns):
        raise ValueError(f"Missing row columns: {sorted(required - set(rows.columns))}")
    if rows.duplicated(KEYS).any():
        raise ValueError("Duplicate evaluated donor-site keys")
    models = [m for m in MODELS if m in rows and m != "random"]
    rows = rows[rows.bin.isin([1, 2, 3, 4])].copy()
    # Every reported approach uses the same available sites.
    rows = rows.dropna(subset=models)
    records = []
    for (donor, tissue, bin_id), group in rows.groupby(["donor", "tissue", "bin"], sort=True):
        for subset, mask, minimum in subsets(group):
            chosen = group.iloc[np.flatnonzero(mask)]
            if len(chosen) < (50 if minimum else 1):
                continue
            truth = chosen.label.to_numpy(int)
            for model in models + ["random"]:
                weighted, classes = random_f1(truth) if model == "random" else f1_from_confusion(confusion(truth, chosen[model].to_numpy()))
                for metric, value in [("weighted_f1", weighted), *[(f"class_{i}_f1", classes[i]) for i in range(3)]]:
                    records.append({"donor":donor, "tissue":tissue, "bin":int(bin_id), "subset":subset,
                                    "model":model, "metric":metric, "value":value, "sites":len(chosen)})
    return pd.DataFrame(records)


def summarize(scores, seed=20261014):
    keys = ["bin", "subset", "model", "metric"]
    rows = []
    for key, group in scores.groupby(keys, sort=True):
        group = group.sort_values("donor")
        rows.append({**dict(zip(keys, key)), **bootstrap(group.value, seed), "sites":int(group.sites.sum())})
    return pd.DataFrame(rows)


def paired(scores, model="donor_excluded", seed=20261014):
    keys = ["donor", "tissue", "bin", "subset", "metric"]
    a = scores[scores.model.eq("personalized")][keys + ["value"]]
    b = scores[scores.model.eq(model)][keys + ["value"]]
    aligned = a.merge(b, on=keys, suffixes=("_personalized", "_reference"), validate="one_to_one")
    aligned["gain"] = aligned.value_personalized - aligned.value_reference
    result = []
    for key, group in aligned.groupby(["bin", "subset", "metric"], sort=True):
        result.append({"bin":key[0], "subset":key[1], "metric":key[2], "reference":model,
                       **bootstrap(group.sort_values("donor").gain, seed), **sign_test(group.gain)})
    return aligned, pd.DataFrame(result)


def contrasts(gains, groups, seed=20261104):
    keys = ["donor", "bin", "metric"]
    result = []
    for higher, lower in groups:
        a = gains[gains.subset.eq(higher)][keys + ["gain"]]
        b = gains[gains.subset.eq(lower)][keys + ["gain"]]
        pair = a.merge(b, on=keys, suffixes=("_higher", "_lower"), validate="one_to_one")
        for (bin_id, metric), group in pair.groupby(["bin", "metric"], sort=True):
            difference = group.sort_values("donor").gain_higher - group.sort_values("donor").gain_lower
            result.append({"bin":bin_id, "higher":higher, "lower":lower, "metric":metric,
                           **bootstrap(difference, seed), **sign_test(difference)})
    frame = pd.DataFrame(result)
    if not frame.empty:
        frame["q_value"] = frame.groupby("metric").p_value.transform(bh_adjust)
    return frame


def run_report(rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    scores = donor_scores(rows)
    write_table(scores, output / "donor_scores.csv")
    write_table(summarize(scores), output / "performance.csv")
    gains, tests = paired(scores)
    write_table(gains, output / "donor_gains.csv")
    regulatory = set(rows.filter(regex="^annotation_").columns.str[11:]) | {"all"}
    family = tests.bin.isin([3, 4]) & tests.subset.isin(regulatory) & tests.metric.eq("weighted_f1")
    tests["q_value"] = np.nan
    tests.loc[family, "q_value"] = bh_adjust(tests.loc[family, "p_value"])
    write_table(tests, output / "personalization_tests.csv")
    for tissue, part in scores.groupby("tissue"):
        write_table(summarize(part), output / f"tissue_{tissue}.csv")
    if "local_class_average" in scores.model.values:
        _, local = paired(scores, "local_class_average")
        local["family"] = "other"
        local.loc[local.subset.eq("all"), "family"] = "overall"
        local.loc[local.subset.isin(["inside", "outside"]), "family"] = "bimodality"
        local.loc[local.subset.isin(regulatory - {"all"}), "family"] = "regulatory"
        local = local[local.bin.isin([3, 4])]
        local["q_value"] = local.groupby(["family", "metric"]).p_value.transform(bh_adjust)
        write_table(local, output / "local_class_tests.csv")
    if "bimodal_support" in rows:
        high = gains[gains.bin.isin([3, 4])]
        write_table(contrasts(high, [("inside", "outside")], seed=20261014), output / "inside_outside.csv")
        write_table(contrasts(high, [("support_1", "support_0"), ("support_2plus", "support_1"), ("support_2plus", "support_0")]), output / "support_contrasts.csv")
    return scores


def donor_matching(rows):
    required = {"donor", "model_donor", "stage", "bin", "window_id", "genomic_position", "label", "predicted_class"}
    if not required.issubset(rows):
        raise ValueError(f"Missing matching columns: {sorted(required - set(rows.columns))}")
    records = []
    for (donor, stage, bin_id), group in rows.groupby(["donor", "stage", "bin"], sort=True):
        keys = ["window_id", "genomic_position"]
        if group.duplicated(keys + ["model_donor"]).any():
            raise ValueError("Duplicate donor-matching rows")
        truth = group.pivot(index=keys, columns="model_donor", values="label")
        if not truth.nunique(axis=1).eq(1).all():
            raise ValueError("Mismatched true labels across models")
        prediction = group.pivot(index=keys, columns="model_donor", values="predicted_class").dropna()
        if donor not in prediction or prediction.shape[1] < 2 or prediction.empty:
            raise ValueError("Matching requires a matched model and at least one other model on shared sites")
        actual = truth.loc[prediction.index].iloc[:, 0]
        f1s = {m:f1_from_confusion(confusion(actual, prediction[m]))[0] for m in prediction}
        other = [v for m, v in f1s.items() if m != donor]
        records.append({"donor":donor, "stage":stage, "bin":bin_id, "matched":f1s[donor],
                        "mismatched_mean":np.mean(other), "gain":f1s[donor]-np.mean(other), "sites":len(prediction)})
    donors = pd.DataFrame(records)
    summary = []
    for (stage, bin_id), group in donors.groupby(["stage", "bin"], sort=True):
        summary.append({"stage":stage, "bin":bin_id, **bootstrap(group.sort_values("donor").gain, 20261102), **sign_test(group.gain)})
    summary = pd.DataFrame(summary)
    summary["q_value"] = bh_adjust(summary.p_value)
    return donors, summary


def chromosome_exclusions(scores):
    required = {"donor", "tissue", "chrom", "bin", "gain"}
    if not required.issubset(scores):
        raise ValueError("Chromosome scores need donor,tissue,chrom,bin,gain")
    if scores.duplicated(["donor", "chrom", "bin"]).any():
        raise ValueError("Duplicate donor-chromosome scores")
    rows = []
    exclusions = [("none", "none")] + [("chrom", x) for x in sorted(scores.chrom.unique())] + [("tissue", x) for x in sorted(scores.tissue.unique())]
    for kind, omitted in exclusions:
        retained = scores if kind == "none" else scores[scores[kind].ne(omitted)]
        for bin_id, group in retained.groupby("bin", sort=True):
            donor_means = group.groupby("donor", sort=True).gain.mean()
            rows.append({"bin":bin_id, "exclusion":kind, "omitted":omitted,
                         **bootstrap(donor_means), **sign_test(donor_means)})
    return pd.DataFrame(rows)
