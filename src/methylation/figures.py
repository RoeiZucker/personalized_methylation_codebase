from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .statistics import bootstrap, sign_test, bh_adjust
from .analysis import chromosome_exclusions

COLORS = {"atlas":"#29977A", "donor_excluded":"#2878AD", "personalized":"#E57A19", "local_class_average":"#9E2A6B", "random":"#A0A6AD"}
NAMES = {"atlas":"Atlas", "donor_excluded":"Donor-excluded model", "personalized":"Personalized model", "local_class_average":"Local class-average approach", "random":"Class-frequency random predictor"}
ALIASES = {"mean_atlas":"atlas", "pretrained_e10":"donor_excluded", "donor_excluded_e10":"donor_excluded", "personalized_e3":"personalized", "class_frequency_random":"random"}
MODELS = list(COLORS)
PURPLE = "#7953A3"


def tidy_wide(frame, group):
    rows = []
    for item in frame.to_dict("records"):
        for key, value in item.items():
            if key.endswith("_weighted_f1") and key[:-12] in ALIASES | {"local_class_average":"local_class_average"}:
                model = ALIASES.get(key[:-12], key[:-12])
                rows.append({"donor":item["person"], "tissue":item["tissue"], "bin":int(item["endpoint"][-1]),
                             "group":item[group], "model":model, "value":value, "sites":item["n_rows"]})
    return pd.DataFrame(rows)


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def bars(ax, frame, groups):
    x = np.arange(len(groups))
    for k, model in enumerate(MODELS):
        stats = []
        for group in groups:
            part = frame[frame.group.eq(group) & frame.model.eq(model)].sort_values("donor")
            stats.append(bootstrap(part.value))
        means = np.array([s["mean"] for s in stats])
        error = np.array([[s["mean"] - s["ci_low"] for s in stats], [s["ci_high"] - s["mean"] for s in stats]])
        ax.bar(x + (k-2)*.15, means, width=.145, color=COLORS[model], label=NAMES[model])
        ax.errorbar(x + (k-2)*.15, means, yerr=error, fmt="none", capsize=2, color="#263238", linewidth=.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Weighted F1")
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)


def gains(ax, frame, groups):
    for j, group in enumerate(groups):
        part = frame[frame.group.eq(group)].pivot(index="donor", columns="model", values="value").sort_index()
        delta = (part.personalized - part.donor_excluded).to_numpy()
        stat = bootstrap(delta)
        if len(delta) > 1 and np.ptp(delta) > 0:
            violin = ax.violinplot(delta, positions=[j], widths=.65, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(PURPLE); body.set_alpha(.2)
        ax.scatter(j + np.random.default_rng(42).uniform(-.1, .1, len(delta)), delta, color=PURPLE, s=14)
        ax.errorbar(j, stat["mean"], yerr=[[stat["mean"]-stat["ci_low"]], [stat["ci_high"]-stat["mean"]]], fmt="D", color="#263238", mfc="white", capsize=3)
    ax.axhline(0, color="#263238", linewidth=.7)
    ax.set_ylabel("Personalized minus\ndonor-excluded F1")
    ax.set_xticks(range(len(groups)), groups)


def two_bins(frame, output, name, groups):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex="col")
    for col, bin_id in enumerate([3, 4]):
        part = frame[frame.bin.eq(bin_id)]
        bars(axes[0, col], part, groups)
        gains(axes[1, col], part, groups)
        axes[0, col].set_title(f"Bin {bin_id}")
        axes[1, col].tick_params(axis="x", rotation=90 if len(groups) > 3 else 0)
    fig.legend(handles=[Patch(color=COLORS[m], label=NAMES[m]) for m in MODELS], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .85))
    save(fig, output, name)


def reproduce(data, output):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False, "pdf.fonttype":42, "svg.fonttype":"none"})
    all_bins = pd.read_csv(data / "all_bins_donor_scores.csv").rename(columns={"person":"donor", "manuscript_bin":"bin", "weighted_f1":"value", "prediction_rows":"sites"})
    all_bins["model"] = all_bins.model.replace(ALIASES)
    all_bins["group"] = all_bins.bin.map(lambda x:f"Bin {x}")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars(ax, all_bins, [f"Bin {x}" for x in range(1, 5)])
    ax.set_xticks(range(4), [f"Bin {x}" for x in range(1, 5)])
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.38))
    fig.tight_layout(); save(fig, output, "figure2")
    tissue = tidy_wide(pd.read_csv(data / "tissue_donor_scores.csv"), "tissue")
    two_bins(tissue, output, "figure3", list(tissue.group.unique()))
    bimodal = tidy_wide(pd.read_csv(data / "bimodality_donor_scores.csv"), "subset")
    groups = ["all", "outside_bimodal", "inside_bimodal"]
    two_bins(bimodal, output, "figure5", groups)
    regulatory = tidy_wide(pd.read_csv(data / "regulatory_donor_scores.csv"), "annotation")
    annotation_order = ["all", "cpg_island", "cpg_shore", "cpg_shelf", "open_sea", "promoter", "gene_body", "regulatory_union", "ccre_pls", "ccre_pels", "ccre_dels"]
    tests = []
    for (bin_id, group), part in regulatory.groupby(["bin", "group"], sort=True):
        pair = part.pivot(index="donor", columns="model", values="value").sort_index()
        delta = pair.personalized - pair.donor_excluded
        tests.append({"bin":bin_id, "group":group, **bootstrap(delta), **sign_test(delta)})
    tests = pd.DataFrame(tests)
    tests["q_value"] = bh_adjust(tests.p_value)
    tests.to_csv(output / "regulatory_tests.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3), sharey=True)
    for col, bin_id in enumerate([3, 4]):
        for j, group in enumerate(annotation_order):
            part = regulatory[regulatory.bin.eq(bin_id) & regulatory.group.eq(group)]
            for k, model in enumerate(MODELS):
                value = part.loc[part.model.eq(model), "value"].mean()
                axes[col].scatter(value, j + (k-2)*.13, color=COLORS[model], s=24)
            t = tests[tests.bin.eq(bin_id) & tests.group.eq(group)].iloc[0]
            star = "**" if t["mean"] > 0 and t.q_value < .05 else "*" if t["mean"] > 0 and t.p_value < .05 else ""
            axes[col].text(.83, j, f'{t.positive}/{t.donors}{star}', va="center", fontsize=10)
        axes[col].set_xlim(.3, 1.02); axes[col].set_ylim(len(annotation_order)-.5, -.5)
        axes[col].set_title(f"Bin {bin_id}"); axes[col].set_xlabel("Weighted F1")
        axes[col].set_yticks(range(len(annotation_order)), annotation_order)
        axes[col].grid(axis="x", alpha=.15)
    fig.legend(handles=[Patch(color=COLORS[m], label=NAMES[m]) for m in MODELS], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .83)); save(fig, output, "figure4")
    summary, donor_gains = [], []
    for (bin_id, group), part in bimodal.groupby(["bin", "group"], sort=True):
        pair = part.pivot(index="donor", columns="model", values="value").sort_index()
        delta = pair.personalized - pair.donor_excluded
        summary.append({"bin":bin_id, "subset":group, "donor_excluded":pair.donor_excluded.mean(), "personalized":pair.personalized.mean(), **bootstrap(delta), **sign_test(delta)})
        donor_gains.append(pd.DataFrame({"bin":bin_id, "subset":group, "donor":pair.index, "gain":delta.to_numpy()}))
    summary = pd.DataFrame(summary)
    summary["q_value"] = np.nan
    mask = summary.subset.ne("all")
    summary.loc[mask, "q_value"] = bh_adjust(summary.loc[mask, "p_value"])
    # Whole-bin tests retain all four bins in their correction family.
    whole = []
    for bin_id, part in all_bins.groupby("bin"):
        pair = part.pivot(index="donor", columns="model", values="value")
        whole.append({"bin":bin_id, **sign_test(pair.personalized - pair.donor_excluded)})
    whole = pd.DataFrame(whole); whole["q_value"] = bh_adjust(whole.p_value)
    for row in summary[~mask].itertuples():
        summary.loc[row.Index, "q_value"] = whole.loc[whole.bin.eq(row.bin), "q_value"].iloc[0]
    summary.to_csv(output / "tableS2_performance.csv", index=False)
    donor_gains = pd.concat(donor_gains)
    direct = []
    for bin_id, part in donor_gains.groupby("bin"):
        pair = part.pivot(index="donor", columns="subset", values="gain").sort_index()
        delta = pair.inside_bimodal - pair.outside_bimodal
        direct.append({"bin":bin_id, **bootstrap(delta), **sign_test(delta)})
    direct = pd.DataFrame(direct); direct["q_value"] = bh_adjust(direct.p_value)
    direct.to_csv(output / "tableS2_contrasts.csv", index=False)
    support = pd.read_csv(data / "support_donor_contrasts.csv")
    contrasts = []
    for key, part in support.groupby(["bin", "higher_support", "lower_support"], sort=True):
        contrasts.append({"bin":key[0], "higher":key[1], "lower":key[2], **bootstrap(part.sort_values("donor").paired_gain_difference, 20261104), **sign_test(part.paired_gain_difference)})
    contrasts = pd.DataFrame(contrasts); contrasts["q_value"] = bh_adjust(contrasts.p_value)
    contrasts.to_csv(output / "table1.csv", index=False)
    chromosome_exclusions(pd.read_csv(data / "chromosome_donor_scores.csv")).to_csv(output / "tableS1.csv", index=False)
    classes = pd.read_csv(data / "class_donor_scores.csv")
    classes["model"] = classes.model.replace(ALIASES)
    for bin_id in [3, 4]:
        part = classes[classes.endpoint.eq(f"bin_{bin_id}")]
        category = part.class_name.unique().tolist()
        long = part.rename(columns={"person":"donor", "class_name":"group", "f1":"value"})
        fig, axes = plt.subplots(2, 1, figsize=(9, 7))
        bars(axes[0], long, category); axes[0].set_ylabel("F1")
        for j, group in enumerate(category):
            pair = long[long.group.eq(group)].pivot(index="donor", columns="model", values="value").sort_index()
            for k, model in enumerate(["atlas", "donor_excluded", "local_class_average", "random"]):
                delta = pair.personalized - pair[model]
                color = PURPLE if model == "donor_excluded" else COLORS[model]
                stat = bootstrap(delta); x = j + (k-1.5)*.18
                axes[1].scatter(x + np.random.default_rng(42).uniform(-.04,.04,len(delta)), delta, color=color, s=10)
                axes[1].errorbar(x, stat["mean"], yerr=[[stat["mean"]-stat["ci_low"]],[stat["ci_high"]-stat["mean"]]], fmt="D", color="#263238", mfc="white", capsize=2)
        axes[1].axhline(0,color="#263238",linewidth=.7); axes[1].set_ylabel("Personalized minus reference F1")
        axes[1].set_xticks(range(len(category)),category); axes[0].set_xticks(range(len(category)),category)
        fig.legend(handles=[Patch(color=COLORS[m],label=NAMES[m]) for m in MODELS],loc="upper center",ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.87)); save(fig,output,f"figureS6_bin{bin_id}")
