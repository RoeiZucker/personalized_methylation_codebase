from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["donor", "window_id", "genomic_position"]


def methylation_class(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Methylation values must lie between zero and one")
    return np.where(values <= .2, 0, np.where(values < .8, 1, 2)).astype(np.int8)


def reference_cpgs(frames):
    if len(frames) < 2:
        raise ValueError("At least two reference donors are needed")
    keys = ["chrom", "start", "end"]
    result = None
    for i, frame in enumerate(frames):
        current = frame[keys + ["methyl_rate"]].rename(columns={"methyl_rate": f"value_{i}"})
        if current.duplicated(keys).any():
            raise ValueError("Duplicate reference CpG interval")
        result = current if result is None else result.merge(current, on=keys, how="inner", validate="one_to_one")
    values = result.filter(regex="^value_").to_numpy(float)
    if not len(values):
        raise ValueError("Reference donors have no common CpG intervals")
    result["mean"] = values.mean(axis=1)
    result["sd"] = values.std(axis=1, ddof=1)
    return result[keys + ["mean", "sd"]]


def attach_reference(rows, cpgs, frozen_bins=None):
    rows = rows.copy()
    maximum = float(cpgs.sd.max())
    if not np.isfinite(maximum) or maximum < 0:
        raise ValueError("Invalid reference variability")
    rows["atlas_value"] = np.nan
    rows["variability"] = np.nan
    for chrom, ix in rows.groupby("chrom", sort=False).groups.items():
        ref = cpgs[cpgs.chrom.eq(chrom)].sort_values("start")
        starts = ref.start.to_numpy(np.int64)
        query = rows.loc[ix, "genomic_position"].to_numpy(np.int64)
        lo = np.searchsorted(starts, query - 1, side="left")
        hi = np.searchsorted(starts, query + 5, side="right")
        count = hi - lo
        for source, dest in [("mean", "atlas_value"), ("sd", "variability")]:
            prefix = np.r_[0., np.cumsum(ref[source].to_numpy(float))]
            rows.loc[ix, dest] = np.divide(prefix[hi] - prefix[lo], count, out=np.full(len(ix), np.nan), where=count > 0)
    valid = rows.atlas_value.notna()
    rows["atlas"] = pd.Series(pd.NA, index=rows.index, dtype="Int8")
    rows.loc[valid, "atlas"] = methylation_class(rows.loc[valid, "atlas_value"])
    if frozen_bins is not None:
        if frozen_bins.duplicated(KEYS).any() or rows.duplicated(KEYS).any():
            raise ValueError("Duplicate frozen-bin or prediction key")
        keep = KEYS + ["bin"] + (["label"] if "label" in frozen_bins else [])
        rows = rows.drop(columns="bin", errors="ignore").merge(frozen_bins[keep], on=KEYS, how="left", validate="one_to_one", suffixes=("", "_frozen"))
        if rows.bin.isna().any():
            raise ValueError("Frozen assignments do not cover every prediction row")
        if "label_frozen" in rows and not rows.label.eq(rows.label_frozen).all():
            raise ValueError("Frozen assignments and prediction truth differ")
        return rows.drop(columns="label_frozen", errors="ignore")
    sd = rows.variability.to_numpy(float)
    rows["bin"] = np.select([sd <= maximum / 5, sd <= 2 * maximum / 5, sd <= 3 * maximum / 5, np.isfinite(sd)], [1, 2, 3, 4], default=0)
    return rows


def local_class_average(rows, training, radius=300):
    if radius < 0:
        raise ValueError("Radius must be nonnegative")
    output = np.empty(len(rows), dtype=np.int8)
    rows = rows.reset_index(drop=True)
    for donor, drows in rows.groupby("donor", sort=False):
        train = training[training.donor.eq(donor)]
        if train.empty or not train.label.isin([0, 1, 2]).all():
            raise ValueError(f"Missing or invalid training labels for {donor}")
        if train.duplicated(["chrom", "genomic_position"]).any():
            raise ValueError("Duplicate training positions")
        if drows.merge(train, on=["donor", "chrom", "genomic_position"]).shape[0]:
            raise ValueError("Training labels overlap held-out prediction sites")
        counts = np.bincount(train.label.astype(int), minlength=3)
        fallback = int(np.flatnonzero(counts == counts.max())[0])
        for chrom, ix in drows.groupby("chrom", sort=False).groups.items():
            ref = train[train.chrom.eq(chrom)].sort_values("genomic_position")
            if ref.empty:
                output[ix] = fallback
                continue
            pos = ref.genomic_position.to_numpy(np.int64)
            labels = ref.label.to_numpy(np.int8)
            q = rows.loc[ix, "genomic_position"].to_numpy(np.int64)
            lo = np.searchsorted(pos, q - radius, side="left")
            hi = np.searchsorted(pos, q + radius, side="right")
            prefix = np.r_[0., np.cumsum(labels, dtype=float)]
            count = hi - lo
            means = np.divide(prefix[hi] - prefix[lo], count, out=np.full(len(ix), np.nan), where=count > 0)
            classes = np.where(means < .5, 0, np.where(means < 1.5, 1, 2))
            right = np.clip(np.searchsorted(pos, q), 0, len(pos) - 1)
            left = np.clip(np.searchsorted(pos, q) - 1, 0, len(pos) - 1)
            nearest = np.where(np.abs(pos[right] - q) < np.abs(q - pos[left]), right, left)
            classes[count == 0] = labels[nearest[count == 0]]
            output[ix] = classes
    return output
