from __future__ import annotations

import numpy as np
import pandas as pd


def overlaps(sites, intervals):
    result = pd.Series(False, index=sites.index)
    for chrom, ix in sites.groupby("chrom", sort=False).groups.items():
        ref = intervals[intervals.chrom.eq(chrom)].sort_values("start")
        if ref.empty:
            continue
        start = ref.start.to_numpy(np.int64)
        end = np.maximum.accumulate(ref.end.to_numpy(np.int64))
        q = sites.loc[ix, "genomic_position"].to_numpy(np.int64)
        site_end = sites.loc[ix, "site_end"].to_numpy(np.int64) if "site_end" in sites else q + 6
        right = np.searchsorted(start, site_end, side="left") - 1
        result.loc[ix] = (right >= 0) & (end[np.maximum(right, 0)] > q)
    return result.to_numpy()


def read_bed(path):
    frame = pd.read_csv(path, sep="\t", comment="#", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"])
    if frame.end.le(frame.start).any() or frame.start.lt(0).any():
        raise ValueError("BED coordinates must be zero-based half-open intervals")
    return frame


def annotate(rows, beds):
    rows = rows.copy()
    for name, path in beds.items():
        rows["annotation_" + name] = overlaps(rows, read_bed(path))
    if "cpg_island" in beds:
        islands = read_bed(beds["cpg_island"])
        shore, shelf = islands.copy(), islands.copy()
        shore["start"] = (shore.start - 2000).clip(lower=0)
        shore["end"] += 2000
        shelf["start"] = (shelf.start - 4000).clip(lower=0)
        shelf["end"] += 4000
        inside = rows.annotation_cpg_island.to_numpy(bool)
        near, wider = overlaps(rows, shore), overlaps(rows, shelf)
        rows["annotation_cpg_shore"] = near & ~inside
        rows["annotation_cpg_shelf"] = wider & ~near & ~inside
        rows["annotation_open_sea"] = ~wider & ~inside
    return rows


def map_to_hg19(rows, chain):
    from pyliftover import LiftOver
    lift = LiftOver(str(chain))
    output = rows.copy()
    chroms, starts, ends, success = [], [], [], []
    for row in rows.itertuples():
        a = lift.convert_coordinate(row.chrom, int(row.genomic_position))
        b = lift.convert_coordinate(row.chrom, int(row.genomic_position) + 5)
        valid = len(a or []) == len(b or []) == 1
        if valid:
            x, y = a[0], b[0]
            valid = x[0] == y[0] and x[2] == y[2] and abs(x[1] - y[1]) + 1 <= 20
        success.append(valid)
        chroms.append(x[0] if valid else row.chrom)
        starts.append(min(x[1], y[1]) if valid else -1)
        ends.append(max(x[1], y[1]) + 1 if valid else -1)
    output["hg19_chrom"], output["hg19_start"], output["hg19_end"], output["mapping_ok"] = chroms, starts, ends, success
    return output


def bimodal_support(rows, regions, cohort):
    required = {"donor", "tissue", "chrom", "start", "end"}
    if not required.issubset(regions.columns):
        raise ValueError("Bimodal catalog needs donor,tissue,chrom,start,end")
    if regions.end.le(regions.start).any():
        raise ValueError("Invalid bimodal intervals")
    rows = rows.copy()
    rows["bimodal_support"] = 0
    for donor, ix in rows.groupby("donor", sort=False).groups.items():
        meta = cohort[cohort.donor.eq(donor)]
        if len(meta) != 1:
            raise ValueError(f"Missing or duplicate cohort donor: {donor}")
        tissue = meta.iloc[0].tissue
        reference = set(cohort.loc[cohort.tissue.eq(tissue), "donor"]) - {donor}
        sites = rows.loc[ix].copy()
        if "hg19_start" in sites:
            sites["chrom"], sites["genomic_position"] = sites.hg19_chrom, sites.hg19_start
            sites["site_end"] = sites.hg19_end
        support = np.zeros(len(sites), dtype=int)
        for other in sorted(reference):
            support += overlaps(sites, regions[regions.donor.eq(other) & regions.tissue.eq(tissue)])
        if "mapping_ok" in sites:
            support[~sites.mapping_ok.to_numpy(bool)] = 0
        rows.loc[ix, "bimodal_support"] = support
    return rows
