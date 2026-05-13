#!/usr/bin/env python3

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


def compute_non_majority_stats(merged_df, file_paths):
    results = []

    current_block = []
    current_cluster = None

    for i, row in merged_df.iterrows():
        ref_cluster = row["_ref_cluster"]

        if pd.isna(ref_cluster):
            continue

        if current_cluster is None:
            current_cluster = ref_cluster

        if ref_cluster != current_cluster:
            results.append(process_block_stats(merged_df.loc[current_block], file_paths, current_cluster))
            current_block = []
            current_cluster = ref_cluster

        current_block.append(i)

    if current_block:
        results.append(process_block_stats(merged_df.loc[current_block], file_paths, current_cluster))

    return pd.DataFrame(results)


def process_block_stats(block_df, file_paths, ref_cluster):
    row = {"reference_cluster": ref_cluster}
    block_size = len(block_df)

    for fp in file_paths:
        vals = block_df[fp].dropna().astype(str)

        if len(vals) == 0:
            row[f"{fp}_count"] = None
            row[f"{fp}_frac"] = None
            continue

        counts = Counter(vals)
        max_count = max(counts.values())
        majority = {k for k, v in counts.items() if v == max_count}

        non_majority_count = sum(1 for v in vals if v not in majority)

        # fraction relative to block size (NOT just non-null values)
        frac = non_majority_count / block_size if block_size > 0 else None

        row[f"{fp}_count"] = non_majority_count
        row[f"{fp}_frac"] = frac

    row["block_size"] = block_size

    return row


def read_path_list(path_list_file):
    file_paths = []
    with open(path_list_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                file_paths.append(line)

    if not file_paths:
        raise ValueError("No file paths found.")

    return file_paths


def detect_delimiter(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline()

    return "\t" if "\t" in first_line else ","


def read_mapping_file(file_path):
    suffix = Path(file_path).suffix.lower()

    if suffix in [".csv", ".tsv", ".txt"]:
        sep = "\t" if suffix == ".tsv" else detect_delimiter(file_path)
        df = pd.read_csv(file_path, sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "nuclei" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{file_path} must have 'nuclei' and 'cluster' columns.")

    df = df[["nuclei", "cluster"]].copy()
    df["nuclei"] = df["nuclei"].astype(str)
    df["cluster"] = df["cluster"].astype(str)

    return df.drop_duplicates(subset="nuclei", keep="first")


def build_merged_dataframe(file_paths, reference_index):
    if reference_index < 0 or reference_index >= len(file_paths):
        raise ValueError("reference-index out of range")

    run_data = {fp: read_mapping_file(fp) for fp in file_paths}

    reference_path = file_paths[reference_index]
    ref_df = run_data[reference_path]

    merged = None
    for fp in file_paths:
        temp = run_data[fp].rename(columns={"cluster": fp})
        if merged is None:
            merged = temp
        else:
            merged = merged.merge(temp, on="nuclei", how="outer")

    # Map reference clusters
    ref_map = dict(zip(ref_df["nuclei"], ref_df["cluster"]))
    merged["_ref_cluster"] = merged["nuclei"].map(ref_map)

    # Preserve cluster block order from reference
    cluster_order = []
    seen = set()
    for c in ref_df["cluster"]:
        if c not in seen:
            cluster_order.append(c)
            seen.add(c)

    cluster_rank = {c: i for i, c in enumerate(cluster_order)}
    ref_nuclei_order = {n: i for i, n in enumerate(ref_df["nuclei"])}

    def sort_key(row):
        if pd.notna(row["_ref_cluster"]):
            return (
                0,
                cluster_rank[row["_ref_cluster"]],
                ref_nuclei_order.get(row["nuclei"], float("inf"))
            )
        else:
            return (1, float("inf"), row["nuclei"])

    merged["_sort"] = merged.apply(sort_key, axis=1)
    merged = merged.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)

    return merged, reference_path


def compute_majority_table(merged_df, file_paths):
    """
    Optional: returns a dataframe showing majority cluster per block per run
    """
    results = []

    current_block = []
    current_cluster = None

    for i, row in merged_df.iterrows():
        ref_cluster = row["_ref_cluster"]

        if pd.isna(ref_cluster):
            continue

        if current_cluster is None:
            current_cluster = ref_cluster

        if ref_cluster != current_cluster:
            results.append(process_block(merged_df.loc[current_block], file_paths, current_cluster))
            current_block = []
            current_cluster = ref_cluster

        current_block.append(i)

    if current_block:
        results.append(process_block(merged_df.loc[current_block], file_paths, current_cluster))

    return pd.DataFrame(results)


def process_block(block_df, file_paths, ref_cluster):
    row = {"reference_cluster": ref_cluster}

    for fp in file_paths:
        vals = block_df[fp].dropna().astype(str)
        if len(vals) == 0:
            row[fp] = None
            continue

        counts = Counter(vals)
        max_count = max(counts.values())
        majority = [k for k, v in counts.items() if v == max_count]

        row[fp] = "|".join(majority)

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path-list", required=True)
    parser.add_argument("--reference-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    #parser.add_argument("--output-majority", default=None)
    parser.add_argument("--output-nonmajority", default=None)

    args = parser.parse_args()

    file_paths = read_path_list(args.path_list)

    merged_df, reference_path = build_merged_dataframe(
        file_paths,
        args.reference_index
    )

    

    # Save main merged file
    merged_df.drop(columns=["_ref_cluster"]).to_csv(args.output, index=False)

    print(f"Saved merged CSV → {args.output}")

    if args.output_nonmajority:
        nm_df = compute_non_majority_stats(merged_df, file_paths)
        nm_df.to_csv(args.output_nonmajority, index=False)
        print(f"Saved non-majority stats → {args.output_nonmajority}")

    print(f"Reference file index: {args.reference_index}")
    print(f"Reference file path: {reference_path}")


if __name__ == "__main__":
    main()