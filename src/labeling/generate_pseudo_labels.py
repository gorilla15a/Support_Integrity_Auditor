import pandas as pd
import numpy as np
from pathlib import Path
from src.fusion.severity_fusion import fuse_scores
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
from src.preprocess import preprocess

from src.signals.embedding_signal import generate_embeddings
from src.signals.clustering_signal import (
    cluster_embeddings,
    cluster_to_severity
)

from src.signals.rule_signal import get_rule_score
INPUT_FILE = (
    ROOT /
    "data/raw/customer_support_tickets.csv"
)

OUTPUT_FILE = (
    ROOT /
    "data/processed/pseudo_labels.csv"
)

SEVERITY_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Critical": 3
}

INV_MAP = {
    0: "Low",
    1: "Medium",
    2: "High",
    3: "Critical"
}


def resolution_to_severity(hours):

    if hours <= 12:
        return 3   # Critical

    elif hours <= 25:
        return 2   # High

    elif hours <= 45:
        return 1   # Medium

    return 0       # Low

def main():
    print("Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    df = preprocess(df)

    print("Generating embeddings...")

    embeddings = generate_embeddings(
        df["full_text"].tolist()
    )

    print("Clustering embeddings...")

    clusters = cluster_embeddings(
        embeddings,
        n_clusters=40
    )

    cluster_severity_map = (
    cluster_to_severity(
        clusters,
        df["Priority_Level"]
    )
    )

    cluster_scores = [
        cluster_severity_map[c]
        for c in clusters
    ]

    print("Computing signals...")

    rule_scores = []
    rule_evidence = []

    for text in df["full_text"]:
        score, evidence = get_rule_score(text)
        rule_scores.append(score)
        rule_evidence.append(
            ",".join(evidence)
        )

    resolution_scores = (
        df["Resolution_Time_Hours"]
        .apply(resolution_to_severity)
        .tolist()
    )

    print("Loading LLM scores...")

    llm_df = pd.read_csv(
    ROOT / "data/processed/llm_scores.csv"
    )

    df = df.merge(
    llm_df,
    on="Ticket_ID",
    how="left"
    )

    llm_scores = (
    df["llm_score"]
    .astype(int)
    .tolist()
    )

    fused_scores = []

    for llm, cluster, resolution, rule in zip(
        llm_scores,
        cluster_scores,
        resolution_scores,
        rule_scores
    ):
        fused = fuse_scores(
            llm,
            cluster,
            resolution,
            rule
        )
        fused_scores.append(fused)

    assigned = (
        df["Priority_Level"]
        .map(SEVERITY_MAP)
        .tolist()
    )

    severity_delta = []
    mismatch = []
    mismatch_type = []

    for gt, pred in zip(
        assigned,
        fused_scores
    ):
        delta = pred - gt
        severity_delta.append(delta)

        if delta == 0:
            mismatch.append(0)
            mismatch_type.append("Consistent")
        else:
            mismatch.append(1)
            if delta > 0:
                mismatch_type.append("Hidden Crisis")
            else:
                mismatch_type.append("False Alarm")

    df["llm_score"] = llm_scores
    df["cluster_score"] = cluster_scores
    df["resolution_score"] = resolution_scores
    df["rule_score"] = rule_scores
    df["rule_evidence"] = rule_evidence
    df["inferred_severity"] = [
        INV_MAP[x]
        for x in fused_scores
    ]
    df["severity_delta"] = severity_delta
    df["mismatch"] = mismatch
    df["mismatch_type"] = mismatch_type

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nPseudo labels saved:")
    print(OUTPUT_FILE)

    print("\nMismatch Distribution")
    print(
        df["mismatch"]
        .value_counts()
    )

    print("\nInferred Severity")
    print(
        df["inferred_severity"]
        .value_counts()
    )


if __name__ == "__main__":
    main()
