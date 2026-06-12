from sklearn import cluster
from sklearn.cluster import KMeans
import pandas as pd

def cluster_embeddings(embeddings,n_clusters=40):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42
    )

    labels = model.fit_predict(
        embeddings
    )

    return labels

def cluster_to_severity(
    clusters,
    priority_levels
):

    temp = pd.DataFrame({
        "cluster": clusters,
        "priority": priority_levels
    })

    severity_map = {}

    sev_map = {
        "Low":0,
        "Medium":1,
        "High":2,
        "Critical":3
    }

    temp["priority_num"] = (
    temp["priority"]
    .astype(str)
    .str.strip()
    .str.title()
    .map(sev_map)
    )

    for cluster in temp.cluster.unique():

        cluster_values = (
            temp[
                temp.cluster == cluster
            ]
            ["priority_num"]
            .dropna()
        )

        if len(cluster_values) == 0:
            severity_map[cluster] = 1
            continue

        median_severity = (
            cluster_values.median()
        )

        severity_map[cluster] = int(
            round(median_severity)
        )

    return severity_map

    severity_map[cluster] = int(
    round(median_severity)
    )