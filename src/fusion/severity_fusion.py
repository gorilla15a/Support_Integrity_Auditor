def fuse_scores(
    llm,
    cluster,
    resolution,
    rule
):

    score = (
        0.30 * llm +
        0.30 * cluster +
        0.25 * resolution +
        0.15 * rule
    )

    score = round(score)

    score = max(
        0,
        min(3, score)
    )

    return score