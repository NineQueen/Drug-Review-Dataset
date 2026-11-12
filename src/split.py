from collections.abc import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

_BUCKETS = 10000
# A part this far from its requested share means the hash buckets overlap a previous cut.
_RELATIVE_TOLERANCE = 0.5


def _bucket_edges(ratios: Sequence[int]) -> list[int]:
    """Turn positive integer ratios into hash-bucket boundaries on `[0, 10000]`."""
    if (
        len(ratios) < 2
        or any(type(ratio) is not int or ratio <= 0 for ratio in ratios)
    ):
        raise ValueError("ratios must contain at least two positive integers")
    total = sum(ratios)
    edges = [0]
    assigned = 0
    for index, ratio in enumerate(ratios):
        width = _BUCKETS - assigned if index == len(ratios) - 1 else ratio * _BUCKETS // total
        assigned += width
        edges.append(assigned)
    return edges


def split(
    df: DataFrame,
    seed: int,
    ratios: Sequence[int] = (7, 1, 2),
) -> tuple[DataFrame, ...]:
    """Hash-split one cached frame per ratio.

    The length of `ratios` is the number of parts. `(8, 2)` is an 80/20 split and
    `(7, 1, 2)` is the default train/validation/test split. Bucket counts that do
    not divide 10000 evenly leave the remainder in the last part. A part whose
    share misses its requested share by more than half is rejected.
    """
    # Fixed ID types, seed, and ratios make hash buckets independent of row order.
    edges = _bucket_edges(ratios)
    bucket = F.pmod(F.xxhash64("uniqueID", F.lit(seed)), F.lit(_BUCKETS))
    frames = [
        df.filter((bucket >= start) & (bucket < end)).cache()
        for start, end in zip(edges, edges[1:])
    ]
    counts = [frame.count() for frame in frames]
    try:
        _check_balance(counts, ratios)
    except ValueError:
        for frame in frames:
            frame.unpersist(blocking=True)
        raise
    return tuple(frames)


def _check_balance(counts: Sequence[int], ratios: Sequence[int]) -> None:
    """Raise when a part's share is far from the ratio that was requested."""
    total = sum(counts)
    if total == 0:
        raise ValueError("split input has no rows")
    ratio_total = sum(ratios)
    for count, ratio in zip(counts, ratios):
        expected = ratio / ratio_total
        actual = count / total
        if abs(actual - expected) > _RELATIVE_TOLERANCE * expected:
            observed = ", ".join(str(count) for count in counts)
            raise ValueError(
                "hash split is unbalanced and does not match the requested ratios "
                f"{tuple(ratios)}: counts are ({observed}). "
                "This seed was already used to cut these rows."
            )


def weight(df: DataFrame) -> DataFrame:
    """Attach normalized square-root class weights and return a cached frame.

    `weight = n / (sqrt(class_count) * sum_sqrt)`. `sum_sqrt` sums
    `sqrt(class_count)` over classes, so the sample weights sum to `n`.
    The input cache is released after the weighted frame has been counted.
    """
    counts = df.groupBy("rating").count()
    totals = counts.agg(
        F.sum("count").alias("n"),
        F.sum(F.sqrt("count")).alias("sqrt_sum"),
    ).first()
    rating_weight = counts.withColumn(
        "weight",
        F.lit(float(totals["n"])) / (F.sqrt("count") * F.lit(float(totals["sqrt_sum"]))),
    ).select("rating", "weight")
    weighted = df.join(rating_weight, "rating", "left").cache()
    weighted.count()
    df.unpersist(blocking=True)
    return weighted
