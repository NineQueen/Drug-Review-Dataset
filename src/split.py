from collections.abc import Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

_BUCKETS = 10000


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
    not divide 10000 evenly leave the remainder in the last part.
    """
    # Fixed ID types, seed, and ratios make hash buckets independent of row order.
    edges = _bucket_edges(ratios)
    bucket = F.pmod(F.xxhash64("uniqueID", F.lit(seed)), F.lit(_BUCKETS))
    frames = [
        df.filter((bucket >= start) & (bucket < end)).cache()
        for start, end in zip(edges, edges[1:])
    ]
    for frame in frames:
        frame.count()
    return tuple(frames)


def weight(df: DataFrame) -> DataFrame:
    """Attach `weight = n / (k * class_count)` and return a cached frame.

    The input cache is released after the weighted frame has been counted.
    """
    n = df.count()
    k = df.select("rating").distinct().count()
    rating_weight = (
        df.groupBy("rating").count()
        .withColumn("weight", F.lit(n) / (F.lit(k) * F.col("count")))
        .select("rating", "weight")
    )
    weighted = df.join(rating_weight, "rating", "left").cache()
    weighted.count()
    df.unpersist(blocking=True)
    return weighted
