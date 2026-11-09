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
        if width <= 0:
            raise ValueError("each ratio must receive at least one hash bucket")
        assigned += width
        edges.append(assigned)
    return edges


def split_and_weight(
    df: DataFrame,
    seed: int,
    ratios: Sequence[int] = (7, 1, 2),
) -> tuple[DataFrame, ...]:
    """Hash-split one frame per ratio and attach class weights to the first frame.

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
    # `weight` = n / (k * count), joined onto the first split only.
    n = frames[0].count()
    for frame in frames[1:]:
        frame.count()
    k = frames[0].select("rating").distinct().count()
    rating_weight = (
        frames[0].groupBy("rating").count()
        .withColumn("weight", F.lit(n) / (F.lit(k) * F.col("count")))
        .select("rating", "weight")
    )
    frames[0] = frames[0].join(rating_weight, "rating", "left").cache()
    return tuple(frames)
