from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def split_and_weight(df: DataFrame, seed: int) -> tuple[DataFrame, DataFrame, DataFrame]:
    # Fixed ID types and seed make hash buckets independent of row order and partitioning.
    # The resulting train/valid/test ratio is approximately 70/10/20.
    bucket = F.pmod(F.xxhash64("uniqueID", F.lit(seed)), F.lit(10000))
    df_train = df.filter(bucket < 7000)
    df_valid = df.filter((bucket >= 7000) & (bucket < 8000))
    df_test = df.filter(bucket >= 8000)
    df_train = df_train.cache()
    df_valid = df_valid.cache()
    df_test = df_test.cache()
    # `weight` = n / (k * count), joined onto the training split only.
    n = df_train.count()
    df_valid.count()
    df_test.count()
    k = df_train.select("rating").distinct().count()
    rating_weight = (
        df_train.groupBy("rating").count()
        .withColumn("weight", F.lit(n) / (F.lit(k) * F.col("count")))
        .select("rating", "weight")
    )
    df_train = df_train.join(rating_weight, "rating", "left").cache()
    return df_train, df_valid, df_test
