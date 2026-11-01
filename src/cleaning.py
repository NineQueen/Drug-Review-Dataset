import json

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.paths import REPLACEMENT_JSON


def data_cleaning(df: DataFrame) -> DataFrame:
    with open(REPLACEMENT_JSON, "r", encoding="utf-8") as f:
        replacement = json.load(f)

    def replace_condition_and_review(df: DataFrame) -> DataFrame:
        pairs = []
        for old, new in replacement["condition"].items():
            pairs.extend((F.lit(old), F.lit(new)))
        # Exact match on the full `condition` string. Unlisted values stay unchanged.
        condition_map = F.create_map(*pairs)
        df = df.withColumn(
            "condition",
            F.coalesce(condition_map[F.col("condition")], F.col("condition")),
        )

        review = F.col("review")
        # JSON order matters: `&amp;` is last, so other entities are decoded first.
        for old, new in replacement["html_entities"].items():
            review = F.regexp_replace(review, old, new)
        return df.withColumn("review", review)

    print("="*32)
    print("Data Cleaning")
    print("="*32)

    row_count = df.count()
    df = df.dropna(subset="condition")
    print(f"After dropping rows with a null `condition`\nRow count: {row_count} -> {df.count()}, removed {row_count - df.count()} rows")
    row_count = df.count()

    df = df.filter(~F.col("condition").contains("</span> users found this comment helpful"))
    print(f"After dropping rows whose `condition` contains '</span> users found this comment helpful'\nRow count: {row_count} -> {df.count()}, removed {row_count - df.count()} rows")

    # This counts only configured condition repairs, not every possible malformed name.
    remaining_conditions = df.filter(F.col("condition").isin(list(replacement["condition"]))).count()
    remaining_entities = df.filter(F.col("review").rlike("&(#x?[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")).count()
    print(f"Before replacement: {remaining_conditions} `condition` values match configured repairs, {remaining_entities} `review` values contain HTML entities")
    df = replace_condition_and_review(df)
    remaining_conditions = df.filter(F.col("condition").isin(list(replacement["condition"]))).count()
    remaining_entities = df.filter(F.col("review").rlike("&(#x?[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")).count()
    print(f"After replacement: {remaining_conditions} `condition` values still match configured repairs, {remaining_entities} `review` values contain HTML entities")

    df = df.withColumn(
        "review",
        F.regexp_replace(F.trim(F.regexp_replace("review", r'^"|"$', "")), r"\s+", " "),
    )

    # `dropDuplicates` does not define which copy survives. Keep the greater `usefulCount`, then the greater `uniqueID`.
    row_count = df.count()
    df = df.withColumn("review", F.lower("review"))
    df_best = (
        df.groupBy("review")
        .agg(F.max(F.struct("usefulCount", "uniqueID")).alias("best"))
        .select(
            "review",
            F.col("best.usefulCount").alias("usefulCount"),
            F.col("best.uniqueID").alias("uniqueID"),
        )
    )
    df = df.join(df_best, ["review", "usefulCount", "uniqueID"]).select(*df.columns)
    print(f"After keeping the row with the highest `usefulCount` for each `review`\nRow count: {row_count} -> {df.count()}, removed {row_count - df.count()} rows")

    # `usefulCount` is skewed and counts votes cast after the review was published.
    df = df.withColumn(
        "usefulCount_log",
        F.log1p("usefulCount")
    )
    print("Added `usefulCount_log` = log(1 + `usefulCount`)")

    useful_stats = df.agg(
        F.lit("usefulCount").alias("column"),
        F.variance("usefulCount").alias("variance"),
        F.stddev("usefulCount").alias("stddev"),
        F.skewness("usefulCount").alias("skewness"),
    )
    useful_log_stats = df.agg(
        F.lit("usefulCount_log").alias("column"),
        F.variance("usefulCount_log").alias("variance"),
        F.stddev("usefulCount_log").alias("stddev"),
        F.skewness("usefulCount_log").alias("skewness"),
    )
    useful_stats.union(useful_log_stats).show()

    # One numeric date column. `rating_p1` is the 0–9 classification label.
    print("Added `days` = days since 1970-01-01 from `date`")
    df = df.withColumn(
        "days",
        F.unix_date("date")
    )

    print("Set `rating_p1` = `rating` - 1")
    df = df.withColumn(
        "rating_p1",
        F.col("rating") - 1
    )
    return df
