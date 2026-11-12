import json

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.column import Column

from src.paths import REPLACEMENT_JSON



def _peel_end_pair(column: Column, marker: str) -> Column:
    """Remove one copy of `marker` when it is present at both ends."""
    width = len(marker)
    interior = F.trim(F.substring(column, width + 1, F.length(column) - (2 * width)))
    matched = (
        F.startswith(column, F.lit(marker))
        & F.endswith(column, F.lit(marker))
        & (F.length(column) >= 2 * width)
    )
    return F.when(matched, interior).otherwise(column)


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

        review = F.trim(F.regexp_replace(F.column("review"), r"\s+", " "))

        # AI change: decide wrappers at quote replacement, instead of `^"|"$`.
        # That pattern deleted a quote on either end, so a second pass removed
        # body quotes such as the closing quote of "side-effect" and changed dedup.
        # One "..." pair is the content wrapper on every parsed review. `&quot;`
        # on both ends is a second wrapper; an entity on only one end is a citation
        # and is decoded by the map below.
        review = _peel_end_pair(review, '"')
        review = _peel_end_pair(review, "&quot;")
        # JSON order matters: `&amp;` is last, so other entities are decoded first.
        for old, new in replacement["html_entities"].items():
            review = F.regexp_replace(review, old, new)
        review = F.trim(F.regexp_replace(review, r"\s+", " "))
        return df.withColumn("review", review)

    # Stage lines only. Counts and summaries are Spark actions and would
    # recompute this frame before the caller caches it.
    print("=" * 32)
    print("Data Cleaning")
    print("=" * 32)

    df = df.dropna(subset="condition")
    print("Dropped rows with a null `condition`")

    df = df.filter(~F.col("condition").contains("</span> users found this comment helpful"))
    print("Dropped rows whose `condition` contains '</span> users found this comment helpful'")

    df = replace_condition_and_review(df)
    print("Applied configured `condition` repairs and HTML-entity decoding")

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
    survivor = df.join(df_best, ["review", "usefulCount", "uniqueID"]).select(*df.columns)
    survivor_key = survivor.select(
        "review",
        F.col("rating").alias("_rating"),
        F.col("condition").alias("_condition"),
        F.col("date").alias("_date"),
    )
    drug_names = (
        df.join(survivor_key, "review")
        .where(
            F.col("rating").eqNullSafe(F.col("_rating"))
            & F.col("condition").eqNullSafe(F.col("_condition"))
            & F.col("date").eqNullSafe(F.col("_date"))
        )
        .groupBy("review")
        .agg(F.sort_array(F.collect_set("drugName")).alias("drugNames"))
    )
    df = survivor.join(drug_names, "review")
    print("Kept the highest `usefulCount` row for each `review`")
    print("Collected `drugNames` whose rating, condition, and date match that row")

    # `usefulCount` is skewed and counts votes cast after the review was published.
    df = df.withColumn(
        "usefulCount_log",
        F.log1p("usefulCount")
    )
    print("Added `usefulCount_log` = log(1 + `usefulCount`)")

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


def webmd_data_cleaning(df: DataFrame) -> DataFrame:
    with open(REPLACEMENT_JSON, "r", encoding="utf-8") as f:
        html_entities = json.load(f)["html_entities"]

    review = F.col("review")
    for old, new in html_entities.items():
        review = F.regexp_replace(review, old, new)
    review = F.lower(F.trim(F.regexp_replace(review, r"(?U)\s+", " ")))

    return df.withColumn("review", review).filter(F.length("review") > 0)