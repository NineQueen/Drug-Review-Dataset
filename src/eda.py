import matplotlib.pyplot as plt

from pyspark.sql import functions as F
from pyspark.sql import DataFrame


class EDA:
    def __init__(self, df: DataFrame) -> None:
        self.df = df

    def dataframe_overview(self) -> None:
        df = self.df
        print("="*32)
        print("DataFrame overview")
        print("="*32)

        # Raw reviews can contain HTML entities such as `&#039;`.
        df.show(5,truncate=80)
        print("Schema:")
        df.printSchema()

    def granularity(self) -> None:
        df = self.df
        print("="*32)
        print("DataFrame granularity")
        print("="*32)

        print("Total row count:", end = " ")
        print(df.count())
        print("`drugName`:")
        df.select("drugName").distinct().show(truncate=50)
        print("`drugName` count:", end = " ")
        print(df.select("drugName").distinct().count())
        print("`condition`:")
        df.select("condition").distinct().show(truncate=50)
        print("`condition` count:",end=" ")
        # Raw `condition` values can include fragments of helpful-vote markup.
        print(df.select("condition").distinct().count())
        print("`uniqueID` is unique" if df.count() == df.select("uniqueID").distinct().count()
            else  "`uniqueID` is not unique")

        df_review = df.groupBy("review").agg(
                        F.count("*").alias("n"),
                        F.count_distinct("rating").alias("rating_n"),
                    ).filter((F.col("n") > 1))
        review_count = df_review.count()
        if review_count == 0:
            print("`review` values are unique")
        else:
            print(f"`review` is not unique: {review_count} groups contain duplicate `review` values")
            df_review.show(5,truncate=80)
            print(
                "But only {} groups of `review` have different `rating` values".format(
                    df_review.filter(F.col("rating_n") > 1).count()
                )
            )

    def distribution_inspect(self) -> None:
        df = self.df
        print("="*32)
        print("Data Distribution")
        print("="*32)

        print("`rating` is in the range [1, 10]" if df.filter((F.col("rating") >= 1) & (F.col("rating") <= 10)).count() == df.count()
            else "`rating` is not in the range [1, 10]")

        print("`rating` record count:")
        rating_count = df.groupBy("rating").count().orderBy("rating")
        rating_count.show()
        df.agg(
            F.variance("rating").alias("variance"),
            F.stddev("rating").alias("stddev"),
            F.skewness("rating").alias("skewness"),
        ).show()
        rating_count.toPandas().plot.bar(x="rating", y="count", legend=False)

        print("`usefulCount`:")
        df.select("usefulCount").summary("min", "25%", "50%", "75%", "max").show()
        # Here we can see duplicate records
        # The same information, but with different drug names.
        print("The highest `usefulCount` record:")
        max_useful = df.agg(F.max("usefulCount")).collect()[0][0]
        df.filter(F.col("usefulCount") == max_useful).show(truncate=80)
        df.agg(
            F.variance("usefulCount").alias("variance"),
            F.stddev("usefulCount").alias("stddev"),
            F.skewness("usefulCount").alias("skewness"),
        ).show()
        useful_ax = df.select("usefulCount").toPandas().plot.hist(bins=40, range=(0, 200), legend=False)
        useful_ax.set_xlabel("usefulCount")

        print("`review` length:")
        df_length = df.withColumn(
            "review_length",
            F.length(F.col("review"))
        )
        max_length = df_length.agg(F.max("review_length")).collect()[0][0]
        min_length = df_length.agg(F.min("review_length")).collect()[0][0]
        df_length.select("review_length").summary("min", "25%", "50%", "75%", "max").show()
        print("The longest `review` record:")
        df_length.filter(F.col("review_length") == max_length).select(*df.columns).show(truncate = 80)
        print("The abnormal short `review` records (shorter than 4 characters):")
        df_length.filter(F.col("review_length") == min_length).select(*df.columns).show(truncate = 80)
        df_length.agg(
            F.variance("review_length").alias("variance"),
            F.stddev("review_length").alias("stddev"),
            F.skewness("review_length").alias("skewness"),
        ).show()
        length_ax = df_length.select("review_length").toPandas().plot.hist(bins=40, range=(0, 1500), legend=False)
        length_ax.set_xlabel("review_length")

        for col in df.columns:
            print(f"`{col}` has {df.filter(F.col(col).isNull()).count()} null records")

    def correlation(self) -> None:
        df = self.df
        print("="*32)
        print("Data Field Correlation")
        print("="*32)
        numeric = df.select(
            "rating",
            "usefulCount",
            F.unix_date("date").alias("date"),
        )
        columns = numeric.columns
        numeric.agg(
            *(
                F.corr(columns[i], columns[j]).alias(f"{columns[i]}_{columns[j]}")
                for i in range(len(columns))
                for j in range(i + 1, len(columns))
            )
        ).show()

    def correlation_heatmap(self) -> None:
        df = self.df
        print("="*32)
        print("Data Field Correlation Heatmap")
        print("="*32)

        numeric = df.select(
            "rating",
            "usefulCount",
            F.unix_date("date").alias("date"),
        )
        columns = numeric.columns
        # Collect only the coefficients, not the full review dataset.
        correlations = numeric.agg(
            *(
                F.corr(columns[i], columns[j]).alias(f"r_{i}_{j}")
                for i in range(len(columns))
                for j in range(len(columns))
            )
        ).collect()[0]
        corr_matrix = [
            [correlations[f"r_{i}_{j}"] for j in range(len(columns))]
            for i in range(len(columns))
        ]

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
        ax.set_yticks(range(len(columns)), columns)
        ax.set(
            title="Pairwise Pearson correlation in raw reviews",
            xlabel="Numeric variable",
            ylabel="Numeric variable",
        )
        for i in range(len(columns)):
            for j in range(len(columns)):
                value = corr_matrix[i][j]
                ax.text(
                    j, i, f"{value:.3f}", ha="center", va="center",
                    color="white" if abs(value) > .6 else "black",
                )
        fig.colorbar(im, ax=ax, label="Pearson r (-1 to 1)")
        fig.tight_layout()
        plt.show()
