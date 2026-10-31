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
                        F.countDistinct("rating").alias("rating_n"),
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
        df.agg(
            F.variance("usefulCount").alias("variance"),
            F.stddev("usefulCount").alias("stddev"),
            F.skewness("usefulCount").alias("skewness"),
        ).show()
        useful_ax = df.select("usefulCount").toPandas().plot.hist(bins=40, range=(0, 200), legend=False)
        useful_ax.set_xlabel("usefulCount")

        print("`review` length:")
        length = F.length(F.col("review"))
        df.select(length.alias("review_length")).summary("min", "25%", "50%", "75%", "max").show()
        df.agg(
            F.variance(length).alias("variance"),
            F.stddev(length).alias("stddev"),
            F.skewness(length).alias("skewness"),
        ).show()
        length_ax = df.select(length.alias("review_length")).toPandas().plot.hist(bins=40, range=(0, 1500), legend=False)
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

