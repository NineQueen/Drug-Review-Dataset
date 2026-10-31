from nltk.stem import PorterStemmer
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType


def text_feature_engineering(df: DataFrame) -> DataFrame:
    # Keep digits and join integer doses to their units, such as `10 mg` to `10mg`.
    df = df.withColumn(
        "review_clean",
        F.col("review"),
    )
    df = df.withColumn(
        "review_clean",
        F.regexp_replace(F.col("review_clean"), r"(\d+)\s+(mg|ml|mcg)\b", r"$1$2"),
    )
    df = df.withColumn(
        "review_clean",
        F.trim(F.regexp_replace(
            F.regexp_replace(F.col("review_clean"), r"[^a-z0-9\s]", " "),
            r"\s+",
            " ",
        )),
    )
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="review_clean", outputCol="words"),
        StopWordsRemover(inputCol="words", outputCol="words_nostop"),
    ])
    df = pipeline.fit(df).transform(df)

    @F.udf(ArrayType(StringType()))
    def stem_words(words):
        stemmer = PorterStemmer()
        if words is None:
            return []
        return [stemmer.stem(word) for word in words]

    # TF-IDF uses stems; other embeddings can use the retained `review` or unstemmed `words`.
    return (
        df.withColumn("words_stem", stem_words(F.col("words_nostop")))
        .select(
            "uniqueID",
            "drugName",
            "condition",
            "words_stem",
            "days",
            "usefulCount_log",
            "rating",
            "rating_p1",
            "review",
            "words",
        )
    )
