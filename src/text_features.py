from nltk.stem import PorterStemmer
from pyspark.ml import Pipeline
from pyspark.ml.feature import StopWordsRemover, Tokenizer
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType


def normalize_review_text(text: Column) -> Column:
    text = F.regexp_replace( # Unify apostrophe characters
        text,
        r"[\u2018\u2019\u02BC\uFF07\u0060\u00B4]",
        "'",
    )
    text = F.regexp_replace(text, r"[\u201C\u201D]", '"') # Unify double quote characters
    text = F.regexp_replace(text, r"\u200B", " ") # Remove zero-width space characters
    text = F.regexp_replace(text, r"\u200E", "") # Remove direction mark characters inside words
    text = F.regexp_replace(text, r"[\u0011\u007F\uFFFD]", "") # Remove control characters and damaged placeholders
    text = F.regexp_replace(text, r"\u0456", "i") # Replace Cyrillic small letter 'i'
    text = F.regexp_replace(text, r"i\u0307", "i") # Replace the observed dotted-i character
    return F.trim(F.regexp_replace(text, r"(?U)\s+", " "))


def text_feature_engineering(df: DataFrame) -> DataFrame:
    review_repaired = normalize_review_text(F.col("review"))
    remaining_repairs = df.filter(
        ~F.col("review").eqNullSafe(review_repaired)
    ).count()
    print(
        f"Before character repair: {remaining_repairs} `review` values "
        "require character or whitespace normalization"
    )

    df = df.withColumn("review", review_repaired)
    remaining_repairs = df.filter(
        ~F.col("review").eqNullSafe(
            normalize_review_text(F.col("review"))
        )
    ).count()
    print(
        f"After character repair: {remaining_repairs} `review` values "
        "require character or whitespace normalization"
    )

    df = df.withColumn("review_clean", F.col("review"))
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
