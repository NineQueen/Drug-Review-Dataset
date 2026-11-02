from nltk.stem import PorterStemmer
from pyspark.ml import Pipeline
from pyspark.ml.feature import NGram, RegexTokenizer, StopWordsRemover, Tokenizer
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

# Words with a basic negative form rule
_NEGATIVE_BASES = r"is|are|was|were|have|has|had|do|does|did|could|should|would|must|need|might|ought"

# Remove the spaces inside constractions
_CONTRACTION_SPACING_RULES = (
    (rf"({_NEGATIVE_BASES}|ca|wo|sha) *n *' *t", "$1n't"),
    (r"(i) *' *m", "$1'm"),
    (r"(you|we|they) *' *re", "$1're"),
    (r"(i|you|we|they|could|would|should|might|must) *' *ve", "$1've"),
    (r"(i|you|he|she|it|we|they|that|there|who|what) *' *ll", "$1'll"),
    (r"let *' *s", "let's"),
)

# Expand constractions according to their rules
_CONTRACTION_EXPANSION_RULES = (
    (r"can't", "can not"),
    (r"won't", "will not"),
    (r"shan't", "shall not"),
    (rf"({_NEGATIVE_BASES})n't", "$1 not"),
    (r"i'm", "i am"),
    (r"(you|we|they)'re", "$1 are"),
    (r"(i|you|we|they|could|would|should|might|must)'ve", "$1 have"),
    (r"(i|you|he|she|it|we|they|that|there|who|what)'ll", "$1 will"),
    (r"let's", "let us"),
)

def baseline_text_feature_engineering(df: DataFrame) -> DataFrame:
    """Basic text feature engineering for baseline TFIDF only: Tokenizer"""
    pipeline = Pipeline(stages=[
        Tokenizer(inputCol="review", outputCol="words_featured"),
    ])
    df = pipeline.fit(df).transform(df)
    return df

def normalize_review_text(text: Column) -> Column:
    """Repair and unify characters"""
    text = F.regexp_replace( # Unify apostrophe characters
        text,
        r"[\u2018\u2019\u02BC\uFF07]",
        "'",
    )
    text = F.regexp_replace(text, r"[\u201C\u201D]", '"') # Unify double quote characters
    text = F.regexp_replace(text, r"\u200B", " ") # Remove zero-width space characters
    text = F.regexp_replace(text, r"\u200E", "") # Remove direction mark characters inside words
    # Separate non-whitespace controls without joining surrounding words.
    text = F.regexp_replace(text, r"[\x00-\x08\x0E-\x1F\x7F-\x9F]", " ")
    return F.trim(F.regexp_replace(text, r"(?U)\s+", " "))

def expand_review_contractions(text: Column) -> Column:
    """Repair supported contractions in lowercased, character-normalized text."""
    for pattern, replacement in (*_CONTRACTION_SPACING_RULES, *_CONTRACTION_EXPANSION_RULES):
        text = F.regexp_replace(
            text,
            rf"\b{pattern}\b",
            replacement,
        )
    return text

def normalize_dose_spacing(text: Column) -> Column:
    """Normalize dose: 0.5 mg -> 0.5mg"""
    return F.regexp_replace(
        text,
        r"\b(\d+(?:\.\d+)?) +(mg|mcg|ml)\b",
        "$1$2",
    )

def text_feature_engineering(
        df: DataFrame,
        use_bigrams: bool = False,
    ) -> DataFrame:
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
        "would change on a second pass (idempotence check)"
    )

    expanded_review = expand_review_contractions(F.col("review"))
    changed_contractions = df.filter(
        ~F.col("review").eqNullSafe(expanded_review)
    ).count()
    df = df.withColumn("review", expanded_review)
    print(f"Contraction repair and expansion: {changed_contractions} `review` values changed")

    df = df.withColumn(
        "review",
        normalize_dose_spacing(F.col("review")),
    )

    df = df.withColumn("review_clean", F.col("review"))

    # Keep negative words when removing stop words
    protected_words = {
        "not", "no", "nor", "never", "neither", "without"
    }
    custom_stop_words = [
        word
        for word in StopWordsRemover.loadDefaultStopWords("english")
        if word not in protected_words
    ]

    pipeline = Pipeline(stages=[
        RegexTokenizer(
            inputCol="review_clean",
            outputCol="words",
            pattern=(
                r"(?:\d+(?:\.\d+)?|\.\d+)(?:mg|mcg|ml)\b"
                r"|\d*\.\d+"
                r"|[\p{L}\p{N}][\p{L}\p{M}\p{N}]*"
                r"(?:'[\p{L}\p{N}][\p{L}\p{M}\p{N}]*)*"
            ),
            gaps=False,
            minTokenLength=1,
            toLowercase=False,
        ),
        StopWordsRemover(
            inputCol="words",
            outputCol="words_nostop",
            stopWords=custom_stop_words,
        ),
    ])
    df = pipeline.fit(df).transform(df)

    @F.udf(ArrayType(StringType()))
    def stem_words(words):
        if words is None:
            return []

        stemmer = PorterStemmer()
        result = []

        # Retain words with digits (e.g., 0.5mg)
        for word in words:
            if not word:
                continue

            if any(char.isdigit() for char in word):
                result.append(word)
            else:
                result.append(stemmer.stem(word))

        return result

    df = df.withColumn(
        "words_featured",
        stem_words(F.col("words_nostop")),
    )

    if use_bigrams:
        df = df.withColumn(
            "_ngram_words",
            stem_words(F.col("words")),
        )

        df = NGram(
            n=2,
            inputCol="_ngram_words",
            outputCol="_bigrams",
        ).transform(df)

        df = df.withColumn(
            "words_featured",
            F.concat(
                F.col("words_featured"),
                F.col("_bigrams"),
            )
        )

    # TF-IDF uses stems; other embeddings can use the retained `review` or unstemmed `words`.
    return (
        df.select(
            "uniqueID",
            "drugName",
            "condition",
            "words_featured",
            "days",
            "usefulCount_log",
            "rating",
            "rating_p1",
            "review",
            "words",
        )
    )
