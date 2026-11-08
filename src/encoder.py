from functools import cache
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, TargetEncoder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import transform

_MIN_DRUG_COUNT = 10
_OTHER_DRUG = "__other__"


def _collapse_rare_drugs(df: DataFrame, frequent_drugs: DataFrame) -> DataFrame:
    return (
        df.join(frequent_drugs, "drugName", "left")
        .withColumn(
            "drugName_grouped",
            F.when(F.col("_frequent"), F.col("drugName")).otherwise(F.lit(_OTHER_DRUG)),
        )
        .drop("_frequent")
    )


def encoder(df_train: DataFrame,df_valid:DataFrame, df_test: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    # Fit on the training split only. An unseen validation or test value is kept rather than rejected.
    frequent_drugs = (
        df_train.groupBy("drugName")
        .count()
        .filter(F.col("count") >= _MIN_DRUG_COUNT)
        .select("drugName", F.lit(True).alias("_frequent"))
    )
    df_train = _collapse_rare_drugs(df_train, frequent_drugs)
    df_valid = _collapse_rare_drugs(df_valid, frequent_drugs)
    df_test = _collapse_rare_drugs(df_test, frequent_drugs)

    condition_idx = StringIndexer(
        inputCol="condition",
        outputCol="condition_idx",
        handleInvalid="keep",
    )
    condition_vec = OneHotEncoder(
        inputCol="condition_idx",
        outputCol="condition_vec",
        handleInvalid="keep",
    )
    drugName_idx = StringIndexer(
        inputCol="drugName_grouped",
        outputCol="drugName_idx",
        handleInvalid="keep",
    )
    drugName_vec = OneHotEncoder(
        inputCol="drugName_idx",
        outputCol="drugName_vec",
        handleInvalid="keep",
    )
    pipeline = Pipeline(stages=[condition_idx, condition_vec, drugName_idx, drugName_vec])
    encoder_model = pipeline.fit(df_train)
    return (
        encoder_model.transform(df_train).cache(), 
        encoder_model.transform(df_valid).cache(),
        encoder_model.transform(df_test).cache()
    )
