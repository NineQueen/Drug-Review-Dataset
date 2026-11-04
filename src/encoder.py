from functools import cache
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, TargetEncoder
from pyspark.sql import DataFrame
from pyspark.sql.functions import transform


def encoder(df_train: DataFrame,df_valid:DataFrame, df_test: DataFrame) -> tuple[DataFrame, DataFrame, DataFrame]:
    # Fit on the training split only. An unseen validation or test value is kept rather than rejected.
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
        inputCol="drugName",
        outputCol="drugName_idx",
        handleInvalid="keep",
    )
    # Use the smoothed mean `rating` as a feature; keep `drugName_idx` as an auxiliary key.
    drug_mean = TargetEncoder(
        inputCol="drugName_idx",
        outputCol="drug_mean_rating",
        labelCol="rating",
        targetType="continuous",
        handleInvalid="keep",
        smoothing=10.0,
    )
    pipeline = Pipeline(stages=[condition_idx, condition_vec, drugName_idx, drug_mean])
    encoder_model = pipeline.fit(df_train)
    return (
        encoder_model.transform(df_train).cache(), 
        encoder_model.transform(df_valid).cache(),
        encoder_model.transform(df_test).cache()
    )
