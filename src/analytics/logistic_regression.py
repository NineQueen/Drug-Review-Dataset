import select
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.feature import Normalizer
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import transform
from src.analytics.base import AnalyticsModel
from src.paths import ANALYTIC_MODELS_DIR

PREDICTION_COLUMNS = ["uniqueID", "rating", "prediction", "probability"]


def _prediction_frames(model, *frames: DataFrame) -> tuple[DataFrame, ...]:
    return tuple(
        model.transform(frame)
        .withColumn("probability", vector_to_array("probability"))
        .select(*PREDICTION_COLUMNS)
        for frame in frames
    )

FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_vec"]

DEFAULT_PARA = {
    "regParam": 1e-5,
    "maxIter": 300,
    "elasticNetParam": 0.0,
    "weightCol": "weight",
}
PARA_RANGE = {
    "regParam": [3e-6, 1e-5, 3e-5, 1e-4],
}

@AnalyticsModel.baseline_register()
def lr_baseline(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_m1` is 0–9. This unweighted model is the required baseline.
    # Exclude the drug target mean so a training label cannot enter this feature.
    train_df, valid_df,test_df = self._feature_transform(FEATURES)
    lr_model = LogisticRegression(featuresCol="features", labelCol="rating_m1").fit(train_df)
    return _prediction_frames(lr_model, train_df, valid_df, test_df)


def _l2_embedding(frame: DataFrame) -> DataFrame:
    # Spark IDF does not length-normalize rows; without this the unstandardized fit does not converge.
    return (
        Normalizer(inputCol="embedding", outputCol="_embedding_l2", p=2.0)
        .transform(frame)
        .drop("embedding")
        .withColumnRenamed("_embedding_l2", "embedding")
    )


def _scale_days(train: DataFrame, valid: DataFrame, test: DataFrame):
    """Replace `days` with its training z-score. Later splits only reuse that fit."""
    row = train.agg(F.mean("days").alias("mean"), F.stddev("days").alias("std")).collect()[0]
    center = float(row["mean"])
    scale = float(row["std"] or 0.0)
    if scale < 1e-8:
        scale = 1.0

    def apply(frame: DataFrame) -> DataFrame:
        return frame.withColumn("days", (F.col("days") - F.lit(center)) / F.lit(scale))

    return tuple(apply(frame) for frame in (train, valid, test))


@AnalyticsModel.improved_baseline_register()
def improved_baseline(
    self: AnalyticsModel,
    para: dict | None = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_m1` is 0–9. Passed keys override DEFAULT_PARA.
    # Standardization stays off, so only this model scales `days` from the training rows.
    params = {
        **DEFAULT_PARA,
        **(para or {}),
    }
    params.pop("standardization", None)
    stored = (self.train_embedding, self.valid_embedding, self.test_embedding)
    normalized = tuple(_l2_embedding(frame) for frame in stored)
    self.train_embedding, self.valid_embedding, self.test_embedding = _scale_days(*normalized)
    try:
        train_df, valid_df, test_df = self._feature_transform(FEATURES)
    finally:
        self.train_embedding, self.valid_embedding, self.test_embedding = stored
    lr_improved_model = LogisticRegression(
        featuresCol="features",
        labelCol="rating_m1",
        standardization=False,
        **params,
    ).fit(train_df)
    return _prediction_frames(lr_improved_model, train_df, valid_df, test_df)
