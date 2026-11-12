import select
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
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
    "regParam": 0.1,
    "maxIter": 100,
    "elasticNetParam": 0.0,
    "weightCol": "weight",
}
PARA_RANGE = {
    "regParam": [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0],
}

@AnalyticsModel.baseline_register()
def lr_baseline(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_p1` is 0–9. This unweighted model is the required baseline.
    # Exclude the drug target mean so a training label cannot enter this feature.
    train_df, valid_df,test_df = self._feature_transform(FEATURES)
    lr_model = LogisticRegression(featuresCol="features", labelCol="rating_p1").fit(train_df)
    return _prediction_frames(lr_model, train_df, valid_df, test_df)

@AnalyticsModel.improved_baseline_register()
def improved_baseline(
    self: AnalyticsModel,
    para: dict | None = None,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_p1` is 0–9. Passed keys override DEFAULT_PARA.
    params = {
        **DEFAULT_PARA,
        **(para or {}),
    }
    train_df, valid_df, test_df = self._feature_transform(FEATURES)
    lr_improved_model = LogisticRegression(
        featuresCol="features",
        labelCol="rating_p1",
        **params,
    ).fit(train_df)
    return _prediction_frames(lr_improved_model, train_df, valid_df, test_df)
