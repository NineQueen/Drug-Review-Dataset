import select
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import transform
from src.analytics.base import AnalyticsModel
from src.paths import ANALYTIC_MODELS_DIR

FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_vec"]

DEFAULT_PARA = {
    "regParam": 0.1,
    "maxIter": 100,
    "elasticNetParam": 0.0,
    "weightCol": "weight",
}
PARA_RANGE = {
    "regParam": [0.01, 0.1, 0.3, 1.0],
}

@AnalyticsModel.baseline_register()
def lr_baseline(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_p1` is 0–9. This unweighted model is the required baseline.
    # Exclude the drug target mean so a training label cannot enter this feature.
    train_df, valid_df,test_df = self._feature_transform(FEATURES)
    lr_model = LogisticRegression(featuresCol="features", labelCol="rating_p1").fit(train_df)
    columns = ["uniqueID", "rating", "prediction"]
    return (
        lr_model.transform(train_df).select(*columns),
        lr_model.transform(valid_df).select(*columns),
        lr_model.transform(test_df).select(*columns),
    )

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
    columns = ["uniqueID", "rating", "prediction"]
    return (
        lr_improved_model.transform(train_df).select(*columns),
        lr_improved_model.transform(valid_df).select(*columns),
        lr_improved_model.transform(test_df).select(*columns),
    )
