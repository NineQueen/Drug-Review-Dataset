import select
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import transform
from src.analytics.base import AnalyticsModel
from src.paths import ANALYTIC_MODELS_DIR

FEATURES = ["embedding", "condition_vec", "drug_mean_rating", "days", "usefulCount_log"]


@AnalyticsModel.baseline_register()
def lr_baseline(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_p1` is 0–9. This unweighted model is the required baseline.
    train_df, valid_df,test_df = self._feature_transform(FEATURES)
    lr_model = LogisticRegression(featuresCol="features", labelCol="rating_p1").fit(train_df)
    columns = ["uniqueID", "rating", "prediction"]
    return (
        lr_model.transform(train_df).select(*columns),
        lr_model.transform(valid_df).select(*columns),
        lr_model.transform(test_df).select(*columns),
    )

@AnalyticsModel.register("lr_improved")
def lr_improved(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_p1` is 0–9. This unweighted model is the required baseline.
    train_df, valid_df,test_df = self._feature_transform(FEATURES)
    lr_improved_model = LogisticRegression(
        featuresCol="features",
        labelCol="rating_p1",
        weightCol="weight",
        maxIter=100,
        regParam=0.1,
        elasticNetParam=0.0,
    ).fit(train_df)
    columns = ["uniqueID", "rating", "prediction"]
    return (
        lr_improved_model.transform(train_df).select(*columns),
        lr_improved_model.transform(valid_df).select(*columns),
        lr_improved_model.transform(test_df).select(*columns),
    )