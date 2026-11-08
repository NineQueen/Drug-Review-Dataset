import select
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.sql import DataFrame
from pyspark.sql.functions import transform
from src.analytics.base import AnalyticsModel
from src.paths import ANALYTIC_MODELS_DIR

FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_vec"]

tfidf_para = {
    "regParam" : 0.1
}
glove_para = {
    "regParam" : 0.03
}

para_list = {
    "tfidf": tfidf_para,
    "glove": glove_para 
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

@AnalyticsModel.register("lr_improved")
def lr_improved(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_p1` is 0–9. This unweighted model is the required baseline.
    train_df, valid_df,test_df = self._feature_transform(FEATURES)
    lr_improved_model = LogisticRegression(
        featuresCol="features",
        labelCol="rating_p1",
        weightCol="weight",
        maxIter=100,
        elasticNetParam=0.0,
        **para_list[self.embedding_model]
    ).fit(train_df)
    columns = ["uniqueID", "rating", "prediction"]
    return (
        lr_improved_model.transform(train_df).select(*columns),
        lr_improved_model.transform(valid_df).select(*columns),
        lr_improved_model.transform(test_df).select(*columns),
    )
