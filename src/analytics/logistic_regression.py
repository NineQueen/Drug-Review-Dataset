from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.sql import DataFrame
from src.analytics.base import AnalyticsModel
from src.paths import ANALYTIC_MODELS_DIR

FEATURES = ["embedding", "condition_vec", "drug_mean_rating", "days", "usefulCount_log"]


@AnalyticsModel.train_register("lr")
def lr_train(self, embedding_model: str) -> None:
    # `rating_p1` is 0–9. This unweighted model is the required baseline.
    train_df, _ = self._feature_transform(FEATURES)
    self.lr_model = LogisticRegression(featuresCol="features", labelCol="rating_p1").fit(train_df)
    model_path = ANALYTIC_MODELS_DIR / f"lr__{embedding_model}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    self.lr_model.write().overwrite().save(str(model_path))


@AnalyticsModel.train_register("lr_weighted")
def lr_weighted_train(self, embedding_model: str) -> None:
    # `weight` exists only on the training split.
    train_df, _ = self._feature_transform(FEATURES)
    self.lr_weighted_model = LogisticRegression(
        featuresCol="features",
        labelCol="rating_p1",
        weightCol="weight",
    ).fit(train_df)
    model_path = ANALYTIC_MODELS_DIR / f"lr_weighted__{embedding_model}"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    self.lr_weighted_model.write().overwrite().save(str(model_path))

@AnalyticsModel.predict_register("lr")
def lr_predict(self, embedding_model: str) -> tuple[DataFrame, DataFrame]:
    train_df, test_df = self._feature_transform(FEATURES)
    if getattr(self, "lr_model", None) is None:
        self.lr_model = LogisticRegressionModel.load(str(ANALYTIC_MODELS_DIR / f"lr__{embedding_model}"))
    
    # Keep the original `rating`; only `prediction` uses the 0–9 class scale.
    columns = ["uniqueID", "rating", "prediction"]
    return (
        self.lr_model.transform(train_df).select(*columns),
        self.lr_model.transform(test_df).select(*columns),
    )


@AnalyticsModel.predict_register("lr_weighted")
def lr_weighted_predict(self, embedding_model: str) -> tuple[DataFrame, DataFrame]:
    train_df, test_df = self._feature_transform(FEATURES)
    if getattr(self, "lr_weighted_model", None) is None:
        self.lr_weighted_model = LogisticRegressionModel.load(str(ANALYTIC_MODELS_DIR / f"lr__{embedding_model}"))
    
    columns = ["uniqueID", "rating", "prediction"]
    return (
        self.lr_weighted_model.transform(train_df).select(*columns),
        self.lr_weighted_model.transform(test_df).select(*columns),
    )
