from pyspark.sql import DataFrame, functions as F

from src.analytics.base import AnalyticsModel


FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log"]
WEIGHT_POWER = 0.35

# Selected by accuracy on a holdout from the GloVe training pool.
DEFAULT_PARA = {
    "n_estimators": 184,
    "max_depth": 6,
    "learning_rate": 0.1,
    "min_child_weight": 35.0,
    "subsample": 0.8,
    "colsample_bytree": 0.85,
    "reg_lambda": 30.0,
    "reg_alpha": 0.5,
    "gamma": 0.2,
    "tree_method": "hist",
    "max_bin": 64,
    "eval_metric": "mlogloss",
    "random_state": 42,
    "num_workers": 1,
    "weight_col": "weight",
}

# Candidate values only; the registered function fits one configuration.
PARA_RANGE = {
    "n_estimators": [100, 184, 200, 300, 600],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.04, 0.05, 0.08, 0.1],
    "min_child_weight": [5.0, 10.0, 20.0, 35.0],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_lambda": [1.0, 5.0, 10.0, 20.0, 30.0],
    "reg_alpha": [0.0, 0.1, 0.5],
    "gamma": [0.0, 0.1, 0.2],
}


@AnalyticsModel.register("xgboost")
def xgboost(
    self: AnalyticsModel, para: dict | None = None
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Fit on training rows only and return the three prediction frames."""
    from xgboost.spark import SparkXGBClassifier

    params = {**DEFAULT_PARA, **(para or {})}
    train_df, valid_df, test_df = self._feature_transform(FEATURES)
    train_df = train_df.withColumn(
        "weight", F.pow(F.col("weight"), F.lit(WEIGHT_POWER))
    )
    # SparkXGBClassifier infers the multiclass objective from training labels.
    model = SparkXGBClassifier(
        features_col="features",
        label_col="rating_p1",
        prediction_col="prediction",
        **params,
    ).fit(train_df)
    columns = ["uniqueID", "rating", "prediction"]
    return tuple(
        model.transform(frame)
        .withColumn("prediction", F.col("prediction").cast("double"))
        .select(*columns)
        for frame in (train_df, valid_df, test_df)
    )
