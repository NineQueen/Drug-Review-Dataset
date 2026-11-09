from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import DataFrame

from src.analytics.base import AnalyticsModel


FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log"]

# Candidate values for future tuning; not used by the default run.
RF_PARAM_RANGES = {
    "numTrees": [100, 200, 300],
    "maxDepth": [8, 12, 16],
    "minInstancesPerNode": [1, 5, 20],
    "minInfoGain": [0.0, 0.001, 0.01],
    "subsamplingRate": [0.6, 0.8, 1.0],
    "featureSubsetStrategy": ["sqrt", "log2", "0.05"],
    "maxBins": [16, 32, 64],
    "impurity": ["gini", "entropy"],
}


@AnalyticsModel.register("random_forest")
def random_forest(
    self: AnalyticsModel, rf_params: dict | None = None
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Fit a class-weighted forest and return train, validation, and test predictions."""
    params = {
        "numTrees": 200,
        "maxDepth": 16,
        "minInstancesPerNode": 20,
        "maxMemoryInMB": 64,
        "seed": 42,
        **(rf_params or {}),
    }
    train_df, valid_df, test_df = self._feature_transform(FEATURES)
    model = RandomForestClassifier(
        featuresCol="features",
        labelCol="rating_p1",
        weightCol="weight",
        **params,
    ).fit(train_df)
    columns = ["uniqueID", "rating", "prediction"]
    return (
        model.transform(train_df).select(*columns),
        model.transform(valid_df).select(*columns),
        model.transform(test_df).select(*columns),
    )
