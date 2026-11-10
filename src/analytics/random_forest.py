from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import DataFrame, functions as F

from src.analytics.base import AnalyticsModel


FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log"]
# Selected on training-only folds with accuracy as the primary metric.
WEIGHT_POWER = 0.75

DEFAULT_PARA = {
    "numTrees": 150,
    "maxDepth": 10,
    "minInstancesPerNode": 50,
    "minInfoGain": 0.0001,
    "subsamplingRate": 0.8,
    "featureSubsetStrategy": "sqrt",
    "maxBins": 32,
    "impurity": "gini",
    "maxMemoryInMB": 256,
    "seed": 42,
    "weightCol": "weight",
}

# Evaluated candidates; the default run trains only one forest.
PARA_RANGE = {
    "numTrees": [50, 100, 150],
    "maxDepth": [8, 10],
    "minInstancesPerNode": [50, 100],
    "minInfoGain": [0.0001, 0.001],
    "subsamplingRate": [0.6, 0.8],
    "featureSubsetStrategy": ["sqrt"],
    "maxBins": [32],
    "impurity": ["gini"],
}


@AnalyticsModel.register("random_forest")
def random_forest(
    self: AnalyticsModel, para: dict = DEFAULT_PARA
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Fit the weighted forest and return train, validation, and test predictions."""
    params = {**DEFAULT_PARA, **para}
    train_df, valid_df, test_df = self._feature_transform(FEATURES)
    train_df = train_df.withColumn(
        "weight", F.pow(F.col("weight"), F.lit(WEIGHT_POWER))
    )
    model = RandomForestClassifier(
        featuresCol="features",
        labelCol="rating_p1",
        **params,
    ).fit(train_df)
    columns = ["uniqueID", "rating", "prediction"]
    return (
        model.transform(train_df).select(*columns),
        model.transform(valid_df).select(*columns),
        model.transform(test_df).select(*columns),
    )
