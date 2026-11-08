from pyspark.ml.classification import LinearSVC, OneVsRest
from pyspark.sql import DataFrame

from src.analytics.base import AnalyticsModel

# Same feature list as the LR baseline for comparability.
FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_idx"]


@AnalyticsModel.register("svm_l0")
def svm_l0(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    """L0: plain linear SVM wrapped in Spark's One-vs-Rest (unweighted).

    `LinearSVC` solves binary problems only; `OneVsRest` trains one
    classifier per class and predicts the class with the highest margin.
    `standardization=True` is built in, so mixed feature scales (`days` ~1e4,
    embedding ~1) are handled inside the solver. Tuned successors live in
    `svm_l1.py` (class-weighted variants) and are compared side by side in
    the runner.
    """
    train_df, valid_df, test_df = self._feature_transform(FEATURES)
    # Materialize the caches before fitting: OneVsRest triggers one action
    # per class, and without this each fit recomputes the (Python-UDF)
    # embedding lineage of the training split. Mirrors svm_l1/svm_tfidf_l2.
    train_df.count(); valid_df.count(); test_df.count()
    svc = LinearSVC(
        featuresCol="features",
        labelCol="rating_p1",
        maxIter=50,
        regParam=0.1,
        standardization=True,
    )
    ovr = OneVsRest(
        classifier=svc,
        featuresCol="features",
        labelCol="rating_p1",
        parallelism=6,
    )
    model = ovr.fit(train_df)
    columns = ["uniqueID", "rating", "prediction"]
    return (
        model.transform(train_df).select(*columns),
        model.transform(valid_df).select(*columns),
        model.transform(test_df).select(*columns),
    )
