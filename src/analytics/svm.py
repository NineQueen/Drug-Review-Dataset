from pyspark.ml.classification import LinearSVC
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.analytics.base import AnalyticsModel

FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_vec"]

# Interface contract: analytics reads only `embedding` plus the encoder's
# features (no words columns, nothing derived from the label) and fits on the
# training split only -- validation and test are transformed, never fitted.
# One flat parameter dict serves every embedding (interface.md); callers may
# override any key via `para`. Per-substrate tuned points (see
# Project/svm_run_log.md, 2026-11-10 matrices): tfidf alpha=1.0/reg=3.1
# (test acc .3981); glove alpha=0.82/reg=0.04 (test acc .3561); bert
# alpha=0.87/reg=0.1 (test acc .3335; five-metric win over lr_improved on
# valid); word2vec alpha=0.8/reg=0.15 (valid acc .3675, best acc among
# cells that beat lr_improved on four of five metrics -- the glove-style
# row). On every substrate weighted precision stays with lr_improved: it
# rewards head-class concentration, and the alpha-balanced margins
# deliberately counteract that (alpha-axis analysis in the run log).
# The default below is the cross-substrate middle ground so unseen
# embeddings still run sensibly.
DEFAULT_PARA = {
    "alpha": 1.0,   # class-weight exponent; 1.0 = fully balanced, 0.0 = unweighted
    "regParam": 1.0,
    "maxIter": 50,
}
# The grid contains each substrate's tuned point (glove 0.82/0.04~0.03,
# bert 0.87/0.1, word2vec 0.8/0.15) plus spread, so the shared k-fold
# tuning pipeline can reach every deliverable working point.
PARA_RANGE = {
    "alpha": [0.0, 0.5, 0.8, 0.82, 0.87, 1.0],
    "regParam": [0.03, 0.1, 0.15, 0.3, 1.0, 3.1],
}


@AnalyticsModel.register("svm")
def svm(self: AnalyticsModel, para: dict | None = None) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_p1` is 0-9. Passed keys override DEFAULT_PARA.
    # Manual One-vs-Rest: Spark's OneVsRest drops weightCol before fitting
    # each sub-classifier, so each binary task applies balanced weights
    # w = (n / (2 * n_class)) ** alpha itself and the prediction is the
    # argmax over the 10 margins.
    params = {
        **DEFAULT_PARA,
        **(para or {}),
    }
    train_df, valid_df, test_df = self._feature_transform(FEATURES)
    # Materialize the caches once: each binary fit re-reads these frames.
    train_df.count(); valid_df.count(); test_df.count()

    n = train_df.count()
    class_counts = {
        r["rating_p1"]: r["count"]
        for r in train_df.groupBy("rating_p1").count().collect()
    }
    classes = sorted(class_counts)

    def _fit_one(c):
        n_pos, n_neg = class_counts[c], n - class_counts[c]
        binary = train_df.withColumn(
            "bin_label", (F.col("rating_p1") == c).cast("int")
        ).withColumn(
            "bin_weight",
            F.when(F.col("bin_label") == 1,
                   F.lit((n / (2.0 * n_pos)) ** params["alpha"]))
             .otherwise(F.lit((n / (2.0 * n_neg)) ** params["alpha"])),
        )
        return LinearSVC(
            featuresCol="features",
            labelCol="bin_label",
            weightCol="bin_weight",
            maxIter=params["maxIter"],
            regParam=params["regParam"],
            standardization=True,
        ).fit(binary)

    models = [_fit_one(c) for c in classes]

    def _predict(frame: DataFrame) -> DataFrame:
        scored = frame
        for c, model in zip(classes, models):
            rp = f"rp_{c}"
            scored = model.setRawPredictionCol(rp).setPredictionCol(f"p_{c}").transform(scored)
            scored = scored.withColumn(f"s_{c}", vector_to_array(F.col(rp))[1]).drop(rp, f"p_{c}")
        best = F.greatest(*[f"s_{c}" for c in classes])
        prediction = None
        for c in classes:
            condition = F.col(f"s_{c}") == best
            prediction = (
                F.when(condition, F.lit(c)) if prediction is None
                else prediction.when(condition, F.lit(c))
            )
        # Ten one-vs-rest margins. Index c is class c.
        slots = {int(c): F.col(f"s_{c}") for c in classes}
        return scored.withColumn("prediction", prediction.cast("double")).withColumn(
            "probability",
            F.array(*[slots[c] for c in range(10)]),
        )

    columns = ["uniqueID", "rating", "prediction", "probability"]
    predictions = (
        _predict(train_df).select(*columns),
        _predict(valid_df).select(*columns),
        _predict(test_df).select(*columns),
    )
    # Release the assembled feature caches (base lifecycle contract); the
    # prediction frames above are lazy and read from the fitted models, so
    # they materialize independently of this cache.
    self.release_features()
    return predictions
