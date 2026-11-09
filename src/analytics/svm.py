from concurrent.futures import ThreadPoolExecutor

from pyspark.ml.classification import LinearSVC
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from src.analytics.base import AnalyticsModel

FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_vec"]

# Interface contract: analytics reads only `embedding` plus the encoder's
# features (no words columns, nothing derived from the label) and fits on the
# training split only -- validation and test are transformed, never fitted.
# Provenance in Project/svm_run_log.md (2026-11-08).

tfidf_para = {
    "alpha": 1.0,   # class-weight exponent; 1.0 = fully balanced, 0.0 = unweighted
    "regParam": 3.1,
}
glove_para = {
    # Tuned on the glove cache (2026-11-09, svm_glove_plan.md + dominance
    # sweep): the balanced-side point that beats lr_improved on ALL five
    # metrics on valid (acc .3341/wP .3664/wF1 .3479/MAE 1.900/RMSE 2.940 vs
    # lr .3237/.3583/.3376/1.974/3.034); the dominance window is alpha 0.85-
    # 0.95 (left edge fails weighted precision, right edge loses accuracy).
    # The acc-focused point alpha=0.2/reg=0.04 (valid acc 0.4017) is
    # documented in Project/svm_run_log.md as the accuracy-line alternative.
    "alpha": 0.9,
    "regParam": 0.04,
}

para_list = {
    "tfidf": tfidf_para,
    "glove": glove_para,
}


@AnalyticsModel.register("svm")
def svm(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    # `rating_p1` is 0-9. Manual One-vs-Rest: Spark's OneVsRest drops
    # weightCol before fitting each sub-classifier, so each binary task
    # applies balanced weights w = (n / (2 * n_class)) ** alpha itself and
    # the prediction is the argmax over the 10 margins.
    para = para_list[self.embedding_model]
    train_df, valid_df, test_df = self._feature_transform(FEATURES)
    # Materialize the caches once: the 10 concurrent fits below each re-read
    # these frames and would otherwise recompute the full lineage per fit.
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
                   F.lit((n / (2.0 * n_pos)) ** para["alpha"]))
             .otherwise(F.lit((n / (2.0 * n_neg)) ** para["alpha"])),
        )
        return LinearSVC(
            featuresCol="features",
            labelCol="bin_label",
            weightCol="bin_weight",
            maxIter=50,
            regParam=para["regParam"],
            standardization=True,
        ).fit(binary)

    with ThreadPoolExecutor(max_workers=6) as pool:
        models = list(pool.map(_fit_one, classes))

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
        return scored.withColumn("prediction", prediction.cast("double"))

    columns = ["uniqueID", "rating", "prediction"]
    return (
        _predict(train_df).select(*columns),
        _predict(valid_df).select(*columns),
        _predict(test_df).select(*columns),
    )
