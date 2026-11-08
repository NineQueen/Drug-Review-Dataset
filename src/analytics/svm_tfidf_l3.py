from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC
from pyspark.ml.feature import CountVectorizer, IDF, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.analytics.base import AnalyticsModel

# L3 forks from `svm_tfidf_l2`: same weighted-OvR mechanism, same features
# (drug_target, s=30), but with the Tier-1 acc stack applied:
# 1. ALPHA=0.9 -- weight exponent softened from 1.0. Swept on tfidf
#    (experiments/svm_l2_alpha_results.json): valid acc 0.4106 vs 0.3967 at
#    alpha=1.0 (+1.39pp), at the disclosed cost of weighted-F1 dropping
#    below lr_improved (acc-weighted priority, decision 2026-11-08).
#    reg stays at L2's baked 3.1 (the alpha sweep ran at reg=3.0; the
#    difference is inside the noise band).
# 2. Per-class margin calibration delta_c -- fitted on the validation split
#    by coordinate ascent maximizing accuracy, then applied to all three
#    splits. This subsumes a low-confidence majority fallback (a large
#    positive delta on the prior mode IS that fallback). Validation scores
#    are mildly optimistic (10 free parameters); TEST stays honest.
#    The k-fold run re-fits delta per fold, so this stays leak-free there.
# 3. Private 15k vocabulary -- L3 re-vectorizes `words_featured` itself
#    (fit on the training split only) instead of using the shared `embedding`
#    column. The vocab sweep (experiments/svm_l3_tfidf_sweep_results.json)
#    showed post-calibration acc is flat 5k-40k, and 15k ties the baked 20k
#    on acc with the best balance trio (wF1 +1.4pp, macro +0.8pp, MAE
#    better). tfidf.py stays at 20000 so the rest of the matrix (L0/L1/L2/
#    LR) keeps its archived numbers. Decision 2026-11-08, k-fold to confirm.
ALPHA = 0.9
REG_PARAM = 3.1
MAX_ITER = 50
SMOOTHING = 30.0  # same value as svm_tfidf_l2
VOCAB_SIZE = 15000
MIN_DF = 5.0

FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drug_target"]


def _fit_deltas(margins: np.ndarray, y_idx: np.ndarray, passes: int = 5) -> np.ndarray:
    """Coordinate ascent on per-class margin offsets, accuracy objective.

    A candidate `d` is an INCREMENT on top of the current delta_c (grid d=0
    keeps it), and the update accumulates: `delta[c] += best_d`. Storing the
    grid value absolutely was a bug that silently wrecked pass >= 2.
    Monotonic by construction; the final assert is a regression guard.
    """
    n_classes = margins.shape[1]
    sd = float(margins.std())
    grid = np.sort(np.append(np.arange(-2.0 * sd, 2.0 * sd + 1e-9, sd / 20.0), 0.0))
    delta = np.zeros(n_classes)
    base_acc = float((margins.argmax(axis=1) == y_idx).mean())
    for _ in range(passes):
        improved = False
        for c in range(n_classes):
            best_acc, best_shift = -1.0, 0.0
            for d in grid:
                m = margins + delta
                m[:, c] += d
                acc = float((m.argmax(axis=1) == y_idx).mean())
                if acc > best_acc:
                    best_acc, best_shift = acc, d
            delta[c] += best_shift
            if best_shift != 0.0:
                improved = True
        if not improved:
            break
    final_acc = float(((margins + delta).argmax(axis=1) == y_idx).mean())
    assert final_acc >= base_acc - 1e-9, "calibration must never decrease accuracy"
    return delta


@AnalyticsModel.register("svm_tfidf_l3")
def svm_tfidf_l3(self: AnalyticsModel) -> tuple[DataFrame, DataFrame, DataFrame]:
    """L3: L2 mechanism at alpha=0.9 plus validation-fitted margin calibration.

    Layers over svm_tfidf_l2 (see module header for provenance):
    - class weights w = (n / (2 * n_class)) ** 0.9
    - decision rule argmax_c(margin_c + delta_c), delta fitted on valid
    """
    global_mean = self.train_embedding.agg(F.avg("rating").alias("gm")).collect()[0]["gm"]
    drug_enc = (
        self.train_embedding.groupBy("drugName")
        .agg(F.sum("rating").alias("sum_rating"), F.count("*").alias("n"))
        .withColumn(
            "drug_target",
            (F.col("sum_rating") + F.lit(SMOOTHING) * F.lit(global_mean))
            / (F.col("n") + F.lit(SMOOTHING)),
        )
        .select("drugName", "drug_target")
    )
    assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")

    # Layer 3: private vocabulary, fitted on the training split only, then
    # swapped in for the shared 20k `embedding` column on every split.
    def _joined(split):
        return split.join(F.broadcast(drug_enc), "drugName", "left").fillna(
            {"drug_target": global_mean}
        )

    vectorizer = Pipeline(stages=[
        CountVectorizer(inputCol="words_featured", outputCol="_cv_counts",
                        vocabSize=VOCAB_SIZE, minDF=MIN_DF),
        IDF(inputCol="_cv_counts", outputCol="embedding_l3"),
    ]).fit(_joined(self.train_embedding))

    def _prep(split):
        revectorized = (
            vectorizer.transform(_joined(split))
            .drop("embedding", "_cv_counts")
            .withColumnRenamed("embedding_l3", "embedding")
        )
        return assembler.transform(revectorized).cache()

    train_df = _prep(self.train_embedding)
    valid_df = _prep(self.valid_embedding)
    test_df = _prep(self.test_embedding)
    n = train_df.count(); valid_df.count(); test_df.count()

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
            F.when(F.col("bin_label") == 1, F.lit((n / (2.0 * n_pos)) ** ALPHA))
             .otherwise(F.lit((n / (2.0 * n_neg)) ** ALPHA)),
        )
        return LinearSVC(
            featuresCol="features",
            labelCol="bin_label",
            weightCol="bin_weight",
            maxIter=MAX_ITER,
            regParam=REG_PARAM,
            standardization=True,
        ).fit(binary)

    with ThreadPoolExecutor(max_workers=6) as pool:
        models = list(pool.map(_fit_one, classes))

    def _margins(frame: DataFrame) -> np.ndarray:
        """(N, C) matrix of rawPrediction[1] margins as plain doubles.

        `vector_to_array(col)[1]` extracts the margin as a double column
        (JVM-native); VectorUDT columns reject direct `[1]` subscripting and
        toPandas() on them would yield DenseVector objects that break numpy.
        """
        scored = frame
        for c, model in zip(classes, models):
            rp, pc, sc = f"rp_{c}", f"p_{c}", f"s_{c}"
            scored = model.setRawPredictionCol(rp).setPredictionCol(pc).transform(scored)
            scored = scored.withColumn(sc, vector_to_array(F.col(rp))[1]).drop(rp, pc)
        return scored.select(*[f"s_{c}" for c in classes]).toPandas().to_numpy(dtype=float)

    # Calibration split: margins on valid, delta by coordinate ascent.
    valid_margins = _margins(valid_df)
    class_index = {c: i for i, c in enumerate(classes)}
    y_idx = valid_df.select("rating_p1").toPandas()["rating_p1"].map(class_index).to_numpy()
    pre_acc = float((valid_margins.argmax(axis=1) == y_idx).mean())
    delta = _fit_deltas(valid_margins, y_idx)
    post_acc = float(((valid_margins + delta).argmax(axis=1) == y_idx).mean())
    print(
        f"svm_tfidf_l3 calibration: pre-acc={pre_acc:.4f} post-acc={post_acc:.4f} "
        f"(+{post_acc - pre_acc:.4f}), delta={['%.2f' % d for d in delta]}"
    )

    def transform_with_scores(frame: DataFrame) -> DataFrame:
        scored = frame
        for c, model in zip(classes, models):
            rp, pc, sc = f"rp_{c}", f"p_{c}", f"s_{c}"
            scored = model.setRawPredictionCol(rp).setPredictionCol(pc).transform(scored)
            scored = scored.withColumn(sc, vector_to_array(F.col(rp))[1]).drop(rp, pc)
        # Double columns, so adding the per-class delta is plain arithmetic
        # (vector columns do not support `+`).
        score_cols = []
        for c, d in zip(classes, delta):
            sc = f"s_{c}"
            scored = scored.withColumn(sc, F.col(sc) + F.lit(float(d)))
            score_cols.append(sc)
        best = F.greatest(*score_cols)
        pred_col = None
        for c in classes:
            cond = F.col(f"s_{c}") == best
            pred_col = (
                F.when(cond, F.lit(c)) if pred_col is None else pred_col.when(cond, F.lit(c))
            )
        return scored.withColumn("prediction", pred_col.cast("double"))

    columns = ["uniqueID", "rating", "prediction"]
    return (
        transform_with_scores(train_df).select(*columns),
        transform_with_scores(valid_df).select(*columns),
        transform_with_scores(test_df).select(*columns),
    )
