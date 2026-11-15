from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src import paths
from src.analytics import AnalyticsModel
from src.dataOI import DataOI
from src.embeddings import EmbeddingModel
from src.encoder import encoder
from src.k_flod import KFold
from src.seed import seed
from src.split import weight
from src.text_features import fit_phrases


K = 3
NUM_CLASSES = 10
WEIGHT_EXPONENT = 0.75
PRIOR_EXPONENT = 0.0
DEFAULT_PARA = {
    "regParam": 0.3,
    "maxIter": 100,
    "elasticNetParam": 0.0,
    "tol": 1e-7,
    "weight_exponent": WEIGHT_EXPONENT,
    "prior_exponent": PRIOR_EXPONENT,
}
# Improved TF-IDF logistic regression stays outside both registries.
# Its saved full-data predictions are `lr_improved__tfidf_improved`.
IMPROVED_EMBEDDING = "tfidf_improved"
IMPROVED_ANALYTICS = "lr_improved"
# Hash levels: seed(0) cut this pool, seed(1) makes the folds, seed(2) is MLP's early stop.


class StackingModel:

    def __init__(self, fold: int = K) -> None:
        if fold < 2:
            raise ValueError("stacking needs at least two folds")
        self.data_oi = DataOI()
        self.fold = fold

    def load_data(self) -> None:
        """Build one out-of-fold score file for every registered pair and improved baseline."""
        pairs = _registered_pairs()
        df_train_raw = self.data_oi.load_raw_train_data()
        raw_released = False
        try:
            row_count = df_train_raw.count()
            if row_count == 0:
                raise ValueError("train_raw has no rows")

            def prepare(fit, holdout):
                fit = weight(fit)
                fit, holdout = encoder(fit, holdout)
                return fit, holdout

            kfold = KFold("stacking", df_train_raw, self.fold, prepare, seed(1))
            self.data_oi.unpersist(df_train_raw)
            raw_released = True
            try:
                for fold_index in range(self.fold):
                    print(f"Fold {fold_index} running")
                    df_fit, df_holdout = kfold.next_fold()
                    # Analytics needs three frames. This row is not a test set.
                    df_empty = df_holdout.limit(1)
                    for embedding_name in sorted(EmbeddingModel.predict_registry):
                        phrases = fit_phrases(df_fit) if embedding_name == "glove" else None
                        self._run_embedding(
                            fold_index,
                            embedding_name,
                            df_fit,
                            df_holdout,
                            df_empty,
                            phrases,
                        )
                        del phrases
                    self._run_improved_baseline(fold_index, df_fit, df_holdout, df_empty)
                self._combine_oof(pairs, row_count)
            finally:
                kfold.reset()
        finally:
            if not raw_released:
                self.data_oi.unpersist(df_train_raw)

    def train(self, para: dict | None = None) -> None:
        """Fit weighted multinomial logistic regression on the out-of-fold scores.

        Passed keys override `DEFAULT_PARA`. The incoming dict is not changed.
        """
        params = {**DEFAULT_PARA, **(para or {})}
        pair_names = self.data_oi.load_stacking_manifest()
        with self.data_oi.scope():
            features = _feature_frame(
                self.data_oi.load_stacking_oof(pair_names), pair_names
            ).cache()
            features.count()
            pipeline, meta = _fit_weighted_lr(features, pair_names, params)
            self.data_oi.save_stacking_model(pipeline, meta)
        print(
            f"Weighted multinomial logistic regression on {len(pair_names)} models, "
            f"{meta['input_dim']} features"
        )

    def predict(self) -> tuple[DataFrame, DataFrame, DataFrame]:
        """Score the saved pipeline on the full-data base predictions."""
        pair_names = self.data_oi.load_stacking_manifest()
        pipeline = self.data_oi.load_stacking_model()
        train_frames, valid_frames, test_frames = self.data_oi.load_prediction_data(pair_names)
        return (
            _score_frame(pipeline, _feature_frame(train_frames, pair_names)),
            _score_frame(pipeline, _feature_frame(valid_frames, pair_names)),
            _score_frame(pipeline, _feature_frame(test_frames, pair_names)),
        )

    def _run_embedding(
        self,
        fold_index: int,
        embedding_name: str,
        df_fit: DataFrame,
        df_holdout: DataFrame,
        df_empty: DataFrame,
        phrases,
    ) -> None:
        print(f"Fold {fold_index} embedding {embedding_name}")
        model_name = f"stacking_f{fold_index}_{embedding_name}"
        model_dir = paths.STACKING_EMBEDDING_DIR
        self.data_oi.remove_stacking_embedding(model_name)
        try:
            with self.data_oi.scope():
                embedding = EmbeddingModel(df_fit, df_holdout, df_empty, phrases)
                EmbeddingModel.train_registry[embedding_name](
                    embedding, model_name, model_dir=model_dir
                )
                predicted = EmbeddingModel.predict_registry[embedding_name](
                    embedding, model_name, model_dir=model_dir
                )
                cached = tuple(frame.cache() for frame in predicted)
                for frame in cached:
                    frame.count()
                for analytics_name in sorted(AnalyticsModel.registry):
                    self._write_holdout(
                        fold_index, embedding_name, analytics_name, cached
                    )
                del embedding, cached, predicted
        finally:
            self.data_oi.remove_stacking_embedding(model_name)

    def _run_improved_baseline(
        self,
        fold_index: int,
        df_fit: DataFrame,
        df_holdout: DataFrame,
        df_empty: DataFrame,
    ) -> None:
        pair_name = f"{IMPROVED_ANALYTICS}__{IMPROVED_EMBEDDING}"
        print(f"Fold {fold_index} {pair_name}")
        model_name = f"stacking_f{fold_index}_{IMPROVED_EMBEDDING}"
        model_dir = paths.STACKING_EMBEDDING_DIR
        self.data_oi.remove_stacking_embedding(model_name)
        try:
            with self.data_oi.scope():
                embedding = EmbeddingModel(df_fit, df_holdout, df_empty)
                EmbeddingModel.baseline_train(
                    embedding,
                    model_name,
                    model_dir=model_dir,
                )
                predicted = EmbeddingModel.baseline_predict(
                    embedding, model_name, model_dir=model_dir
                )
                cached = tuple(frame.cache() for frame in predicted)
                for frame in cached:
                    frame.count()
                # The notebook stores this para as baseline_improved_baseline.json.
                para = self.data_oi.load_analytics_para("baseline", "improved_baseline")
                model = AnalyticsModel(IMPROVED_EMBEDDING, *cached)
                train_pred, holdout_pred, test_pred = model.improved_baseline(para)
                self.data_oi.save_stacking_fold(pair_name, fold_index, holdout_pred)
                del train_pred, test_pred, holdout_pred
                model.release_features()
                del model, cached, predicted, embedding
        finally:
            self.data_oi.remove_stacking_embedding(model_name)

    def _write_holdout(
        self,
        fold_index: int,
        embedding_name: str,
        analytics_name: str,
        embeddings: tuple[DataFrame, DataFrame, DataFrame],
    ) -> None:
        pair_name = f"{analytics_name}__{embedding_name}"
        print(f"Fold {fold_index} {pair_name}")
        with self.data_oi.scope():
            model = AnalyticsModel(embedding_name, *embeddings)
            para = self.data_oi.load_analytics_para(embedding_name, analytics_name)
            train_pred, holdout_pred, test_pred = AnalyticsModel.registry[analytics_name](model, para)
            self.data_oi.save_stacking_fold(pair_name, fold_index, holdout_pred)
            del train_pred, test_pred, holdout_pred
            model.release_features()
            del model

    def _combine_oof(self, pairs: list[tuple[str, str]], row_count: int) -> None:
        pair_names = []
        for embedding_name, analytics_name in pairs:
            pair_name = f"{analytics_name}__{embedding_name}"
            parts = self.data_oi.load_stacking_folds(pair_name, self.fold)
            combined = parts[0]
            for part in parts[1:]:
                combined = combined.unionByName(part)
            self.data_oi.save_stacking_oof(pair_name, combined)
            pair_names.append(pair_name)
        saved = self.data_oi.load_stacking_oof(pair_names)
        first = saved[pair_names[0]]
        saved_count = first.count()
        distinct_count = first.select("uniqueID").distinct().count()
        if saved_count != row_count or distinct_count != row_count:
            raise ValueError(
                f"OOF has {saved_count} rows and {distinct_count} ids; "
                f"train_raw has {row_count}"
            )
        for pair_name in pair_names[1:]:
            other_count = saved[pair_name].count()
            if other_count != row_count:
                raise ValueError(f"{pair_name} has {other_count} rows; train_raw has {row_count}")
        self.data_oi.save_stacking_manifest(pair_names)
        self.data_oi.remove_stacking_folds()
        print(f"Saved OOF for {len(pair_names)} models, {row_count} rows")


def _registered_pairs() -> list[tuple[str, str]]:
    pairs = [
        (embedding_name, analytics_name)
        for embedding_name in sorted(EmbeddingModel.predict_registry)
        for analytics_name in sorted(AnalyticsModel.registry)
    ]
    pairs.append((IMPROVED_EMBEDDING, IMPROVED_ANALYTICS))
    return pairs


def _feature_frame(frames: dict[str, DataFrame], pair_names: list[str]) -> DataFrame:
    """Join base scores on `uniqueID` and `rating`, then concatenate their probabilities."""
    if not pair_names:
        raise ValueError("no out-of-fold models")
    joined = frames[pair_names[0]].select("uniqueID", "rating", F.col("probability").alias("p0"))
    pieces = [F.col("p0")]
    for index, name in enumerate(pair_names[1:], start=1):
        alias = f"p{index}"
        part = frames[name].select("uniqueID", "rating", F.col("probability").alias(alias))
        joined = joined.join(part, ["uniqueID", "rating"], "inner")
        pieces.append(F.col(alias))
    expected = frames[pair_names[0]].count()
    rows = joined.count()
    if rows == 0 or rows != expected:
        raise ValueError("base scores are not aligned on uniqueID and rating")
    return joined.select(
        "uniqueID",
        "rating",
        (F.col("rating") - F.lit(1)).alias("rating_p1"),
        array_to_vector(F.concat(*pieces)).alias("features"),
    )


def _training_summary(frame: DataFrame) -> tuple[list[int], int]:
    """Return class counts in order 0..9 and the shared feature width."""
    def width():
        return F.size(vector_to_array("features"))

    columns = [
        F.count("*").alias("rows"),
        F.min(width()).alias("min_width"),
        F.max(width()).alias("max_width"),
        F.sum(F.when(width().isNull(), F.lit(1)).otherwise(F.lit(0))).alias("null_width"),
    ]
    for label in range(NUM_CLASSES):
        matched = F.when(F.col("rating_p1") == F.lit(label), F.lit(1)).otherwise(F.lit(0))
        columns.append(F.sum(matched).alias(f"c{label}"))
    row = frame.agg(*columns).collect()[0]
    counts = [int(row[f"c{label}"] or 0) for label in range(NUM_CLASSES)]
    total = int(row["rows"])
    covered = sum(counts)
    if covered != total:
        raise ValueError(
            "training OOF rating_p1 must be an integer from 0 to "
            f"{NUM_CLASSES - 1}; {total - covered} rows are outside that range"
        )
    missing = [label for label, count in enumerate(counts) if count == 0]
    if missing:
        raise ValueError(
            f"training OOF is missing classes {missing}; priors and rows are not filled in"
        )
    if row["min_width"] is None or int(row["null_width"] or 0) != 0:
        raise ValueError("stacked features are missing")
    min_width = int(row["min_width"])
    max_width = int(row["max_width"])
    if min_width != max_width:
        raise ValueError(f"stacked feature width is not constant: {min_width} to {max_width}")
    return counts, min_width


def _meta_weights(counts: list[int], exponent: float) -> list[float]:
    """Class weights from n_c ** -exponent, scaled so the sample weights sum to N."""
    total = float(sum(counts))
    raw = [count ** (-exponent) for count in counts]
    normalizer = total / sum(count * raw_weight for count, raw_weight in zip(counts, raw))
    return [raw_weight * normalizer for raw_weight in raw]


def _weight_column(weights: list[float]):
    entries = []
    for label, value in enumerate(weights):
        entries.extend((F.lit(label), F.lit(float(value))))
    return F.create_map(*entries)[F.col("rating_p1").cast("int")]


def _check_weight_sum(frame: DataFrame, total: int) -> None:
    row = frame.agg(
        F.count("*").alias("rows"),
        F.sum("meta_weight").alias("weight_sum"),
    ).collect()[0]
    rows = int(row["rows"])
    weight_sum = float(row["weight_sum"])
    if rows != total or abs(weight_sum - total) > 1e-5 * total + 1e-4:
        raise ValueError(f"meta weights sum to {weight_sum} over {rows} rows; expected {total}")


def _fit_weighted_lr(
    features: DataFrame, pair_names: list[str], para: dict
) -> tuple[PipelineModel, dict]:
    params = dict(para)
    weight_exponent = float(params.pop("weight_exponent"))
    prior_exponent = float(params.pop("prior_exponent"))
    for key in ("weightCol", "featuresCol", "labelCol", "family", "standardization"):
        params.pop(key, None)
    counts, width = _training_summary(features)
    expected = len(pair_names) * NUM_CLASSES
    if width != expected:
        raise ValueError(
            f"stacked features have dimension {width}; "
            f"{len(pair_names)} models need {expected}"
        )
    total = sum(counts)
    weights = _meta_weights(counts, weight_exponent)
    priors = [count / total for count in counts]
    weighted = features.withColumn("meta_weight", _weight_column(weights))
    _check_weight_sum(weighted, total)
    scaler_model = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True,
    ).fit(weighted)
    lr_model = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="rating_p1",
        weightCol="meta_weight",
        family="multinomial",
        standardization=False,
        **params,
    ).fit(scaler_model.transform(weighted))
    # thresholds[c] = prior_c ** prior_exponent, listed from class 0 through class 9.
    lr_model = lr_model.setThresholds([float(prior ** prior_exponent) for prior in priors])
    if int(lr_model.numFeatures) != expected or int(lr_model.numClasses) != NUM_CLASSES:
        raise ValueError(
            f"fitted logistic regression has {lr_model.numFeatures} features and "
            f"{lr_model.numClasses} classes"
        )
    pipeline = PipelineModel(stages=[scaler_model, lr_model])
    meta = {
        "pairs": list(pair_names),
        "num_classes": NUM_CLASSES,
        "input_dim": width,
        "weight_exponent": weight_exponent,
        "prior_exponent": prior_exponent,
        "priors": priors,
    }
    return pipeline, meta


def _score_frame(pipeline: PipelineModel, frame: DataFrame) -> DataFrame:
    scored = pipeline.transform(frame)
    return scored.select(
        "uniqueID",
        "rating",
        F.col("prediction").cast("double").alias("prediction"),
        vector_to_array("probability").alias("probability"),
    )
