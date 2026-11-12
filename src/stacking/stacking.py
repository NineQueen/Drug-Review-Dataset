from pyspark.ml.classification import LogisticRegression
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
# Hash levels: seed(0) cut this pool, seed(1) makes the folds, seed(2) is MLP's early stop.
PREDICTION_COLUMNS = ["uniqueID", "rating", "prediction", "probability"]


class StackingModel:

    def __init__(self, fold: int = K) -> None:
        if fold < 2:
            raise ValueError("stacking needs at least two folds")
        self.data_oi = DataOI()
        self.fold = fold

    def load_data(self) -> None:
        """Build one out-of-fold score file for every registered pair."""
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
                self._combine_oof(pairs, row_count)
            finally:
                kfold.reset()
        finally:
            if not raw_released:
                self.data_oi.unpersist(df_train_raw)

    def train(self) -> None:
        pair_names = self.data_oi.load_stacking_manifest()
        features = _feature_frame(self.data_oi.load_stacking_oof(pair_names), pair_names)
        model = LogisticRegression(featuresCol="features", labelCol="rating_p1").fit(features)
        self.data_oi.save_stacking_model(model)
        print(f"Logistic regression on {len(pair_names)} models")

    def predict(self) -> tuple[DataFrame, DataFrame, DataFrame]:
        pair_names = self.data_oi.load_stacking_manifest()
        model = self.data_oi.load_stacking_model()
        train_frames, valid_frames, test_frames = self.data_oi.load_prediction_data(pair_names)
        return _prediction_frames(
            model,
            _feature_frame(train_frames, pair_names),
            _feature_frame(valid_frames, pair_names),
            _feature_frame(test_frames, pair_names),
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
    return [
        (embedding_name, analytics_name)
        for embedding_name in sorted(EmbeddingModel.predict_registry)
        for analytics_name in sorted(AnalyticsModel.registry)
    ]


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


def _prediction_frames(model, *frames: DataFrame) -> tuple[DataFrame, ...]:
    return tuple(
        model.transform(frame)
        .withColumn("probability", vector_to_array("probability"))
        .select(*PREDICTION_COLUMNS)
        for frame in frames
    )
