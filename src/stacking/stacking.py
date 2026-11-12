import numpy as np
from pyspark.ml.functions import array_to_vector
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StructField, StructType

from src import paths
from src.analytics import AnalyticsModel
from src.dataOI import DataOI
from src.embeddings import EmbeddingModel
from src.encoder import encoder
from src.k_flod import KFold
from src.seed import SEED, seed
from src.split import weight
from src.text_features import fit_phrases


K = 3
NUM_CLASSES = 10
# Hash levels: seed(0) cut this pool, seed(1) makes the folds, seed(2) is MLP's early stop.
_ORDINAL_EPOCHS = 300
_ORDINAL_LR = 0.01
_ORDINAL_L2 = 1e-4


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
        """Fit the ordinal logistic regression on the out-of-fold scores."""
        pair_names = self.data_oi.load_stacking_manifest()
        features = _feature_frame(self.data_oi.load_stacking_oof(pair_names), pair_names).cache()
        features.count()
        try:
            matrix, labels = _matrix_from_frame(features)
            fitted = _fit_ordinal(matrix, labels)
            self.data_oi.save_stacking_ordinal({"pairs": pair_names, **fitted})
            print(
                f"Ordinal logistic regression on {len(pair_names)} models, "
                f"loss {fitted['train_loss']:.6f}"
            )
        finally:
            self.data_oi.unpersist(features)

    def predict(self) -> tuple[DataFrame, DataFrame, DataFrame]:
        """Score the saved ordinal logistic regression on the full-data base predictions."""
        pair_names = self.data_oi.load_stacking_manifest()
        ordinal = self.data_oi.load_stacking_ordinal()
        train_frames, valid_frames, test_frames = self.data_oi.load_prediction_data(pair_names)
        cached = tuple(
            _feature_frame(frames, pair_names).cache()
            for frames in (train_frames, valid_frames, test_frames)
        )
        for frame in cached:
            frame.count()
        return tuple(_ordinal_frame(ordinal, frame) for frame in cached)

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


def _matrix_from_frame(frame: DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows = list(frame.select("rating_p1", "features").toLocalIterator())
    if not rows:
        raise ValueError("out-of-fold scores have no rows")
    labels = np.asarray([int(row.rating_p1) for row in rows], dtype=np.int64)
    matrix = np.stack([np.asarray(row.features.toArray(), dtype=np.float64) for row in rows])
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError("rating_p1 must be an integer from 0 to 9")
    return matrix, labels


def _fit_ordinal(features: np.ndarray, labels: np.ndarray) -> dict:
    """Ordinal logistic regression: one linear score and increasing thresholds."""
    import torch

    torch.manual_seed(SEED)
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    scaled = (features - mean) / scale
    inputs = torch.tensor(scaled, dtype=torch.float32)
    target_labels = torch.tensor(labels, dtype=torch.long)
    levels = torch.arange(NUM_CLASSES - 1)
    target = (target_labels.unsqueeze(1) > levels).to(torch.float32)
    weight = torch.zeros(inputs.shape[1], dtype=torch.float32, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float32, requires_grad=True)
    gap = float(np.log(np.exp(0.5) - 1.0))
    raw = torch.tensor(
        [-2.0, *([gap] * (NUM_CLASSES - 2))],
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam((weight, bias, raw), lr=_ORDINAL_LR)
    loss_value = None
    for _ in range(_ORDINAL_EPOCHS):
        thresholds = _threshold_tensor(raw)
        score = inputs @ weight + bias
        logits = score.unsqueeze(1) - thresholds.unsqueeze(0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        loss = loss + _ORDINAL_L2 * weight.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    thresholds = _threshold_tensor(raw).detach().numpy()
    return {
        "mean": mean.astype(np.float64).tolist(),
        "scale": scale.astype(np.float64).tolist(),
        "weight": weight.detach().numpy().astype(np.float64).tolist(),
        "bias": float(bias.detach()),
        "thresholds": thresholds.astype(np.float64).tolist(),
        "train_loss": loss_value,
        "epochs": _ORDINAL_EPOCHS,
    }


def _threshold_tensor(raw):
    import torch

    gaps = torch.nn.functional.softplus(raw[1:])
    return torch.cumsum(torch.cat((raw[:1], gaps)), dim=0)


def _ordinal_scores(features: np.ndarray, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(spec["mean"], dtype=np.float64)
    scale = np.asarray(spec["scale"], dtype=np.float64)
    weight = np.asarray(spec["weight"], dtype=np.float64)
    thresholds = np.asarray(spec["thresholds"], dtype=np.float64)
    if features.shape[1] != weight.shape[0] or thresholds.shape != (NUM_CLASSES - 1,):
        raise ValueError("saved ordinal model does not match these scores")
    if np.any(np.diff(thresholds) < -1e-5):
        raise ValueError("ordinal thresholds must not decrease")
    score = (features - mean) / scale @ weight + float(spec["bias"])
    greater = 1.0 / (1.0 + np.exp(-(score[:, None] - thresholds)))
    probability = np.empty((len(features), NUM_CLASSES), dtype=np.float64)
    probability[:, 0] = 1.0 - greater[:, 0]
    probability[:, 1:-1] = greater[:, :-1] - greater[:, 1:]
    probability[:, -1] = greater[:, -1]
    return probability.argmax(axis=1).astype(np.float64), probability


def _ordinal_frame(spec: dict, frame: DataFrame) -> DataFrame:
    rows = list(frame.select("uniqueID", "rating", "features").toLocalIterator())
    if not rows:
        raise ValueError("base scores have no rows")
    matrix = np.stack([np.asarray(row.features.toArray(), dtype=np.float64) for row in rows])
    prediction, probability = _ordinal_scores(matrix, spec)
    schema = StructType([
        frame.schema["uniqueID"],
        frame.schema["rating"],
        StructField("prediction", DoubleType(), False),
        StructField("probability", ArrayType(DoubleType(), False), False),
    ])
    records = [
        (row.uniqueID, row.rating, float(label), [float(value) for value in vector])
        for row, label, vector in zip(rows, prediction, probability)
    ]
    return frame.sparkSession.createDataFrame(records, schema)
