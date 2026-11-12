from copy import deepcopy
from itertools import islice

import numpy as np
from pyspark.ml.feature import StandardScaler
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, StructField, StructType

from src.analytics.base import AnalyticsModel


FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_vec"]
NUM_CLASSES = 10

DEFAULT_PARA = {
    "hidden_layers": (64, 32),
    "weighted": True,
    "epochs": 50,
    "patience": 5,
    "min_delta": 0.0005,
    "batch_size": 128,
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "seed": 42,
    "weight_col": "weight",
}
PARA_RANGE = {
    "hidden_layers": [(64,), (64, 32)],
    "weighted": [False, True],
}


class MLPTrainer:
    """Train minibatches and restore the best validation checkpoint."""

    def __init__(
        self, train: DataFrame, hidden_layers: tuple[int, ...], weighted: bool,
        *, epochs: int, batch_size: int,
        patience: int, min_delta: float,
        lr: float, betas: tuple[float, float], eps: float,
        weight_decay: float, seed: int, weight_col: str,
    ) -> None:
        import torch
        from torch import nn

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_delta = min_delta
        self.rng = np.random.default_rng(seed)
        scaler = StandardScaler(
            inputCol="features", outputCol="scaled_features", withMean=True, withStd=True
        ).fit(train)
        self.mean = scaler.mean.toArray().astype(np.float32)
        self.scale = scaler.std.toArray().astype(np.float32)
        self.scale[self.scale == 0] = 1
        columns = ["features", "rating_p1"]
        if weighted:
            columns.append(weight_col)
        self.rows = list(train.orderBy("uniqueID").select(*columns).toLocalIterator())
        self.targets = torch.tensor([row.rating_p1 for row in self.rows], dtype=torch.long)
        self.weights = torch.tensor(
            [row[weight_col] for row in self.rows] if weighted else np.ones(len(self.rows)),
            dtype=torch.float32,
        )
        layers = []
        width = len(self.mean)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            for hidden in hidden_layers:
                layers.extend([nn.Linear(width, hidden), nn.ReLU()])
                width = hidden
            layers.append(nn.Linear(width, NUM_CLASSES))
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
        )

    def _batch(self, rows):
        import torch

        values = np.stack([row.features.toArray() for row in rows]).astype(np.float32)
        values = (values - self.mean) / self.scale
        return torch.from_numpy(values)

    def train_epoch(self) -> float:
        from torch.nn.functional import cross_entropy

        self.network.train()
        order = self.rng.permutation(len(self.rows))
        total_loss = 0.0
        for start in range(0, len(order), self.batch_size):
            indices = order[start:start + self.batch_size]
            inputs = self._batch([self.rows[index] for index in indices])
            self.optimizer.zero_grad(set_to_none=True)
            losses = cross_entropy(self.network(inputs), self.targets[indices], reduction="none")
            loss = (losses * self.weights[indices]).mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(indices)
        return total_loss / len(self.rows)

    def fit(self, valid: DataFrame) -> None:
        from sklearn.metrics import f1_score

        rows = list(valid.select("rating_p1", "features").toLocalIterator())
        labels = np.asarray([row.rating_p1 for row in rows], dtype=np.int64)
        self.history = []
        best_score = last_improvement = -float("inf")
        stale_epochs = 0
        for epoch in range(1, self.epochs + 1):
            loss = self.train_epoch()
            predictions = self.predict_rows(rows)
            score = float(f1_score(labels, predictions, average="weighted", zero_division=0))
            self.history.append({"epoch": epoch, "train_loss": loss, "valid_weighted_f1": score})
            if score > best_score:
                best_score, self.best_epoch = score, epoch
                best_state = deepcopy(self.network.state_dict())
            if score > last_improvement + self.min_delta:
                last_improvement, stale_epochs = score, 0
            else:
                stale_epochs += 1
            print(f"Epoch {epoch}: loss {loss:.5f}, validation F1 {score:.5f}")
            if stale_epochs >= self.patience:
                break
        self.network.load_state_dict(best_state)
        self.best_valid_weighted_f1 = best_score

    def predict_rows(self, rows) -> np.ndarray:
        import torch

        self.network.eval()
        predictions = []
        iterator = iter(rows)
        with torch.inference_mode():
            while batch := list(islice(iterator, self.batch_size)):
                predictions.extend(self.network(self._batch(batch)).argmax(dim=1).tolist())
        return np.asarray(predictions, dtype=np.int64)

    def predict(self, frame: DataFrame) -> DataFrame:
        rows = frame.select("uniqueID", "rating", "features").toLocalIterator()
        records = []
        while batch := list(islice(rows, self.batch_size)):
            labels = self.predict_rows(batch)
            records.extend(
                (row.uniqueID, row.rating, float(label))
                for row, label in zip(batch, labels)
            )
        schema = StructType([
            frame.schema["uniqueID"], frame.schema["rating"],
            StructField("prediction", DoubleType(), False),
        ])
        return frame.sparkSession.createDataFrame(records, schema)


@AnalyticsModel.register("mlp")
def mlp(
    self: AnalyticsModel, para: dict | None = None
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Use the supplied validation frame for early stopping and return predictions."""
    params = {**DEFAULT_PARA, **(para or {})}
    train, valid, test = self._feature_transform(FEATURES)
    trainer = MLPTrainer(train, **params)
    trainer.fit(valid)
    self.training_info = {
        "best_epoch": trainer.best_epoch,
        "epochs_run": len(trainer.history),
        "best_valid_weighted_f1": trainer.best_valid_weighted_f1,
        "history": trainer.history,
    }
    print(f"MLP best epoch: {trainer.best_epoch}/{len(trainer.history)}, validation F1 {trainer.best_valid_weighted_f1:.5f}")
    return trainer.predict(train), trainer.predict(valid), trainer.predict(test)
