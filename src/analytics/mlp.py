from itertools import islice

import numpy as np
from pyspark.ml.feature import StandardScaler
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType, StructField, StructType

from src.analytics.base import AnalyticsModel


FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_vec"]
NUM_CLASSES = 10
HIDDEN_LAYERS = (64, 32)
USE_WEIGHTS = True

# D / 30 is the initial default, pending the 12 validation experiments.
DEFAULT_PARA = {
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.001,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 0.0,
    "seed": 42,
    "weight_col": "weight",
}
PARA_RANGE = {}


class MLPTrainer:
    """Share minibatch training between the registered model and experiments."""

    def __init__(
        self, train: DataFrame, hidden_layers: tuple[int, ...], weighted: bool,
        *, epochs: int, batch_size: int,
        lr: float, betas: tuple[float, float], eps: float,
        weight_decay: float, seed: int, weight_col: str,
    ) -> None:
        import torch
        from torch import nn

        self.epochs = epochs
        self.batch_size = batch_size
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
    self: AnalyticsModel, para: dict = DEFAULT_PARA
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Fit a fixed epoch budget on train and return the three prediction frames."""
    params = {**DEFAULT_PARA, **para}
    train, valid, test = self._feature_transform(FEATURES)
    trainer = MLPTrainer(train, HIDDEN_LAYERS, USE_WEIGHTS, **params)
    for _ in range(trainer.epochs):
        trainer.train_epoch()
    return trainer.predict(train), trainer.predict(valid), trainer.predict(test)
