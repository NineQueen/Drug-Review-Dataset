"""Train-only Word2Vec with saved configuration and mean/max review vectors."""
import json
import math
from pathlib import Path
import re

from pyspark.ml.feature import Word2Vec, Word2VecModel
from src.embeddings.base import EmbeddingModel
from src.paths import EMBEDDING_MODELS_DIR

DEFAULT_CONFIG = {
    "vector_size": 300, "window_size": 5, "min_count": 5, "max_iter": 10,
    "step_size": 0.025, "seed": 42, "num_partitions": 1,
    "max_sentence_length": 1000, "pooling": "meanmax",
}


def _configuration(overrides=None):
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError("config must be a dictionary or None")
    unknown = set(overrides or {}) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"Unknown Word2Vec config keys: {sorted(unknown)}")
    config = {**DEFAULT_CONFIG, **(overrides or {})}
    for key in ("vector_size", "window_size", "max_iter", "num_partitions", "max_sentence_length", "min_count"):
        minimum = 0 if key == "min_count" else 1
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if type(config["seed"]) is not int or not -(2**63) <= config["seed"] < 2**63:
        raise ValueError("seed must be a signed 64-bit integer")
    rate = config["step_size"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("step_size must be finite and positive")
    if config["pooling"] not in ("p01", "meanmax"):
        raise ValueError("pooling must be p01 or meanmax")
    return config


def _model_path(model_name):
    if not isinstance(model_name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", model_name) or "__" in model_name:
        raise ValueError("model_name must be lowercase with single underscores, without path separators")
    return EMBEDDING_MODELS_DIR / model_name


def _save_configuration(path, config):
    dimension = config["vector_size"] * (2 if config["pooling"] == "meanmax" else 1)
    record = {"format_version": 1, "config": config, "output_dimension": dimension,
              "input_column": "words", "training_corpus": "self.train only"}
    (path / "embedding_config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    # Preserve Hadoop CRCs and metadata bytes when committing a fitted model.
    (path / ".gitattributes").write_text("** -text\n", encoding="utf-8", newline="\n")


@EmbeddingModel.train_register("word2vec")
def word2vec_train(self, model_name="word2vec", *, config=None):
    """Always fit afresh on complete train.words; unspecified keys use defaults."""
    actual = _configuration(config)
    path = _model_path(model_name)
    estimator = Word2Vec(
        inputCol="words", outputCol="embedding", vectorSize=actual["vector_size"],
        windowSize=actual["window_size"], minCount=actual["min_count"],
        maxIter=actual["max_iter"], stepSize=actual["step_size"], seed=actual["seed"],
        numPartitions=actual["num_partitions"], maxSentenceLength=actual["max_sentence_length"],
    )
    model = estimator.fit(self.train.select("words"))
    path.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(path))
    _save_configuration(path, actual)


def _meanmax(tokens, vectors, dimension):
    import numpy as np
    tokens = tokens or []
    known = [vectors[word] for word in tokens if word in vectors]
    if not known:
        return np.zeros(2 * dimension)
    matrix = np.asarray(known, dtype=np.float64)
    return np.concatenate((matrix.sum(axis=0) / len(tokens), matrix.max(axis=0)))


@EmbeddingModel.predict_register("word2vec")
def word2vec_predict(self, model_name="word2vec"):
    """Read this model's persisted config; registry wrapper narrows output columns."""
    path = _model_path(model_name)
    saved = json.loads((path / "embedding_config.json").read_text(encoding="utf-8"))
    if saved["format_version"] != 1 or saved["input_column"] != "words":
        raise ValueError("Unsupported Word2Vec embedding config")
    actual = _configuration(saved["config"])
    model = Word2VecModel.load(str(path))
    d = model.getVectorSize()
    dimension = d * (2 if actual["pooling"] == "meanmax" else 1)
    if d != actual["vector_size"] or dimension != saved["output_dimension"]:
        raise ValueError("Saved word model and embedding config dimensions disagree")
    frames = (self.train, self.valid, self.test)
    if actual["pooling"] == "p01":
        return tuple(model.transform(frame) for frame in frames)
    import numpy as np
    from pyspark.ml.linalg import VectorUDT, Vectors
    from pyspark.sql import functions as F
    vectors = {row.word: np.asarray(row.vector, dtype=np.float64) for row in model.getVectors().collect()}
    shared = self.train.sparkSession.sparkContext.broadcast(vectors)

    @F.udf(VectorUDT(), useArrow=False)
    def pool(tokens):
        return Vectors.dense(_meanmax(tokens, shared.value, d))

    return tuple(frame.withColumn("embedding", pool("words").alias("embedding", metadata={"ml_attr": {"num_attrs": dimension}})) for frame in frames)
