from pyspark.sql import DataFrame
from src.embeddings.base import EmbeddingModel
from pyspark.ml.linalg import VectorUDT, Vectors
import numpy as np
from gensim.models import KeyedVectors
from pyspark.sql import functions as F
from src.paths import EMBEDDING_MODELS_DIR

glove_para = {
    "vector_size": 200,
    "window_size": 5,
    "min_count": 5,
    "iter": 50,
    "initial_learning_rate": 0.05,
    "alpha": 0.75,
    "symmetric": True,
    "distance_weighting": True,
    "memory": 1.0,
}


@EmbeddingModel.train_register("glove")
def glove_train(self:EmbeddingModel, model_name:str = "glove") -> None:
    from glovpy import GloVe
    tokens = self.train_expanded_phrase.select("words_nostop").toPandas()["words_nostop"]
    reviews = [list(sentence) if sentence is not None else [] for sentence in tokens]
    model = GloVe(**glove_para)
    model.train(reviews)

    model_path = EMBEDDING_MODELS_DIR / model_name
    model_path.mkdir(parents= True, exist_ok=True)
    model.wv.save(str(model_path/"glove.kv"))

@EmbeddingModel.predict_register("glove")
def glove_predict(self: EmbeddingModel, model_name:str = "glove") -> tuple[DataFrame, DataFrame, DataFrame]:
    keyed = KeyedVectors.load(str(EMBEDDING_MODELS_DIR / model_name / "glove.kv"))
    token_vectors = {word:keyed.get_vector(word) for word in keyed.index_to_key}
    broadcast_vectors = self.train.sparkSession.sparkContext.broadcast(token_vectors)
    dimension = int(keyed.vector_size)

    @F.udf(VectorUDT())
    def review_vector(tokens):
        known = broadcast_vectors.value
        vectors = [known.get(token) for token in tokens or []]
        if not vectors:
            return Vectors.dense(np.zeros(dimension * 3, np.float64).tolist())
        bounds = [round(i * len(vectors) / 3) for i in range(4)]
        parts = []
        for start, end in zip(bounds, bounds[1:]):
            rows = [vector for vector in vectors[start:end] if vector is not None]
            if rows:
                parts.append(np.mean(rows, axis=0))
            else:
                parts.append(np.zeros(dimension, np.float64))
        return Vectors.dense(np.concatenate(parts).tolist())
    
    def transform(df: DataFrame) -> DataFrame:
        return df.withColumn("embedding", review_vector(F.col("words_nostop")))
    
    return (
        transform(self.train_phrase),
        transform(self.valid_phrase),
        transform(self.test_phrase),
    )