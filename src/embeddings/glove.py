from pyspark.sql import DataFrame, udf
from src.embeddings.base import EmbeddingModel
from pyspark.ml.linalg import VectorUDT, Vectors
import numpy as np
from pyspark.sql import functions as F
from src.paths import EMBEDDING_MODELS_DIR

glove_para = {
    "vector_size": 100,
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

    reviews = self.train_expanded.toPandas()["words_nostop"].tolist()
    model = GloVe(**glove_para)
    model.train(reviews)

    model_path = EMBEDDING_MODELS_DIR / model_name
    model_path.mkdir(parents= True, exist_ok=True)
    model.wv.save(str(model_path/"glove.kv"))

@EmbeddingModel.predict_register("glove")
def glove_predict(self: EmbeddingModel, model_name:str = "glove") -> tuple[DataFrame, DataFrame, DataFrame]:
    from gensim.models import KeyedVectors

    keyed = KeyedVectors.load(str(EMBEDDING_MODELS_DIR / model_name / "glove.kv"))
    token_vectors = {word:keyed.get_vector(word) for word in keyed.index_to_key}
    broadcast_vectors = self.train.sparkSession.sparkContext.broadcast(token_vectors)
    dimension = int(keyed.vector_size)

    @F.udf(VectorUDT())
    def review_vector(tokens):
        total = np.zeros(dimension,np.float64)
        count = 0
        known = broadcast_vectors.value
        for token in tokens or []:
            vector = known.get(token)
            if vector is not None:
                total += vector
                count += 1
        
        if count:
            total = total / count
        return Vectors.dense(total.tolist())
    
    def transform(df: DataFrame) -> DataFrame:
        return df.withColumn("embedding", review_vector(F.col("words_nostop")))
    
    return transform(self.train), transform(self.valid), transform(self.test)
