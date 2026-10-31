from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql import DataFrame

from src.embeddings.base import EmbeddingModel
from src.paths import EMBEDDING_MODELS_DIR


@EmbeddingModel.train_register("tfidf")
def TFIDF(self) -> None:
    # Fit on `self.train` only. Downstream models read the column `embedding`.
    pipeline = Pipeline(stages=[
        CountVectorizer(
            inputCol="words_stem",
            outputCol="tf",
            vocabSize=5000,
            minDF=5,
        ),
        IDF(inputCol="tf", outputCol="embedding"),
    ])
    tfidf_model = pipeline.fit(self.train)
    model_path = EMBEDDING_MODELS_DIR / "tfidf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tfidf_model.write().overwrite().save(str(model_path))


@EmbeddingModel.predict_register("tfidf")
def TFIDF_predict(self) -> tuple[DataFrame, DataFrame]:
    tfidf_model = PipelineModel.load(str(EMBEDDING_MODELS_DIR / "tfidf"))
    return tfidf_model.transform(self.train), tfidf_model.transform(self.test)
