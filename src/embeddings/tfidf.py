from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.sql import DataFrame
from pyspark.sql.functions import transform

from src.embeddings.base import EmbeddingModel
from src.paths import EMBEDDING_MODELS_DIR


@EmbeddingModel.baseline_train_register()
def tfidf_train(self: EmbeddingModel, model_name: str = "baseline") -> None:
    # Fit on `self.train` only. Downstream models read the column `embedding`.
    pipeline = Pipeline(stages=[
        CountVectorizer(
            inputCol="words_featured",
            outputCol="tf",
            vocabSize=20000,
            minDF=5,
        ),
        IDF(inputCol="tf", outputCol="embedding"),
    ])
    tfidf_model = pipeline.fit(self.train)
    model_path = EMBEDDING_MODELS_DIR / model_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tfidf_model.write().overwrite().save(str(model_path))


@EmbeddingModel.baseline_predict_register()
def tfidf_predict(self: EmbeddingModel, model_name: str = "baseline") -> tuple[DataFrame, DataFrame, DataFrame]:
    tfidf_model = PipelineModel.load(str(EMBEDDING_MODELS_DIR / model_name))
    
    return (
        tfidf_model.transform(self.train), 
        tfidf_model.transform(self.valid),
        tfidf_model.transform(self.test)
    )
