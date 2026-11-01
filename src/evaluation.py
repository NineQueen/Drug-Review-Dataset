from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType


class Scorer:
    metric_names = [
        "accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "mae",
        "rmse",
    ]

    def __init__(self, test_pred: dict[str, DataFrame]) -> None:
        self.spark = SparkSession.getActiveSession()
        score_schema = StructType([
            StructField("Embedding_Model", StringType(), False),
            StructField("Analytic_Model", StringType(), False),
            *[
                StructField(metric, DoubleType(), False) for metric in self.metric_names
            ],
        ])
        self.score = self.spark.createDataFrame([], score_schema)
        predictions_schema = StructType([
            StructField("Embedding_Model", StringType(), False),
            StructField("Analytic_Model", StringType(), False),
            StructField("uniqueID", IntegerType(), False),
            StructField("rating", IntegerType(), False),
            StructField("rating_pred", DoubleType(), False),
        ])
        self.predictions = self.spark.createDataFrame([], predictions_schema)
        for model_name, df_pred in test_pred.items():
            # Directory names are `{analytics}__{embedding}`; score columns use embedding first.
            analytics_name, embedding_name = model_name.split("__")
            self.add_model(embedding_name, analytics_name, df_pred)

    def _metrics(self, pred: DataFrame) -> dict[str, float]:
        # Precision, recall, and F1 are weighted by class support. Lower MAE and RMSE are better.
        classification = MulticlassClassificationEvaluator(
            labelCol="rating", predictionCol="rating_pred"
        )
        regression = RegressionEvaluator(
            labelCol="rating", predictionCol="rating_pred"
        )
        return {
            "accuracy": classification.evaluate(pred, {classification.metricName: "accuracy"}),
            "weighted_precision": classification.evaluate(
                pred, {classification.metricName: "weightedPrecision"}
            ),
            "weighted_recall": classification.evaluate(
                pred, {classification.metricName: "weightedRecall"}
            ),
            "weighted_f1": classification.evaluate(pred, {classification.metricName: "f1"}),
            "mae": regression.evaluate(pred, {regression.metricName: "mae"}),
            "rmse": regression.evaluate(pred, {regression.metricName: "rmse"}),
        }

    def add_model(self, embedding_name: str, analytics_name: str, df_pred: DataFrame) -> "Scorer":
        # Drop a stored row only when both names match.
        df_pred = df_pred.withColumn(
                "rating_pred",
                # Classifiers emit 0–9. Metrics compare `rating` with `rating_pred`.
                F.col("prediction") + 1,
        )
        self.score = self.score.filter(
            (F.col("Embedding_Model") != embedding_name)
            | (F.col("Analytic_Model") != analytics_name)
        )
        self.predictions = self.predictions.filter(
            (F.col("Embedding_Model") != embedding_name)
            | (F.col("Analytic_Model") != analytics_name)
        )

        pred = df_pred.select(
            F.lit(embedding_name).alias("Embedding_Model"),
            F.lit(analytics_name).alias("Analytic_Model"),
            "uniqueID", "rating", "rating_pred",
        ).cache()
        pred.count()

        metric_values = self._metrics(pred)
        row = self.spark.createDataFrame(
            [(embedding_name, analytics_name, *[metric_values[metric] for metric in self.metric_names])],
            schema=self.score.schema,
        )
        self.score = self.score.unionByName(row)
        self.predictions = self.predictions.unionByName(pred).cache()
        self.predictions.count()
        return self

    def show_metrics(self, **kwargs) -> None:
        self.score.show(**kwargs)
