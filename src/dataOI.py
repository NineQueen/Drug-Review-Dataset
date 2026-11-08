from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DateType, IntegerType, StringType, StructField, StructType

import src.paths as paths


class DataOI:
    def __init__(self) -> None:
        # The notebook creates the only session. This call does not start one.
        self.spark = SparkSession.getActiveSession()

    def load_raw_data(self) -> DataFrame:
        schema = StructType([
            StructField("uniqueID", IntegerType()),
            StructField("drugName", StringType()),
            StructField("condition", StringType()),
            StructField("review", StringType()),
            StructField("rating", IntegerType()),
            StructField("date", DateType()),
            StructField("usefulCount", IntegerType()),
        ])

        df_raw = self.spark.read.csv(
            str(paths.RAW_CSV),
            schema = schema,
            header=True,
            multiLine=True,
            escape='"',
            dateFormat = "d-MMM-yy",
            locale = "en-US"
        )
        return df_raw
    
    def load_webmd_data(self) -> DataFrame:
        df_raw = self.spark.read.csv(
            str(paths.WEBMD_CSV),
            inferSchema= True,
            header=True,
            multiLine=True,
            escape='"'
        )
        return df_raw.select("Reviews").withColumnRenamed("Reviews","review")

    def _require_train_test(self, directory: Path, train_name: str,valid_name: str, test_name: str) -> None:
        train_path = directory / train_name
        valid_path = directory / valid_name
        test_path = directory / test_name
        if not directory.is_dir() or not train_path.exists() or not valid_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                f"{directory} must contain both {train_name}, {valid_name} and {test_name}"
            )

    def save_split_data(self, df_train: DataFrame, df_valid: DataFrame, df_test: DataFrame) -> None:
        """Save the training, validation, and test DataFrames as separate Parquet datasets."""
        if not paths.SPLIT_DIR.exists():
            paths.SPLIT_DIR.mkdir(parents=True)
        df_train.write.mode("overwrite").parquet(str(paths.SPLIT_DIR / paths.TRAIN_PARQUET))
        df_valid.write.mode("overwrite").parquet(str(paths.SPLIT_DIR / paths.VALID_PARQUET))
        df_test.write.mode("overwrite").parquet(str(paths.SPLIT_DIR / paths.TEST_PARQUET))

    def load_split_data(self) -> tuple[DataFrame, DataFrame, DataFrame]:
        self._require_train_test(paths.SPLIT_DIR, paths.TRAIN_PARQUET, paths.VALID_PARQUET, paths.TEST_PARQUET)
        df_train = self.spark.read.parquet(str(paths.SPLIT_DIR / paths.TRAIN_PARQUET)).cache()
        df_valid = self.spark.read.parquet(str(paths.SPLIT_DIR / paths.VALID_PARQUET)).cache()
        df_test = self.spark.read.parquet(str(paths.SPLIT_DIR / paths.TEST_PARQUET)).cache()
        return df_train, df_valid, df_test

    def save_embedding_data(self, model_name: str, df_train: DataFrame, df_valid: DataFrame, df_test: DataFrame) -> None:
        target_path = paths.EMBEDDING_DIR / model_name
        if not target_path.exists():
            target_path.mkdir(parents=True)
        df_train.write.mode("overwrite").parquet(str(target_path / paths.TRAIN_PARQUET))
        df_valid.write.mode("overwrite").parquet(str(target_path / paths.VALID_PARQUET))
        df_test.write.mode("overwrite").parquet(str(target_path / paths.TEST_PARQUET))

    def load_embedding_data(self) -> tuple[dict[str, DataFrame], dict[str, DataFrame], dict[str, DataFrame]]:
        if not paths.EMBEDDING_DIR.is_dir():
            raise FileNotFoundError(f"{paths.EMBEDDING_DIR} does not exist")
        train_dict = {}
        valid_dict = {}
        test_dict = {}
        # The directory name is the dict key, for example `tfidf`.
        for model_dir in paths.EMBEDDING_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            self._require_train_test(model_dir, paths.TRAIN_PARQUET, paths.VALID_PARQUET, paths.TEST_PARQUET)
            name = model_dir.name
            train_dict[name] = self.spark.read.parquet(str(model_dir / paths.TRAIN_PARQUET)).cache()
            valid_dict[name] = self.spark.read.parquet(str(model_dir / paths.VALID_PARQUET)).cache()
            test_dict[name] = self.spark.read.parquet(str(model_dir / paths.TEST_PARQUET)).cache()
        return train_dict, valid_dict, test_dict

    def save_prediction_data(self, model_name: str, df_train: DataFrame, df_valid: DataFrame, df_test: DataFrame) -> None:
        target_path = paths.PREDICTION_DIR / model_name
        if not target_path.exists():
            target_path.mkdir(parents=True)
        df_train.write.mode("overwrite").parquet(str(target_path / paths.TRAIN_PREDICTION_PARQUET))
        df_valid.write.mode("overwrite").parquet(str(target_path / paths.VALID_PREDICTION_PARQUET))
        df_test.write.mode("overwrite").parquet(str(target_path / paths.TEST_PREDICTION_PARQUET))

    def load_prediction_data(self) -> tuple[dict[str, DataFrame], dict[str, DataFrame], dict[str, DataFrame]]:
        if not paths.PREDICTION_DIR.is_dir():
            raise FileNotFoundError(f"{paths.PREDICTION_DIR} does not exist")
        train_dict = {}
        valid_dict = {}
        test_dict = {}
        # The directory name is `{analytics}__{embedding}`, for example `lr__tfidf`.
        for model_dir in paths.PREDICTION_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            self._require_train_test(
                model_dir,
                paths.TRAIN_PREDICTION_PARQUET,
                paths.VALID_PREDICTION_PARQUET,
                paths.TEST_PREDICTION_PARQUET,
            )
            name = model_dir.name
            train_dict[name] = self.spark.read.parquet(str(model_dir / paths.TRAIN_PREDICTION_PARQUET))
            valid_dict[name] = self.spark.read.parquet(str(model_dir / paths.VALID_PREDICTION_PARQUET))
            test_dict[name] = self.spark.read.parquet(str(model_dir / paths.TEST_PREDICTION_PARQUET))
        return train_dict, valid_dict, test_dict
