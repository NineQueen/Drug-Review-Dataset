import json
import warnings
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import DateType, IntegerType, StringType, StructField, StructType

import src.paths as paths


class CacheScope:
    """Written By AI: Release DataFrame caches created inside a `with` block.

    Exit calls DataFrame.unpersist(), which drops the RDD and the SQL cache
    registration together.
    """

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        self._before: set[int] = set()

    def __enter__(self) -> "CacheScope":
        self._before = self._cached_plan_ids()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._release_new_frames()
        return False

    def _cached_data(self):
        manager = self.spark._jsparkSession.sharedState().cacheManager()
        field = manager.getClass().getDeclaredField("cachedData")
        field.setAccessible(True)
        return field.get(manager)

    def _cached_plan_ids(self) -> set[int]:
        cached = self._cached_data()
        found: set[int] = set()
        for index in range(int(cached.size())):
            found.add(self._plan_id(cached.apply(index).plan()))
        return found

    def _plan_id(self, plan) -> int:
        return int(self.spark._jvm.java.lang.System.identityHashCode(plan))

    def _release_new_frames(self) -> None:
        cached = self._cached_data()
        plans = []
        for index in range(int(cached.size())):
            plan = cached.apply(index).plan()
            if self._plan_id(plan) not in self._before:
                plans.append(plan)
        session = self.spark._jsparkSession
        dataset = self.spark._jvm.org.apache.spark.sql.classic.Dataset
        for plan in plans:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="DataFrame constructor is internal",
                    category=UserWarning,
                )
                frame = DataFrame(dataset.ofRows(session, plan), self.spark)
            frame.unpersist(blocking=True)


class DataOI:
    def __init__(self) -> None:
        # The notebook creates the only session. This call does not start one.
        self.spark = SparkSession.getActiveSession()

    def scope(self) -> CacheScope:
        return CacheScope(self.spark)

    @staticmethod
    def unpersist(*frames: DataFrame) -> None:
        for frame in frames:
            frame.unpersist(blocking=True)

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

    def load_embedding_data(self, embedding_model: str) -> tuple[DataFrame, DataFrame, DataFrame]:
        target_path = paths.EMBEDDING_DIR / embedding_model
        self._require_train_test(target_path, paths.TRAIN_PARQUET, paths.VALID_PARQUET, paths.TEST_PARQUET)
        df_train = self.spark.read.parquet(str(target_path / paths.TRAIN_PARQUET)).cache()
        df_valid = self.spark.read.parquet(str(target_path / paths.VALID_PARQUET)).cache()
        df_test = self.spark.read.parquet(str(target_path / paths.TEST_PARQUET)).cache()
        return df_train, df_valid, df_test

    def save_prediction_data(self, model_name: str, df_train: DataFrame, df_valid: DataFrame, df_test: DataFrame) -> None:
        target_path = paths.PREDICTION_DIR / model_name
        if not target_path.exists():
            target_path.mkdir(parents=True)
        columns = ["uniqueID", "rating", "prediction"]
        frames = (
            (df_train, paths.TRAIN_PREDICTION_PARQUET),
            (df_valid, paths.VALID_PREDICTION_PARQUET),
            (df_test, paths.TEST_PREDICTION_PARQUET),
        )
        # Four partitions keep a large forest from being applied on every core at once.
        for df, filename in frames:
            (
                df.select(*columns)
                .coalesce(4)
                .write.mode("overwrite")
                .parquet(str(target_path / filename))
            )

    def load_prediction_data(
        self, model_names: list[str] | None = None
    ) -> tuple[dict[str, DataFrame], dict[str, DataFrame], dict[str, DataFrame]]:
        """Read selected prediction triplets, or all saved runs when names are omitted."""
        if not paths.PREDICTION_DIR.is_dir():
            raise FileNotFoundError(f"{paths.PREDICTION_DIR} does not exist")
        train_dict = {}
        valid_dict = {}
        test_dict = {}
        # The directory name is `{analytics}__{embedding}`, for example `lr__tfidf`.
        if model_names is None:
            model_dirs = (entry for entry in paths.PREDICTION_DIR.iterdir() if entry.is_dir())
        else:
            model_dirs = (paths.PREDICTION_DIR / name for name in model_names)
        for model_dir in model_dirs:
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

    def _analytics_para_file(self, embedding_model: str, analytics_model: str) -> Path:
        return paths.PARAMETER_DIR / f"{embedding_model}_{analytics_model}.json"

    def save_analytics_para(
        self,
        embedding_model: str,
        analytics_model: str,
        para: dict,
        weighted_f1: float,
    ) -> bool:
        """Write constructor values and the holdout weighted F1 that selected them.

        An existing score is replaced only when `weighted_f1` is strictly higher.
        A file that already exists without `weighted_f1` is left unchanged.
        """
        if not paths.PARAMETER_DIR.exists():
            paths.PARAMETER_DIR.mkdir(parents=True)
        file_name = self._analytics_para_file(embedding_model, analytics_model)
        if file_name.exists():
            stored = json.loads(file_name.read_text()).get("weighted_f1")
            if stored is None or not weighted_f1 > float(stored):
                return False
        record = {
            "embedding_model": embedding_model,
            "analytics_model": analytics_model,
            "weighted_f1": weighted_f1,
        }
        record.update(para)
        record["embedding_model"] = embedding_model
        record["analytics_model"] = analytics_model
        record["weighted_f1"] = weighted_f1
        file_name.write_text(json.dumps(record, indent=2) + "\n")
        return True

    def load_analytics_para(self, embedding_model: str, analytics_model: str) -> dict:
        """Return constructor values. A file without `weighted_f1` is refused."""
        file_name = self._analytics_para_file(embedding_model, analytics_model)
        record = json.loads(file_name.read_text())
        stored = record.get("weighted_f1")
        if stored is None:
            raise ValueError(f"{file_name.name} has no weighted_f1")
        for key in ("embedding_model", "analytics_model", "weighted_f1"):
            record.pop(key, None)
        return record
