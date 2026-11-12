from collections.abc import Callable

from pyspark.sql import DataFrame, SparkSession

from src.dataOI import DataOI
from src.paths import TEMP_DIR, TRAIN_PARQUET, VALID_PARQUET
from src.seed import seed
from src.split import split


class KFold:
    """Prepare k folds, then return the next saved pair from `next_fold`."""

    def __init__(
        self,
        model_name: str,
        df: DataFrame,
        k: int,
        prepare_function: Callable[[DataFrame, DataFrame], tuple[DataFrame, DataFrame]],
        SEED: int = seed(1),
    ) -> None:
        self.model_name = model_name
        self.k = k
        self._index = 0
        self._loaded: tuple[DataFrame, DataFrame] | None = None
        self._prepare(df, prepare_function, SEED)

    @classmethod
    def open(cls, model_name: str, k: int) -> "KFold":
        """Read folds already written under temp, without preparing them again."""
        folds = cls.__new__(cls)
        folds.model_name = model_name
        folds.k = k
        folds._index = 0
        folds._loaded = None
        return folds

    def reset(self) -> None:
        """Drop the loaded pair and start again at the first fold."""
        if self._loaded is not None:
            DataOI.unpersist(*self._loaded)
            self._loaded = None
        self._index = 0

    def next_fold(self) -> tuple[DataFrame, DataFrame]:
        """Load the next saved `(fit, holdout)` pair and release the previous one."""
        if self._index >= self.k:
            raise IndexError(f"{self.model_name} has {self.k} folds")
        if self._loaded is not None:
            DataOI.unpersist(*self._loaded)
        directory = TEMP_DIR / self.model_name / f"fold_{self._index}"
        spark = SparkSession.getActiveSession()
        fit = spark.read.parquet(str(directory / TRAIN_PARQUET)).cache()
        holdout = spark.read.parquet(str(directory / VALID_PARQUET)).cache()
        fit.count()
        holdout.count()
        self._loaded = (fit, holdout)
        self._index += 1
        return fit, holdout

    def _prepare(
        self,
        df: DataFrame,
        prepare_function: Callable[[DataFrame, DataFrame], tuple[DataFrame, DataFrame]],
        seed: int,
    ) -> None:
        source = df.drop("weight") if "weight" in df.columns else df
        with DataOI().scope():
            parts = list(split(source, seed, (1,) * self.k))
            for fold in range(self.k):
                self._write_fold(parts, fold, prepare_function)

    def _write_fold(
        self,
        parts: list[DataFrame],
        fold: int,
        prepare_function: Callable[[DataFrame, DataFrame], tuple[DataFrame, DataFrame]],
    ) -> None:
        fit = None
        for index, part in enumerate(parts):
            if index == fold:
                continue
            fit = part if fit is None else fit.unionByName(part)
        # The copied fit is cached inside this scope, so releasing it does not
        # release the saved fold part.
        with DataOI().scope():
            fit = fit.select("*").cache()
            fit.count()
            prepared_fit, prepared_holdout = prepare_function(fit, parts[fold])
            directory = TEMP_DIR / self.model_name / f"fold_{fold}"
            directory.mkdir(parents=True, exist_ok=True)
            prepared_fit.write.mode("overwrite").parquet(str(directory / TRAIN_PARQUET))
            prepared_holdout.write.mode("overwrite").parquet(str(directory / VALID_PARQUET))
