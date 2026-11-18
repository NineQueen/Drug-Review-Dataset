from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame

from src.dataOI import DataOI


class AnalyticsModel:
    """Predictor trained on frames that already contain `embedding`.

    The constructor stores the embedding name and three frames:
    `self.train_embedding`, `self.valid_embedding`, and `self.test_embedding`.
    One registered function fits on the training frame and returns train,
    validation, and test predictions, in that order. One instance is tied to
    one embedding name. Every embedding uses that same function.

    The function may take `para`, defaulting to the module-level `DEFAULT_PARA`.
    That dict is flat and is the same for every embedding. `PARA_RANGE` is one
    flat dict of candidate lists; the model decides how to search it. The
    shared runner calls `registry[key](instance)` and therefore uses the
    default. A tuning call may pass another flat dict with the same keys. The required
    baseline uses `baseline_register`, takes no parameter dict, and is called as
    `instance.baseline()`. Weighted LR uses `improved_baseline_register` and is
    called as `instance.improved_baseline()`. Neither is in `registry`. The
    registry key must not contain `__`.

    Readable inputs are `embedding` and the already encoded non-text columns.
    Do not read text columns or build label-derived features. Fit only on the
    training frame. Validation and test are transformed, not used to fit,
    calibrate, or select parameters.

    Each returned frame has `uniqueID`, `rating`, `prediction`, and
    `probability`. `rating` stays on the original 1-10 scale. `prediction`
    uses the same 0-9 classes as `rating_m1`. Leave the prediction on that
    scale; `Scorer` builds `rating_pred = prediction + 1`. `probability` is an
    array of 10 doubles indexed by class. SVM stores one-vs-rest scores there;
    the other classifiers store class probabilities. The function returns those
    frames and does not save a classifier.

    A model that needs one feature vector calls `_feature_transform` with its
    fixed column list. That call returns the training, validation, and test
    frames. `weight`, when a model uses it, exists only on the training split.
    """

    registry = {}
    baseline = None
    improved_baseline = None

    def __init__(self,embedding_model:str, df_train: DataFrame,df_valid:DataFrame, df_test: DataFrame) -> None:
        self.embedding_model = embedding_model
        self.train_embedding = df_train
        self.valid_embedding = df_valid
        self.test_embedding = df_test

    def _feature_transform(self, features: list[str]) -> tuple[DataFrame,DataFrame, DataFrame]:
        """Assemble `features` on the training, validation, and test frames.

        Call from the registered function with that model's fixed column list.
        The list is not an argument of the registered function. The previous
        feature cache on this instance is released first.
        """
        self.release_features()
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        train_df = assembler.transform(self.train_embedding).cache()
        valid_df = assembler.transform(self.valid_embedding).cache()
        test_df = assembler.transform(self.test_embedding).cache()
        self._feature_frames = (train_df, valid_df, test_df)
        return train_df, valid_df, test_df

    def release_features(self) -> None:
        # Written by AI to release memory held by the previous feature cache.
        """Drop feature caches created by the last `_feature_transform`."""
        DataOI.unpersist(*getattr(self, "_feature_frames", ()))
        self._feature_frames = ()

    @classmethod
    def register(cls, name: str):
        def decorator(function):
            setattr(cls, function.__name__, function)
            cls.registry[name] = function
            return function
        return decorator
    
    @classmethod
    def baseline_register(cls):
        def decorator(function):
            # The notebook calls `instance.baseline()`.
            setattr(cls, function.__name__, function)
            cls.baseline = function
            return function
        return decorator

    @classmethod
    def improved_baseline_register(cls):
        def decorator(function):
            # The notebook calls `instance.improved_baseline()`.
            setattr(cls, function.__name__, function)
            cls.improved_baseline = function
            return function
        return decorator


# Import sibling modules so their registration decorators run on package import.
def _load_registered_modules() -> None:
    import importlib
    import pkgutil
    from pathlib import Path

    package_dir = str(Path(__file__).resolve().parent)
    for module in pkgutil.iter_modules([package_dir]):
        if module.name == "base":
            continue
        importlib.import_module(f"{__package__}.{module.name}")


_load_registered_modules()
