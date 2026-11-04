from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


class AnalyticsModel:
    """Predictor trained on frames that already contain `embedding`.

    The constructor stores the embedding name and three frames:
    `self.train_embedding`, `self.valid_embedding`, and `self.test_embedding`.
    One registered function fits on the training frame and returns train,
    validation, and test predictions, in that order. One instance is tied to
    one embedding name.

    `register` stores that function in `registry`. The runner calls
    `registry[key](instance)`. The required baseline uses `baseline_register`
    and is called as `instance.baseline()`. The registry key must not contain `__`.

    Each returned frame has only `uniqueID`, `rating`, and `prediction`.
    `rating` stays on the original 1-10 scale. `prediction` uses the same 0-9 classes as `rating_p1`.
    Leave the prediction on that scale; `Scorer` builds `rating_pred = prediction + 1`.

    A model that needs one feature vector calls `_feature_transform` with its own column list.
    That call returns the training, validation, and test frames.
    `weight`, when a model uses it, exists only on the training split.
    The validation split checks the fitted model. The test split is reserved for final evaluation.
    """

    registry = {}   
    baseline = None

    def __init__(self,embedding_model:str, df_train: DataFrame,df_valid:DataFrame, df_test: DataFrame) -> None:
        self.embedding_model = embedding_model
        self.train_embedding = df_train
        self.valid_embedding = df_valid
        self.test_embedding = df_test

    def _feature_transform(self, features: list[str]) -> tuple[DataFrame,DataFrame, DataFrame]:
        """Assemble `features` on the training, validation, and test frames.

        Call from the registered function. The column list is not fixed here.
        """
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        train_df = assembler.transform(self.train_embedding).cache()
        valid_df = assembler.transform(self.valid_embedding).cache()
        test_df = assembler.transform(self.test_embedding).cache()
        return train_df, valid_df, test_df

    @classmethod
    def register(cls, name: str):
        def decorator(function):
            # `name` is the registry key. The notebook calls `function(instance)`.
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
