from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


class AnalyticsModel:
    train_registry = {}
    predict_registry = {}

    def __init__(self, df_train: DataFrame, df_test: DataFrame) -> None:
        self.train_embedding = df_train
        self.test_embedding = df_test

    def _feature_transform(self, features: list[str]) -> tuple[DataFrame, DataFrame]:
        """Assemble `features` from the columns this model declares.

        Call at the start of train and predict. The column list is not fixed here.
        """
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        train_df = assembler.transform(self.train_embedding).cache()
        test_df = assembler.transform(self.test_embedding).cache()
        return train_df, test_df

    @classmethod
    def train_register(cls, name: str):
        def decorator(function):
            # `name` is the registry key. The notebook calls `function(instance, embedding_name)`.
            setattr(cls, function.__name__, function)
            cls.train_registry[name] = function
            return function
        return decorator

    @classmethod
    def predict_register(cls, name: str):
        def decorator(function):
            setattr(cls, function.__name__, function)
            cls.predict_registry[name] = function
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
