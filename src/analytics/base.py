from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame


class AnalyticsModel:
    """Predictor trained on frames that already contain `embedding`.

    Give the fitted classifier one attribute on this instance, such as `self.lr_model`.
    `train(self, embedding_model)` fits on the training frame only, assigns that attribute,
    and saves the model under `{registry_key}__{embedding_model}`.
    `predict(self, embedding_model)` uses the attribute when it is already set. When the
    attribute is missing or None, it loads that directory and assigns the attribute.
    One instance keeps one fitted object. A different embedding name needs another instance.

    Each returned frame has only `uniqueID`, `rating`, and `prediction`.
    `rating` stays on the original 1-10 scale. `prediction` uses the same 0-9 classes as `rating_p1`.
    Leave the prediction on that scale; `Scorer` builds `rating_pred = prediction + 1`.

    A model that needs one feature vector calls `_feature_transform` with its own column list.
    `weight`, when a model uses it, exists only on the training split.
    The train and predict functions share one registry key, and that key must not contain `__`.
    """

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
