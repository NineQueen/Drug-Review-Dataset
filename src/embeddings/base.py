from pyspark.sql import DataFrame


class EmbeddingModel:
    """Text representation fitted on the training split.

    Give the fitted object one attribute on this instance, such as `self.tfidf_model`.
    `train(self, model_name)` fits on `self.train` only, assigns that attribute, and saves
    the model under `EMBEDDING_MODELS_DIR / model_name`.
    `predict(self, model_name)` uses the attribute when it is already set. When the attribute
    is missing or None, it loads `model_name` from disk and assigns the attribute.
    One instance keeps one fitted object. A different `model_name` needs another instance.

    Each returned frame keeps the input columns and adds a vector column named `embedding`.
    Analytics models read that column. Fit vocabulary and other statistics on `self.train` only.
    The train and predict functions share one registry key, and that key must not contain `__`.
    """

    train_registry = {}
    predict_registry = {}

    def __init__(self, df_train: DataFrame, df_test: DataFrame) -> None:
        self.train = df_train
        self.test = df_test

    @classmethod
    def train_register(cls, name: str):
        def decorator(function):
            # `name` is the registry key. The notebook calls `function(instance)`, so `self` is not bound here.
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
