from pyspark.sql import DataFrame


class EmbeddingModel:
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
