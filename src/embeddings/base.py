from pyspark.sql import DataFrame


class EmbeddingModel:
    """Text representation fitted on the training split.

    The instance holds `self.train`, `self.valid`, and `self.test`.
    `train` fits on `self.train` and saves the model under `EMBEDDING_MODELS_DIR / model_name`.
    `predict` loads that saved model and transforms the training, validation, and test frames.
    It returns those three frames in that order.

    Each frame keeps the input columns and adds a vector column named `embedding`.
    Analytics models read that column. Fit vocabulary and other statistics on `self.train` only.
    The validation split checks the fitted model. The test split is reserved for final evaluation.
    The train and predict functions share one registry key, and that key must not contain `__`.
    """

    train_registry = {}
    predict_registry = {}

    def __init__(self, df_train: DataFrame,df_valid:DataFrame, df_test: DataFrame) -> None:
        self.train = df_train
        self.valid = df_valid
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
