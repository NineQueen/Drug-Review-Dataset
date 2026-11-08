from pyspark.sql import DataFrame
from src.dataOI import DataOI
from src.cleaning import webmd_data_cleaning
from src.text_features import text_feature_engineering
from pyspark.sql import functions as F
import contextlib
import io

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

    def __init__(
        self, 
        df_train: DataFrame,
        df_valid:DataFrame,
        df_test: DataFrame,
        use_expanded_data:bool = True
    ) -> None:
        self.train = df_train
        self.valid = df_valid
        self.test = df_test
        if use_expanded_data: self.train_expanded = self._expand_webmd_data()

    def _expand_webmd_data(self) -> DataFrame:
        words_col = ["words","words_nostop","words_featured"]
        data_oi = DataOI()
        with contextlib.redirect_stdout(io.StringIO()):
            reviews_webmd = text_feature_engineering(
                webmd_data_cleaning(data_oi.load_webmd_data())
            )
        valid_test_reviews = (
            self.valid.select("review")
            .unionByName(self.test.select("review"))
            .dropDuplicates(["review"])
        )
        reviews_webmd = (
            reviews_webmd.join(valid_test_reviews, "review", "left_anti")
            .filter(F.size("words_nostop") > 0)
            .select(*words_col)
        )
        return (
            self.train.select(*words_col)
            .unionByName(reviews_webmd)
            .dropDuplicates(["words_nostop"])
        )

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
