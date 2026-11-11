from functools import wraps

from gensim.models.phrases import FrozenPhrases
from pyspark.sql import DataFrame
from src.dataOI import DataOI
from src.cleaning import webmd_data_cleaning
from src.text_features import text_feature_engineering, apply_phrases
from pyspark.sql import functions as F

class EmbeddingModel:
    """Text representation fitted on the training split.

    The instance holds `self.train`, `self.valid`, and `self.test`.
    `train` fits on `self.train` and saves the model under `EMBEDDING_MODELS_DIR / model_name`.
    `predict` loads that saved model and transforms the training, validation, and test frames.
    It returns those three frames in that order.

    `predict` adds a vector column named `embedding`, then keeps only the columns
    analytics reads. Fit vocabulary and other statistics on `self.train` only.
    The validation split checks the fitted model. The test split is reserved for final evaluation.
    The train and predict functions share one registry key, and that key must not contain `__`.
    """

    train_registry = {}
    predict_registry = {}
    # TF-IDF is the baseline representation and stays out of the two registries.
    baseline_train = None
    baseline_predict = None
    # Text columns and vectorizer intermediates stay on `self.train` for fitting.
    # Predict output keeps the label, the training weight, and the encoded model inputs.
    OUTPUT_COLUMNS = (
        "uniqueID",
        "rating",
        "rating_p1",
        "weight",
        "condition_idx",
        "condition_vec",
        "days",
        "usefulCount_log",
        "drugName_idx",
        "drugName_vec",
        "embedding",
    )

    def __init__(
        self, 
        df_train: DataFrame,
        df_valid:DataFrame,
        df_test: DataFrame,
        phrases: FrozenPhrases | None = None
    ) -> None:
        self.train = df_train
        self.valid = df_valid
        self.test = df_test
        self.phrases = phrases
        self._train_expanded = None
        self._train_expanded_phrase = None
        if phrases is not None:
            self.train_phrase = apply_phrases(df_train,phrases)
            self.valid_phrase = apply_phrases(df_valid,phrases)
            self.test_phrase = apply_phrases(df_test,phrases)

    @property
    def train_expanded(self) -> DataFrame:
        if self._train_expanded is None:
            self._train_expanded = self._load_expand_webmd_data()
        return self._train_expanded

    @property
    def train_expanded_phrase(self) -> DataFrame:
        if self.phrases is None:
            raise ValueError("Phrase rules must be supplied for this training split.")

        if self._train_expanded_phrase is None:
            self._train_expanded_phrase = apply_phrases(
                self.train_expanded, self.phrases
            )
        return self._train_expanded_phrase

    def _load_expand_webmd_data(self) -> DataFrame:
        words_col = ["words","words_nostop","words_featured"]
        data_oi = DataOI()
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
            @wraps(function)
            def train(self, *args, **kwargs):
                with DataOI().scope():
                    return function(self, *args, **kwargs)
            setattr(cls, function.__name__, train)
            cls.train_registry[name] = train
            return train
        return decorator

    @staticmethod
    def _model_columns(df: DataFrame) -> DataFrame:
        keep = [column for column in EmbeddingModel.OUTPUT_COLUMNS if column in df.columns]
        return df.select(*keep)

    @classmethod
    def predict_register(cls, name: str):
        def decorator(function):
            def predict(self, model_name: str = name):
                train_df, valid_df, test_df = function(self, model_name)
                return (
                    cls._model_columns(train_df),
                    cls._model_columns(valid_df),
                    cls._model_columns(test_df),
                )
            predict.__name__ = function.__name__
            setattr(cls, function.__name__, predict)
            cls.predict_registry[name] = predict
            return predict
        return decorator

    @classmethod
    def baseline_train_register(cls):
        def decorator(function):
            setattr(cls, function.__name__, function)
            cls.baseline_train = function
            return function
        return decorator

    @classmethod
    def baseline_predict_register(cls):
        def decorator(function):
            def predict(self, model_name: str = "baseline"):
                train_df, valid_df, test_df = function(self, model_name)
                return (
                    cls._model_columns(train_df),
                    cls._model_columns(valid_df),
                    cls._model_columns(test_df),
                )
            predict.__name__ = function.__name__
            setattr(cls, function.__name__, predict)
            cls.baseline_predict = predict
            return predict
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
