"""Spark random forest classifier for the analytics registry."""

from typing import Any

from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import DataFrame

from src.analytics.base import AnalyticsModel


# Current TF-IDF baseline (test accuracy 0.360036); copy before overrides.
# Unspecified parameters retain their PySpark defaults, including leaf size 1.
RF_DEFAULT_PARAMS: dict[str, Any] = {
    "numTrees": 200,
    "maxDepth": 16,
    "seed": 42,
}

# Suggested initial search bounds, not proven optima or full API limits.
# "grid" gives discrete candidates; the other fields describe a Bayesian space.
RF_PARAM_RANGES: dict[str, dict[str, Any]] = {
    "numTrees": {
        "type": "int", "low": 50, "high": 300, "step": 50,
        "grid": [100, 200, 300], "note": "More trees increase training and inference cost.",
    },
    "maxDepth": {
        "type": "int", "low": 4, "high": 16, "step": 2,
        "grid": [8, 12, 16], "note": "Shallower trees impose stronger regularization.",
    },
    "minInstancesPerNode": {
        "type": "int", "low": 1, "high": 30, "step": 1,
        "grid": [1, 5, 20], "note": "Larger values restrict small leaves.",
    },
    "minInfoGain": {
        "type": "float", "low": 0.0, "high": 0.02, "log": False,
        "grid": [0.0, 0.001, 0.01], "note": "Higher minimum split gain restricts growth.",
    },
    "subsamplingRate": {
        "type": "float", "low": 0.6, "high": 1.0, "log": False,
        "grid": [0.6, 0.8, 1.0], "note": "Fraction of training rows sampled per tree.",
    },
    "featureSubsetStrategy": {
        "type": "categorical", "choices": ["sqrt", "log2", "0.05", "0.1"],
        "grid": ["sqrt", "log2", "0.05"],
        "note": "Fractions must be strings. auto equals sqrt for multiclass forests with multiple trees.",
    },
    "maxBins": {
        "type": "categorical", "choices": [16, 32, 64],
        "grid": [16, 32, 64],
        "note": "Must cover categorical feature cardinalities; larger values use more histogram memory.",
    },
    "impurity": {
        "type": "categorical", "choices": ["gini", "entropy"],
        "grid": ["gini", "entropy"], "note": "Split-quality criterion.",
    },
}

# Small first grid: 2 * 3 * 3 = 18 fits, before any cross-validation multiplier.
# Add other keys from RF_PARAM_RANGES only in a later, narrower search.
RF_PARAM_GRID: dict[str, list[Any]] = {
    name: list(RF_PARAM_RANGES[name]["grid"])
    for name in ("numTrees", "maxDepth", "minInstancesPerNode")
}

# Library-neutral metadata: map int/float/categorical to the optimizer's sampler.
RF_BAYES_SPACE: dict[str, dict[str, Any]] = {
    name: {key: value for key, value in spec.items() if key not in {"grid", "note"}}
    for name, spec in RF_PARAM_RANGES.items()
}

# Search resource settings are separate from RF_DEFAULT_PARAMS: the restored
# baseline keeps its original defaults. maxMemoryInMB is not the JVM heap limit.
RF_SEARCH_FIXED_PARAMS: dict[str, Any] = {"seed": 42, "maxMemoryInMB": 64}

# These are analytics-function arguments, NOT RandomForestClassifier arguments.
RF_RUN_PARAM_GRID: dict[str, list[bool]] = {"balance_classes": [False, True]}

RF_SEARCH_NOTES: dict[str, str] = {
    "memory": "Deep trees, large maxBins and large feature fractions can still exceed the 4 GB JVM heap; start with a narrow search.",
    "evaluation": "Fit on train, select on validation (record accuracy and weighted F1), and evaluate test only after selection.",
    "weights": "Pass balance_classes separately to the registered analytics function; do not pass it inside rf_params.",
    "thresholds": "Prediction thresholds are post-fit settings, excluded from the constructor search space.",
    "artifacts": "Record any active embedding-specific selection/calibration files so trials use comparable preprocessing and decision rules.",
    "usage": "Pass rf_params={**RF_SEARCH_FIXED_PARAMS, **candidate} to the registered analytics function; do not modify RF_DEFAULT_PARAMS in place.",
}


@AnalyticsModel.register("random_forest")
def random_forest(
    self: AnalyticsModel,
    *,
    feature_cols: list[str] | None = None,
    rf_params: dict[str, Any] | None = None,
    balance_classes: bool = True,
    save_model: bool = False,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Fit once on train and return train, validation, and test predictions.

    Defaults use 200 trees, depth 16, condition/date/vote features and class weights.
    A saved training-selected text subset is applied when available.
    The original depth-5 unweighted configuration predicted only rating 10.
    Use rf_params for tree regularization and balance_classes=False for an
    explicitly unweighted comparison.
    Categorical feature metadata is preserved when assembling custom columns.
    """
    features = (
        ["embedding", "condition_vec", "days", "usefulCount_log"]
        if feature_cols is None else list(feature_cols)
    )
    from src.paths import ANALYTIC_MODELS_DIR
    import json
    selected_path = ANALYTIC_MODELS_DIR / f"random_forest__{self.embedding_model}_selection.json"
    selection = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else None
    if not features:
        raise ValueError("feature_cols must contain at least one feature")
    invalid_features = {"uniqueID", "rating", "rating_p1", "weight"}.intersection(features)
    if invalid_features:
        raise ValueError(f"Columns cannot be used as features: {sorted(invalid_features)}")

    parameters: dict[str, Any] = dict(RF_DEFAULT_PARAMS)
    if rf_params is not None:
        parameters.update(rf_params)
    calibration_path = ANALYTIC_MODELS_DIR / f"random_forest__{self.embedding_model}_calibration.json"
    if calibration_path.exists() and "thresholds" not in parameters:
        parameters["thresholds"] = json.loads(calibration_path.read_text(encoding="utf-8"))["thresholds"]
    fixed_columns = {"featuresCol", "labelCol", "predictionCol"}.intersection(parameters)
    if fixed_columns:
        raise ValueError(f"Column parameters are fixed by the analytics interface: {sorted(fixed_columns)}")

    if balance_classes:
        if parameters.get("weightCol", "weight") != "weight":
            raise ValueError("balance_classes requires weightCol='weight'")
        parameters["weightCol"] = "weight"
    weight_col = parameters.get("weightCol")
    if weight_col and weight_col not in self.train_embedding.columns:
        raise ValueError(f"Training data is missing weight column: {weight_col}")

    if selection and "embedding" in features:
        train_df, valid_df, test_df, preprocessing = _slice_rf_frames(
            [self.train_embedding, self.valid_embedding, self.test_embedding],
            features, selection["indices"])
    else:
        train_df, valid_df, test_df = self._feature_transform(features)
    # Spark exposes thresholds on the fitted model, not the RF constructor.
    thresholds = parameters.pop("thresholds", None)
    classifier = RandomForestClassifier(
        featuresCol="features",
        labelCol="rating_p1",
        predictionCol="prediction",
        **parameters,
    )
    print("Random forest feature columns:", features)
    print("Random forest effective parameters:", parameters)
    if thresholds is not None:
        print("Random forest prediction thresholds:", thresholds)

    model = classifier.fit(train_df)
    if thresholds is not None:
        model.setThresholds(thresholds)
    if save_model:
        from src.paths import ANALYTIC_MODELS_DIR

        ANALYTIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        saved = model
        if selection and "embedding" in features:
            from pyspark.ml import PipelineModel
            saved = PipelineModel(stages=preprocessing + [model])
        saved.write().overwrite().save(str(ANALYTIC_MODELS_DIR / f"random_forest__{self.embedding_model}"))
    columns = ["uniqueID", "rating", "prediction"]

    return (
        model.transform(train_df).select(*columns),
        model.transform(valid_df).select(*columns),
        model.transform(test_df).select(*columns),
    )


def _slice_rf_frames(frames, features, indices):
    from pyspark.ml.feature import VectorSlicer, VectorAssembler

    slicer = VectorSlicer(inputCol="embedding", outputCol="rf_text", indices=indices)
    assembler = VectorAssembler(inputCols=["rf_text" if c == "embedding" else c for c in features],
                                outputCol="features")
    transformed = [assembler.transform(slicer.transform(frame)).cache() for frame in frames]
    return (*transformed, [slicer, assembler])
