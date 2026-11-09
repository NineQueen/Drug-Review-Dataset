# Coding-agent contract

Read this file before editing. It applies to the whole repository. The user's current task determines which files you may change; the boundaries below are the default for a model assignment. See `readme.md` for setup and `docs/interface.md` for the full interface. Keep these documents consistent with the implementation.

## Goal and ownership

Predict the original drug-review `rating` as ten classes, 1–10, with PySpark. The required baseline is unweighted logistic regression with `tfidf`. The unweighted classifier is registered with `baseline_register` and called as `baseline()`. Other classifiers use `register` and a unique key, such as `lr_improved`. N is the number of group members: advanced modelling needs N distinct non-LR classifiers paired with separate non-TF-IDF embeddings. TF-IDF is not a word embedding, and neither LR nor weighted LR counts toward N. Do not use one model as both the embedding and classifier. Advanced modelling and final report assembly are separate tasks.

- **Embedding owner:** implement only the assigned `src/embeddings/<name>.py`. Read the encoded split through the existing runner, fit the representation, save its own model, and return train, validation, and test frames containing `embedding`.
- **Analytics owner:** implement only the assigned `src/analytics/<name>.py`. Read `embedding` and the already encoded non-text columns, fit the classifier, and return train, validation, and test predictions from one registered function. Text processing, encoding, and representation stay in their own stages.
- **Integration owner:** maintain preprocessing, splitting, encoding, base classes, registration discovery, I/O, evaluation, shared notebooks, dependencies, and documentation. Assemble final deliverables.

A model assignment does not authorize editing another model, `base.py`, `__init__.py`, `src/paths.py`, preprocessing, evaluation, or the main notebook. Implement helper functions and model-specific transforms in the assigned model file. If a shared change or new dependency is necessary, report the concrete need to the integration owner instead of silently expanding the task. An explicit user request to maintain shared code or documentation overrides these default boundaries.

A personal experiment may pass another `para` dict of the same shape. That changes constructor values only and does not change the feature list, weighting decision, or pipeline. Do not require another member to modify shared code to register or use your model. Do not invent member names, model assignments, or completed experiments.

## Registration

Add one model module directly under its assigned folder. Import the base class from `src.embeddings.base` or `src.analytics.base` **inside model modules**, not from the package entry point: discovery runs while the base module is being imported. Notebooks may import from `src.embeddings` and `src.analytics`.

Importing either package discovers sibling modules automatically. Do not edit `__init__.py` or discovery loops. Keep module import free of training, downloads, file writes, and Spark-session creation. Put optional model-specific imports inside the functions that need them so another model can still be imported without that dependency.

Use a unique lowercase registry key. A single underscore is allowed; `__`, path separators, and duplicate keys are not. An embedding registers train and predict with the same key and distinct function names such as `<name>_train` and `<name>_predict`. An analytics model registers one function. The shared runner calls `AnalyticsModel.registry[key](instance)` and therefore uses the default `para`. The required unweighted baseline uses `baseline_register`, takes no parameter dict, and is called as `instance.baseline()`.

An embedding `train` fits on the training split and saves the model directory. `predict` loads that directory and transforms the training, validation, and test frames. An embedding directory name is the `model_name` argument. An analytics function fits on the training frame and returns the three prediction frames in the same call. Its combined result name is `{registry_key}__{self.embedding_model}`; that embedding name is fixed when the instance is constructed.

```python
# Embedding module: self.train, self.valid, and self.test are Spark DataFrames.
@EmbeddingModel.train_register("my_embedding")
def my_embedding_train(self, model_name: str = "my_embedding") -> None:
    model = ...  # fit on self.train, then save EMBEDDING_MODELS_DIR / model_name

@EmbeddingModel.predict_register("my_embedding")
def my_embedding_predict(self, model_name: str = "my_embedding") -> tuple[DataFrame, DataFrame, DataFrame]:
    model = ...  # load model_name
    ...  # Return train, then validation, then test.

# Analytics module. One flat parameter dict for every embedding.
# DEFAULT_PARA values are constructor arguments.
# PARA_RANGE values are candidate lists. How this model searches them is its own choice.
# The runner does not read PARA_RANGE. Do not key either dict by embedding name,
# and do not put feature lists, labels, paths, or pipeline switches in them.
DEFAULT_PARA = {"regParam": 0.1, "maxIter": 100}
PARA_RANGE = {"regParam": [0.01, 0.1, 0.3, 1.0], "maxIter": [50, 100]}

@AnalyticsModel.register("my_classifier")
def my_classifier(self, para: dict = DEFAULT_PARA) -> tuple[DataFrame, DataFrame, DataFrame]:
    model = ...  # fit on the training frame with **para; do not mutate para
    ...  # Return train, validation, then test predictions.
```

These are interface sketches; implement the algorithm before running them. Each registry stores an unbound function. The shared runner calls `registry[key](instance)`. A tuning call may pass a flat `para` with the same keys as `DEFAULT_PARA`. Keep an embedding's train and predict separate. Do not split an analytics model into train and predict. The integration runner trains selected embeddings before calling their predict entries, then calls each selected analytics function once. Do not special-case your model in its loops. Every embedding name shares that one function and the same default parameters.

## Data contract

- Fit vocabulary, IDF, encoders, scalers, reducers, and classifiers on training data only. Any value that changes a prediction, including thresholds, class offsets, and selected feature subsets, is also fit on training rows only. Inside an analytics function the validation and test frames are only transformed. Their scores are computed after the function returns, and the reported comparison uses the test score. A pretrained embedding may be loaded, but any dataset-specific fitting uses training rows only.
- Preserve `uniqueID`, rows, and original labels. Embeddings keep all existing columns and add the Spark ML `VectorUDT` column `embedding`, with the same dimension for every train, validation, and test row. Do not return a Python list or Spark array in place of the vector.
- Inputs retain `review`, `words`, `words_nostop`, and `words_featured`. TF-IDF uses `words_featured`. An embedding may instead read `review`, `words`, or `words_nostop` without changing shared text processing. An analytics function does not read those text columns and does not fit another vectorizer or embedding.
- Keep the preprocessing boundary fixed: `data_cleaning` retains the original record filtering, configured condition/HTML replacements, paired outer-wrapper removal, ordinary-whitespace cleanup, lowercasing, deterministic review deduplication, and numeric derived columns. Remove a wrapper only when the same marker is at both ends: one literal `"` pair, then one `&quot;` pair. A quote on only one end stays. Collapse ordinary whitespace before that check and again after HTML decoding. Shared text repairs belong to `text_feature_engineering` and must update `review` before the temporary `review_clean` copy used for tokens. The repairs currently written back to `review` are character and Unicode whitespace normalization, contraction expansion, and dose spacing. Tokenization, stop-word removal, and stemming must not overwrite `review`. Keep negation words when removing stop words. Return `review`, `words`, `words_nostop`, and `words_featured`, without adding another permanent text column. The main notebook enables bigrams. `baseline_text_feature_engineering` only tokenizes `review` for the separate baseline TF-IDF path. The text stage preserves rows, IDs, and labels and does not deduplicate again. Audit normalization collisions separately before model comparison.
- New text repairs must address broadly applicable patterns. Do not build dataset-specific typo dictionaries, add per-review exceptions, or require exhaustive manual correction of the dataset. Preserve ambiguous expressions rather than guessing; aggregate checks and representative examples validate the supported rules.
- Classifiers train on `rating_p1 = rating - 1`, range 0–9. Return `prediction` as an integer-valued double in that range, not probabilities or original 1–10 ratings. Only `Scorer` adds 1. Analytics outputs contain exactly `uniqueID`, `rating`, `prediction`.
- Analytics receives `self.train_embedding`, `self.valid_embedding`, and `self.test_embedding`. Its readable model inputs are `embedding` and the existing encoded columns `condition_idx`, `condition_vec`, `days`, `usefulCount_log`, `drugName_idx`, and `drugName_vec`. The feature list is a module constant, not a function argument. `self._feature_transform(features)` returns training, validation, and test frames with an assembled `features` column; it does not mutate the stored inputs. Use that same feature list for the fit and the returned predictions. Do not use `rating`, `rating_p1`, `weight`, or any aggregate, smoothed value, or encoding computed from the labels. Do not create a new drug-target column; the current logistic models use the existing `drugName_vec`.
- The encoder fits `condition` one-hot columns `condition_idx` and `condition_vec` on the training split only, with `handleInvalid="keep"`, then transforms validation and test. Drug names that appear at least 10 times in training stay distinct; rarer and unseen names become `__other__` in `drugName_grouped`, then `drugName_idx` and one-hot `drugName_vec`. It does not add `drug_mean_rating`, and an analytics model must not compute a label-derived drug target either. The current logistic models assemble `embedding`, `condition_vec`, `days`, `usefulCount_log`, and `drugName_vec`. Do not change this shared encoding as part of a model assignment. Rebuild downstream artifacts after a shared encoding change.
- `days` is days since 1970-01-01. `usefulCount_log` is `log1p(usefulCount)`, where useful votes accumulate after review publication. The training-only `weight` is `n / (k * class_count)`; pass it as a model weight when appropriate, and do not require it on validation or test rows.
- Splitting uses `pmod(xxhash64(uniqueID, SEED), 10000)`. Buckets below 7000 are train, 7000–7999 are validation, and 8000 and above are test, approximately 70/10/20. Train is the only fitting split. Validation and test predictions are scored after the analytics function returns; the reported comparison uses test. With the same data, ID types, Spark version, and seed, membership is independent of row order and partitioning. Keep this shared policy; do not replace it with `randomSplit`.

## Artifacts and evaluation

- Raw CSV: `data/raw/drugsComTrain_raw.csv`.
- Encoded split: `data/processed/split/{train,valid,test}.parquet`.
- Embedding frames: `data/processed/embeddings/<embedding>/{train,valid,test}.parquet`.
- Prediction frames: `data/processed/predictions/<analytics>__<embedding>/{train_predictions,valid_predictions,test_predictions}.parquet`.
- Fitted representation: `src/embeddings/models/<embedding>/`.
- Analytics deliverable: the three prediction frames. The registered function does not save a classifier, thresholds, or search results.

Use path constants from `src/paths.py`. Create only a Spark embedding-model directory's parent before `write().overwrite().save()`. An embedding owner writes only that directory; the runner writes DataFrames, including analytics predictions, through `DataOI`. Do not overwrite another model's artifacts. Assume the runner supplied the documented frames and a `para` dict of the documented shape; do not add branches that recover missing columns, optional files, or alternate pipelines.

Combined names use `analytics__embedding` in that order. Keep public column names `Embedding_Model`, `Analytic_Model`, `rating_p1`, and class name `DataOI` unchanged. Scoring uses accuracy and support-weighted precision and F1. Support-weighted recall is omitted because it equals accuracy for this single-label task. MAE/RMSE are supplementary errors on original rating units, not a separate regression task. In `Scorer.add_model`, keep the `|` between the two nonmatching-name conditions: only an exact model pair is replaced.

## Running, caching, and imports

The entry notebook creates one active SparkSession before constructing `DataOI`. Use the existing `local[*]` / 4g setup. Do not start a session in a model module or in `DataOI`. Restart the kernel after changing driver memory or after the JVM exits.

`project.ipynb` is the integration runner. Personal runners are `experiments/embedding_pipeline.ipynb` and `experiments/analytics_pipeline.ipynb`; select `embedding_name = "tfidf"` in the embedding runner to check the baseline representation. Choose the assigned registry key in the runner. Personal runners score that run's train, validation, and test predictions; the reported comparison uses the test predictions. The main runner may aggregate saved results.

`REBUILD` controls input preparation, not whether a model trains. On a fresh checkout, generated caches are absent: rebuild inputs once. Changes to cleaning, text features, `replacement.json`, split/seed, or shared encoding require rebuilding the split and downstream artifacts. Changes to an embedding require regenerating its frames and downstream classifiers. Existing files never update merely because code changed. Coordinate shared-data changes with the integration owner; do not independently change the dataset for one model comparison.

After editing an already-imported shared function, restart the kernel before rebuilding. Re-executing `from ... import ...` reuses the loaded module and can keep the old implementation in memory.

Use `.cache()` for reused frames. Preserve the current cache policy: do not introduce `StorageLevel`, `unpersist`, or `MEMORY_AND_DISK` as part of a model task. After editing/adding a registered module, restart the kernel and rerun. Do not reload only `base`: that can leave registries and already-imported modules out of sync. `importlib.reload` accepts modules, not classes; instances created before a reload retain the old class.

## Verification and handoff

Parse the edited Python and notebook cells, import the package to check registration, and run the assigned path on a small sample when dependencies permit. For an embedding, check that train and predict share a key. For an analytics model, check that its single registry key is callable once. Check vector dimension, preserved sample IDs, and the three prediction columns. Report what ran, the model/embedding pair, input version or seed, dependencies, and limitations. Do not report unrun experiments as completed.

Source data and fitted embedding models are intended to be versioned; the current directory migration must include them when committed. Generated `data/processed/` frames stay out of Git. A classifier does not need a saved model directory. Do not commit, push, or change Git config unless the user asks. Clear notebook outputs before committing. Keep English comments brief and factual, preserve public interfaces, and use snake_case for new local variables. Do not expand a text/comment cleanup into a modelling change.
