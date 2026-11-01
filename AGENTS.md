# Coding-agent contract

Read this file before editing. It applies to the whole repository. The user's current task determines which files you may change; the boundaries below are the default for a model assignment. See `readme.md` for setup and `docs/interface.md` for the full interface. Keep these documents consistent with the implementation.

## Goal and ownership

Predict the original drug-review `rating` as ten classes, 1–10, with PySpark. The required baseline is unweighted `lr` with `tfidf`; preserve both registry keys. N is the number of group members: advanced modelling needs N distinct non-LR classifiers paired with separate non-TF-IDF embeddings. TF-IDF is not a word embedding, and neither LR nor weighted LR counts toward N. Do not use one model as both the embedding and classifier. Advanced modelling and final report assembly are separate tasks.

- **Embedding owner:** implement only the assigned `src/embeddings/<name>.py`. Read the encoded split through the existing runner, fit the representation, save its own model, and return train/test frames containing `embedding`.
- **Analytics owner:** implement only the assigned `src/analytics/<name>.py`. Consume embedding frames, choose and assemble its own features, fit its classifier, save its own model, and return train/test predictions.
- **Integration owner:** maintain preprocessing, splitting, encoding, base classes, registration discovery, I/O, evaluation, shared notebooks, dependencies, and documentation. Assemble final deliverables.

A model assignment does not authorize editing another model, `base.py`, `__init__.py`, `src/paths.py`, preprocessing, evaluation, or the main notebook. Implement helper functions and model-specific transforms in the assigned model file. If a shared change or new dependency is necessary, report the concrete need to the integration owner instead of silently expanding the task. An explicit user request to maintain shared code or documentation overrides these default boundaries.

Run parameters in a personal experiment may be adjusted without changing its pipeline structure. Do not require another member to modify shared code to register or use your model. Do not invent member names, model assignments, or completed experiments.

## Registration

Add one model module directly under its assigned folder. Import the base class from `src.embeddings.base` or `src.analytics.base` **inside model modules**, not from the package entry point: discovery runs while the base module is being imported. Notebooks may import from `src.embeddings` and `src.analytics`.

Importing either package discovers sibling modules automatically. Do not edit `__init__.py` or discovery loops. Keep module import free of training, downloads, file writes, and Spark-session creation. Put optional model-specific imports inside the functions that need them so another model can still be imported without that dependency.

Use a unique lowercase registry key. A single underscore is allowed; `__`, path separators, and duplicate keys are not. Register both train and predict with the same key. Use distinct function names such as `<name>_train` and `<name>_predict`.

```python
# Embedding module: self.train and self.test are Spark DataFrames.
@EmbeddingModel.train_register("my_embedding")
def my_embedding_train(self) -> None:
    ...

@EmbeddingModel.predict_register("my_embedding")
def my_embedding_predict(self) -> tuple[DataFrame, DataFrame]:
    ...  # Return train, then test.

# Analytics module: embedding_model is the embedding registry key, a string.
@AnalyticsModel.train_register("my_classifier")
def my_classifier_train(self, embedding_model: str) -> None:
    ...

@AnalyticsModel.predict_register("my_classifier")
def my_classifier_predict(self, embedding_model: str) -> tuple[DataFrame, DataFrame]:
    ...  # Return train predictions, then test predictions.
```

These are interface sketches; implement the algorithm before running them. The registry stores an unbound function: the runner calls `registry[key](instance, ...)`. Keep train and predict separate. The integration runner trains all selected entries before calling their predict entries; do not special-case your model in its loops.

## Data contract

- Fit vocabulary, IDF, encoders, scalers, reducers, and classifiers on training data only. Do not tune on the test split. A pretrained embedding may be loaded, but any dataset-specific fitting uses training rows only.
- Preserve `uniqueID`, rows, and original labels. Embeddings keep all existing columns and add the Spark ML `VectorUDT` column `embedding`, with the same dimension for every train/test row. Do not return a Python list or Spark array in place of the vector.
- Inputs retain `review`, `words`, and `words_stem`. TF-IDF uses `words_stem`; a model may instead use `review` or unstemmed `words` without changing shared text processing.
- Keep the preprocessing boundary fixed: `data_cleaning` retains the original record filtering, configured condition/HTML replacements, outer-quote trimming, ordinary-whitespace cleanup, lowercasing, deterministic review deduplication, and numeric derived columns. Additional shared text repairs belong to `text_feature_engineering` and must update `review` with the repaired full text. As further character, whitespace, contraction, or dose-format repairs are implemented, apply them before creating the temporary `review_clean` copy for token preparation. That copy is for token-specific processing, which must not overwrite the repaired `review`. The current repair implementation contains only the migrated character/whitespace function; the remaining repairs are pending in `docs/todo.md`. Return `review`, `words`, and `words_stem`, without adding a permanent text column. The text stage preserves rows, IDs, and labels and does not deduplicate again. Audit normalization collisions separately before model comparison.
- New text repairs must address broadly applicable patterns. Do not build dataset-specific typo dictionaries, add per-review exceptions, or require exhaustive manual correction of the dataset. Preserve ambiguous expressions rather than guessing; aggregate checks and representative examples validate the supported rules.
- Classifiers train on `rating_p1 = rating - 1`, range 0–9. Return `prediction` as an integer-valued double in that range, not probabilities or original 1–10 ratings. Only `Scorer` adds 1. Analytics outputs contain exactly `uniqueID`, `rating`, `prediction`.
- Analytics receives `self.train_embedding` and `self.test_embedding`. `self._feature_transform(features)` returns train/test frames with an assembled `features` column; it does not mutate the stored inputs. Use the same feature list in train and predict. Do not use `rating`, `rating_p1`, or `weight` as input features.
- `condition_vec` is one-hot; `drug_mean_rating` is the smoothed training mean encoding of the drug. `drugName_idx` remains an auxiliary categorical index, not an ordinal numeric feature. The existing encoder uses `targetType="continuous"`, `smoothing=10.0`, and keeps unseen values. Do not change this shared encoding as part of a model assignment.
- `days` is days since 1970-01-01. `usefulCount_log` is `log1p(usefulCount)`, where useful votes accumulate after review publication. The training-only `weight` is `n / (k * class_count)`; pass it as a model weight when appropriate, never require it on test rows.
- Splitting uses `xxhash64(uniqueID, SEED)` buckets, approximately 80/20. With the same data, ID types, Spark version, and seed, membership is independent of row order and partitioning. Keep this shared policy; do not replace it with `randomSplit`.

## Artifacts and evaluation

- Raw CSV: `data/raw/drugsComTrain_raw.csv`.
- Encoded split: `data/processed/split/{train,test}.parquet`.
- Embedding frames: `data/processed/embeddings/<embedding>/{train,test}.parquet`.
- Prediction frames: `data/processed/predictions/<analytics>__<embedding>/{train_predictions,test_predictions}.parquet`.
- Fitted representation: `src/embeddings/models/<embedding>/`.
- Fitted classifier: `src/analytics/models/<analytics>__<embedding>/`.

Use path constants from `src/paths.py`. Create only a Spark model directory's parent before `write().overwrite().save()`. A model owner writes only their own fitted-model directory; the runner writes DataFrames through `DataOI`. Do not overwrite another model's artifacts.

Combined names use `analytics__embedding` in that order. Keep public column names `Embedding_Model`, `Analytic_Model`, `rating_p1`, and class name `DataOI` unchanged. Scoring uses accuracy and support-weighted precision/recall/F1; MAE/RMSE are supplementary errors on original rating units, not a separate regression task. In `Scorer.add_model`, keep the `|` between the two nonmatching-name conditions: only an exact model pair is replaced.

## Running, caching, and imports

The entry notebook creates one active SparkSession before constructing `DataOI`. Use the existing `local[*]` / 4g setup. Do not start a session in a model module or in `DataOI`. Restart the kernel after changing driver memory or after the JVM exits.

`project.ipynb` is the integration runner. Personal runners are `experiments/embedding_pipeline.ipynb` and `experiments/analytics_pipeline.ipynb`; select `embedding_name = "tfidf"` in the embedding runner to check the baseline representation. Choose the assigned registry key in the runner. Personal scoring uses only that run's `test_pred_dict`; the main runner may aggregate saved results.

`REBUILD` controls input preparation, not whether a model trains. On a fresh checkout, generated caches are absent: rebuild inputs once. Changes to cleaning, text features, `replacement.json`, split/seed, or shared encoding require rebuilding the split and downstream artifacts. Changes to an embedding require regenerating its frames and downstream classifiers. Existing files never update merely because code changed. Coordinate shared-data changes with the integration owner; do not independently change the dataset for one model comparison.

After editing an already-imported shared function, restart the kernel before rebuilding. Re-executing `from ... import ...` reuses the loaded module and can keep the old implementation in memory.

Use `.cache()` for reused frames. Preserve the current cache policy: do not introduce `StorageLevel`, `unpersist`, or `MEMORY_AND_DISK` as part of a model task. After editing/adding a registered module, restart the kernel and rerun. Do not reload only `base`: that can leave registries and already-imported modules out of sync. `importlib.reload` accepts modules, not classes; instances created before a reload retain the old class.

## Verification and handoff

Parse the edited Python and notebook cells, import the package to check registration, and run the assigned train/predict path on a small sample when dependencies permit. Check matching train/predict keys, vector dimension, preserved sample IDs, and the three prediction columns. Report what ran, the model/embedding pair, input version or seed, dependencies, and limitations. Do not report unrun experiments as completed.

Source data and fitted models are intended to be versioned; the current directory migration must include them when committed. Generated `data/processed/` frames stay out of Git. Do not commit, push, or change Git config unless the user asks. Clear notebook outputs before committing. Keep English comments brief and factual, preserve public interfaces, and use snake_case for new local variables. Do not expand a text/comment cleanup into a modelling change.
