# Coding-agent contract

Read this before editing. It applies to the whole repository. A model assignment may change only the assigned model file. An explicit request to maintain shared code or documentation overrides that boundary. Setup is in `readme.md`. Columns, registration, and outputs are in `docs/interface.md`. Keep those documents consistent with the code.

## Goal and ownership

Predict drug-review `rating` as ten classes, 1–10, with PySpark. The required baseline is unweighted logistic regression on TF-IDF. TF-IDF uses `baseline_train_register` and `baseline_predict_register`, is called as `baseline_train()` and `baseline_predict()`, and is not in the embedding registries. In `project.ipynb`, the tokenizer-only run is saved as `tfidf` and scored only by `baseline()`; the engineered-text run is saved as `tfidf_improved` and scored only by `improved_baseline()`. Those prediction directories are `lr__tfidf` and `lr_improved__tfidf_improved`. Other classifiers use `register` and a unique key, such as `random_forest` or `svm`. Advanced modelling needs N distinct non-LR classifiers, where N is the number of group members, each paired with a separate non-TF-IDF embedding. TF-IDF is not a word embedding, and neither logistic model counts toward N. One model is not both the embedding and the classifier. Advanced modelling and the final report are separate tasks.

- **Embedding owner:** only `src/embeddings/<name>.py`. Read the encoded split, fit the representation, save its model, and return train, validation, and test frames with `embedding`.
- **Analytics owner:** only `src/analytics/<name>.py`. Read `embedding` and the encoded non-text columns, fit on the training frame, and return the three prediction frames from one registered function.
- **Integration owner:** preprocessing, splitting, encoding, base classes, discovery, I/O, evaluation, shared notebooks, dependencies, documentation, and final assembly.

A model assignment does not include another model, `base.py`, `__init__.py`, `src/paths.py`, preprocessing, evaluation, or the main notebook. Put helpers in the assigned file. Report a needed shared change or new dependency instead of making it. A personal `para` dict changes constructor values only. Do not invent member names, assignments, or completed experiments.

## Registration

Add one module under the assigned folder. Import the base class from `src.embeddings.base` or `src.analytics.base` inside the model module. Notebooks may import `src.embeddings` and `src.analytics`. Import discovers sibling modules. Do not edit `__init__.py`. Import must not train, download, write files, or create a Spark session. Import optional libraries inside the functions that use them.

A registry key is lowercase and may contain one underscore. `__`, path separators, and duplicate keys are rejected. An embedding registers train and predict on the same key, with names such as `<name>_train` and `<name>_predict`. An analytics model registers one function. The shared runner calls `registry[key](instance)` and therefore uses the default `para`. `baseline()` takes no parameter dict. `improved_baseline(para=None)` fills missing keys from `DEFAULT_PARA`. Neither function is in `registry`.

`train` fits on the training split and saves `EMBEDDING_MODELS_DIR / model_name`. `predict` loads that directory and returns train, validation, then test. An analytics function fits and returns the three prediction frames in the same call. Its result name is `{registry_key}__{self.embedding_model}`.

```python
@EmbeddingModel.train_register("my_embedding")
def my_embedding_train(self, model_name: str = "my_embedding") -> None:
    ...  # fit on self.train; save EMBEDDING_MODELS_DIR / model_name

@EmbeddingModel.predict_register("my_embedding")
def my_embedding_predict(self, model_name: str = "my_embedding") -> tuple[DataFrame, DataFrame, DataFrame]:
    ...  # load model_name; return train, validation, test

DEFAULT_PARA = {"regParam": 0.1, "maxIter": 100}
PARA_RANGE = {"regParam": [0.01, 0.1, 0.3, 1.0], "maxIter": [50, 100]}

@AnalyticsModel.register("my_classifier")
def my_classifier(self, para: dict = DEFAULT_PARA) -> tuple[DataFrame, DataFrame, DataFrame]:
    ...  # fit on the training frame with **para; return train, validation, test predictions
```

`DEFAULT_PARA` holds constructor values. `PARA_RANGE` holds candidate lists. The runner does not read `PARA_RANGE`. Neither dict is keyed by embedding name, and neither contains feature lists, labels, paths, or pipeline switches. Do not mutate `para`. Keep an embedding's train and predict separate. An analytics model is one function. The shared loops do not special-case a model. Every embedding uses that function and the same default parameters.

## Data contract

- Fit vocabulary, IDF, encoders, scalers, reducers, classifiers, thresholds, class offsets, and selected feature subsets on training rows only. Inside an analytics function, validation and test are only transformed. Scores are computed after the function returns. The reported comparison uses the test score. A pretrained embedding may be loaded; any dataset-specific fit still uses training rows only.
- Preserve `uniqueID`, rows, and labels. Embedding `predict` adds a Spark ML `VectorUDT` column `embedding`, with one dimension for every row, then returns `uniqueID`, `rating`, `rating_p1`, `weight` on the training split, `condition_idx`, `condition_vec`, `days`, `usefulCount_log`, `drugName_idx`, `drugName_vec`, and `embedding`.
- Inputs keep `review`, `words`, `words_nostop`, and `words_featured`. TF-IDF reads `words_featured`. Another embedding may read `review`, `words`, or `words_nostop`. An analytics function does not read those columns or fit another vectorizer.
- `fit_phrases(train_df)` fits on the training `words` column and returns a frozen model. It takes no parameter dict. The fixed arguments are `min_count=20`, `scoring="npmi"`, `threshold=0.5`, and `delimiter="_"`. Pass that object to `EmbeddingModel`. `apply_phrases` builds `train_phrase`, `valid_phrase`, and `test_phrase` from `words_nostop`. `train_expanded` unions the training word columns with WebMD reviews whose `review` is absent from validation and test. `train_expanded_phrase` applies the same rules and requires `phrases`. GloVe trains on `train_expanded_phrase` and predicts on the three phrase frames. TF-IDF reads `self.train`.
- `data_cleaning` keeps the existing row filter, configured condition and HTML replacements, paired outer-wrapper removal, whitespace cleanup, lowercasing, deterministic review deduplication, and numeric columns. Remove a wrapper only when the same marker is at both ends: one `"` pair, then one `&quot;` pair. Collapse ordinary whitespace before that check and again after HTML decoding. `text_feature_engineering` writes character and Unicode whitespace normalization, contraction expansion, and dose spacing back to `review`, then tokenizes a temporary copy. Tokenization, stop-word removal, and stemming do not overwrite `review`. Keep `not`, `no`, `nor`, `never`, `neither`, `without`, `cannot`, `but`, `only`, `very`, `too`, `more`, and `most`. Return `review`, `words`, `words_nostop`, and `words_featured`. The main notebook enables bigrams. `baseline_text_feature_engineering` only tokenizes `review` for the separate TF-IDF path. The text stage does not drop rows or deduplicate again. New text repairs must be general patterns. Do not add a dataset-specific typo list or per-review exceptions. Check normalization collisions separately before comparing models.
- Classifiers train on `rating_p1 = rating - 1`, range 0–9. Return `prediction` as an integer-valued double in that range. Only `Scorer` adds 1. Analytics output is exactly `uniqueID`, `rating`, `prediction`.
- Analytics reads `self.train_embedding`, `self.valid_embedding`, and `self.test_embedding`. Usable inputs are `embedding`, `condition_idx`, `condition_vec`, `days`, `usefulCount_log`, `drugName_idx`, and `drugName_vec`. The feature list is a module constant. `_feature_transform(features)` returns three frames with `features` and does not mutate the stored inputs. Use that same list for the fit and the returned predictions. Do not use `rating`, `rating_p1`, `weight`, or a value computed from the labels. Logistic regression and SVM include `drugName_vec`. Random forest omits it.
- The encoder fits `condition_idx`, `condition_vec`, and the drug one-hot on the first frame only, with `handleInvalid="keep"`, then transforms later frames. Later frames are optional. Drug names that appear at least 10 times in training stay distinct; rarer and unseen names become `__other__`. It does not add `drug_mean_rating`. Rebuild downstream artifacts after a shared encoding change.
- `days` is days since 1970-01-01. `usefulCount_log` is `log1p(usefulCount)`. The training-only `weight` is `n / (k * class_count)`.
- Splitting uses `pmod(xxhash64(uniqueID, seed), 10000)`. `split` takes positive integer ratios, one per frame, for example `(8, 2)` or `(7, 1, 2)`. The default `(7, 1, 2)` puts buckets below 7000 in train, 7000–7999 in validation, and 8000 and above in test. Any remainder goes to the last part. `weight` attaches the class weight to one frame, caches it, and releases the input cache. Call it on the fitting split only. Do not replace this hash split with `randomSplit`.
- `KFold` in `src/k_flod.py` calls `split` with `(1,) * k`, runs `prepare_function` on each fit/holdout pair, and writes `data/processed/temp/<model_name>/fold_<i>/{train,valid}.parquet`. Each fold's new caches sit in `data_oi.scope()` and are released after the write. `scope()` drops DataFrame caches created inside the block, including their SQL registration. `next_fold()` loads the next pair and releases the previous one. `reset()` returns to the first fold. `KFold.open(model_name, k)` reads an existing directory. If `df` was already cut with the same seed, pass a different `SEED`. The tuning notebook uses `SEED + 1` on the training pool. Its embedding and classifier both receive one row of the fit frame as the unused third frame.

## Artifacts and evaluation

- Raw CSV: `data/raw/drugsComTrain_raw.csv`.
- Encoded split: `data/processed/split/{train,valid,test}.parquet`.
- Embedding frames: `data/processed/embeddings/<embedding>/{train,valid,test}.parquet`.
- Prediction frames: `data/processed/predictions/<analytics>__<embedding>/{train_predictions,valid_predictions,test_predictions}.parquet`.
- K-fold frames: `data/processed/temp/<model_name>/fold_<i>/{train,valid}.parquet`.
- Fitted representation: `src/embeddings/models/<embedding>/`.
- Analytics deliverable: the three prediction frames. The registered function does not save a classifier.

Use path constants from `src/paths.py`. Create only the parent of a Spark embedding-model directory before `write().overwrite().save()`. The runner writes DataFrames through `DataOI`. Do not overwrite another model's artifacts. Assume the runner supplied the documented frames and `para`. Do not add branches for missing columns or alternate pipelines.

Keep `Embedding_Model`, `Analytic_Model`, `rating_p1`, and the class name `DataOI`. Scoring uses accuracy and support-weighted precision and F1. Support-weighted recall equals accuracy for this single-label task, so it is omitted. MAE and RMSE are errors on the original 1–10 ratings. In `Scorer.add_model`, keep the `|` between the two nonmatching-name conditions so only an exact pair is replaced.

## Running, caching, and imports

The entry notebook creates one Spark session before `DataOI`. `project.ipynb` uses `local[4]` and a `16g` driver. Experiment notebooks set their own master and driver memory in the first cell. Do not start a session in a model module or in `DataOI`. Restart the kernel after changing driver memory or after the JVM exits.

`project.ipynb` is the integration runner. `experiments/embedding_pipeline.ipynb` and `experiments/analytics_pipeline.ipynb` are the personal templates. TF-IDF stays in the main baseline block. The analytics runner iterates `EmbeddingModel.predict_registry` and loads each saved frame by registry name. `experiments/para_tuning/para_tuning_pipeline.ipynb` resolves the written names with `getattr` and tunes one pair with `KFold`. Fold vectors go under `data/processed/temp/` and do not replace the shared split. Personal runners score that run. The reported comparison uses the test predictions.

`REBUILD` controls input preparation in the experiment notebooks, not training. `project.ipynb` uses `RUN`: `rebuild` prepares inputs, and `train_embedding`, `predict_embedding`, `baseline`, and `analytics` each turn on one later stage. A later stage reads saved files, so an earlier key can stay off. A fresh checkout has no `data/processed/` cache, so rebuild inputs once. Changes to cleaning, text features, `replacement.json`, the split, or shared encoding require rebuilding the split and everything downstream. An embedding change requires regenerating its frames and the classifiers that read them. Existing files do not update because the code changed.

After editing an imported shared function, restart the kernel before rebuilding. Use `.cache()` for reused frames. Do not introduce `StorageLevel`, `unpersist`, or `MEMORY_AND_DISK` in a model file. After adding a registered module, restart the kernel. Do not reload only `base`. `importlib.reload` accepts modules, and instances created earlier keep the old class.

## Verification and handoff

Parse the edited Python and notebook cells, import the package to check registration, and run the assigned path on a small sample when dependencies permit. An embedding's train and predict share a key. An analytics key is callable once. Check the vector dimension, preserved IDs, and the three prediction columns. Report what ran, the model pair, seed, dependencies, and limitations. Do not report an unrun experiment as completed.

Source data and fitted embedding models are versioned. Generated `data/processed/` frames stay out of Git. A classifier does not need a saved model directory. Do not commit, push, or change Git config unless asked. Clear notebook outputs before committing. Keep comments brief, preserve public interfaces, and use snake_case for new local variables. A comment cleanup does not become a modelling change.
