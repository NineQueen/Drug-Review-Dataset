# Interface

Predict `rating` as ten classes, 1–10. This is the contract for the current code. Edit only the assigned model file.

## Scope

An embedding lives in `src/embeddings/<name>.py`. It reads the encoded split, fits on the training rows, saves its model, and returns train, validation, and test frames with `embedding`.

An analytics model lives in `src/analytics/<name>.py`. It reads `embedding` and the encoded non-text columns, fits on the training frame, and returns three prediction frames from one registered function.

Shared preprocessing, splitting, encoding, base classes, I/O, evaluation, notebooks, and dependencies stay with the integrator.

## Data

Fit vocabulary, IDF, encoders, scalers, reducers, classifiers, thresholds, and selected feature subsets on training rows only. Validation and test are transformed. The reported comparison uses the test score.

`split` uses `pmod(xxhash64(uniqueID, seed), 10000)` with positive integer ratios, default `(7, 1, 2)`. `seed(0)` cuts the full table. `weight` attaches the class weight to the fitting split only: `n / (sqrt(class_count) * sum of sqrt(class_count))`, so the weights sum to `n`.

`days` is days since 1970-01-01. `usefulCount_log` is `log1p(usefulCount)`. `rating` stays 1–10. Classifiers train on `rating_m1 = rating - 1`, range 0–9.

Columns the models see:

- `uniqueID`, `review`, `words`, `words_nostop`, `words_featured`
- `condition_idx`, `condition_vec`, `days`, `usefulCount_log`, `drugName_idx`, `drugName_vec`
- `weight` on the training split only

Drug names that appear at least 10 times in training stay distinct. Rarer and unseen names become `__other__`. The encoder fits on the first frame with `handleInvalid="keep"`.

TF-IDF reads `words_featured`. Another embedding may read `review`, `words`, or `words_nostop`. An analytics function does not read those text columns. `fit_phrases(train_df)` fits on training `words` with `min_count=20`, `scoring="npmi"`, `threshold=0.5`, and `delimiter="_"`. GloVe trains on `train_expanded_phrase`.

## Registration

One module per model. Import the base class inside that module. Import must not train, download, write files, or create a Spark session. Import optional libraries inside the functions that use them.

A registry key is lowercase and may contain one underscore. `__` is rejected.

TF-IDF uses `baseline_train()` and `baseline_predict()` and is not in the embedding registries. Unweighted logistic regression is `baseline()`. Weighted logistic regression is `improved_baseline(para=None)`. Neither function is in `registry`. The shared runner calls `registry[key](instance)` and therefore uses the default `para`.

In `project.ipynb`, tokenizer-only TF-IDF is saved as `tfidf` and scored by `baseline()`. Engineered-text TF-IDF is saved as `tfidf_improved` and scored by `improved_baseline()`. Those prediction directories are `lr__tfidf` and `lr_improved__tfidf_improved`.

```python
@EmbeddingModel.train_register("my_embedding")
def my_embedding_train(self, model_name: str = "my_embedding") -> None:
    ...  # fit on self.train; save EMBEDDING_MODELS_DIR / model_name

@EmbeddingModel.predict_register("my_embedding")
def my_embedding_predict(self, model_name: str = "my_embedding") -> tuple[DataFrame, DataFrame, DataFrame]:
    ...  # load model_name; return train, validation, test

@AnalyticsModel.register("my_classifier")
def my_classifier(self, para: dict = DEFAULT_PARA) -> tuple[DataFrame, DataFrame, DataFrame]:
    ...  # fit on the training frame; return train, validation, test predictions
```

`DEFAULT_PARA` holds constructor values. `PARA_RANGE` holds candidate lists. The runner does not read `PARA_RANGE`.

## Embedding output

`predict` adds a Spark ML vector column `embedding`, then keeps `uniqueID`, `rating`, `rating_m1`, training `weight`, `condition_idx`, `condition_vec`, `days`, `usefulCount_log`, `drugName_idx`, `drugName_vec`, and `embedding`. Every row is kept, and the three frames share one dimension. `train` and `predict` for GloVe, Word2Vec, BERT, and TF-IDF accept `model_dir`. The default directory is `src/embeddings/models/<name>/`.

## Analytics output

Usable inputs are `embedding`, `condition_idx`, `condition_vec`, `days`, `usefulCount_log`, `drugName_idx`, and `drugName_vec`. The feature list is a module constant. Call `_feature_transform(features)` and use that same list for the fit and the returned predictions. Logistic regression and SVM include `drugName_vec`. Random forest omits it.

`improved_baseline()` forces `standardization=False`, L2-normalizes `embedding`, and replaces `days` with its training z-score. Its default `regParam` is `1e-5` and `maxIter` is 300. `baseline()` and the other classifiers keep the original `embedding` and `days`. TF-IDF `vocabSize` is 20000.

Each prediction frame has `uniqueID`, `rating`, `prediction`, and `probability`. `prediction` is an integer-valued double in 0–9. `probability` is an array of 10 doubles indexed by class. Logistic regression, random forest, XGBoost, and MLP store class probabilities. SVM stores ten one-vs-rest scores. `Scorer` uses `prediction + 1`. Scores are accuracy and support-weighted precision and F1, plus MAE and RMSE on the original 1–10 ratings.

The result name is `{analytics}__{embedding}`.

## Saved files

- `data/processed/split/{train,valid,test}.parquet`
- `data/processed/embeddings/<embedding>/{train,valid,test}.parquet`
- `data/processed/predictions/<analytics>__<embedding>/{train,valid,test}_predictions.parquet`
- `data/parameter/<embedding>_<analytics>.json`

`load_analytics_para` requires `weighted_f1` and returns constructor keys only. A saved score is replaced only when the new weighted F1 is higher.

## Stacking

`StackingModel` is not registered. `load_data()` builds 3-fold out-of-fold scores on `train_raw.parquet` with `seed(1)`. Each fold runs the registered embedding-classifier pairs, then `tfidf_improved` with `improved_baseline()`. That extra pair is `lr_improved__tfidf_improved`. The unweighted baseline is not included. Fold embeddings are written under `data/processed/stacking/models` and deleted after the fold.

`train(para=None)` concatenates the ten 10-d outputs in manifest order. Passed keys override `DEFAULT_PARA`; a missing key keeps `regParam=0.3`, `elasticNetParam=0.0`, `maxIter=100`, `tol=1e-7`, weight exponent `0.75`, and prior exponent `0.0`. It fits a standard scaler and a multinomial logistic regression on the training OOF only, with `standardization=False`. Class weights are `n_c ** -weight_exponent`. Decision thresholds are `prior ** prior_exponent`. The pipeline is saved to `src/stacking/model/`.

`predict()` reads the ten saved full-data prediction sets and writes `stacking__stacking`.

## Check before handoff

The assigned file is the only model file changed. Train and predict share one embedding key. An analytics key is one function. Import does not train. IDs and row counts are preserved. Validation and test do not fit anything. Split, embedding frames, and temp files under `data/processed/` stay out of Git. Prediction frames and stacking outputs are kept. Notebook outputs stay out of Git. Restart the kernel after adding or editing a registered module.
