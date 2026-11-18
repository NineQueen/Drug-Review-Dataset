# DSAI4205 Big Data Analytics — UCI Drug Review

Predict drug-review `rating` as ten classes, 1–10, with PySpark. Shared code is in `src/`. The final project notebook is `project.ipynb`. The required baseline is TF-IDF with unweighted logistic regression; a weighted logistic regression is trained separately. Columns, registration, and outputs are in [docs/interface.md](docs/interface.md).

## Setup

Python 3.12, PySpark 4.2.0, and Java 21. Point `JAVA_HOME` at Java 21. `DataEnv` is only the local environment name. The Jupyter kernel must be that same Python.

```bash
conda activate DataEnv
python -m pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger_eng wordnet
```

Create the environment with `conda create -n DataEnv python=3.12` if it is missing, and install `ipykernel` there if the editor asks for it.

The delivered Word2Vec model uses NLTK 3.9.3 and checks the noun-lemmatizer resource fingerprint. Run the download once on this machine. Spark Python workers must be able to read the same resources. After an NLTK upgrade or a resource change, retrain Word2Vec.

Raw data is `data/raw/drugsComTrain_raw.csv`. Replacements are in `replacement.json`. Raw data and fitted embeddings are part of the repository. Split, embedding frames, and temp files under `data/processed/` stay out of Git. Prediction frames and stacking outputs are kept. Classifiers are not saved as model files. Start notebooks from the repository root. Imports also walk upward to `src/` and `project.ipynb`, so a notebook started in `experiments/` can still import the shared package.

`project.ipynb` uses `local[4]` and a `16g` driver. Experiment notebooks set their own master and memory in the first cell. The entry notebook creates the Spark session once. Restart the kernel after changing driver memory, after a `Connection refused` JVM exit, or after editing an imported `src/` function.

## What to edit

Each member edits one assigned model file. Shared preprocessing, base classes, and the main notebook stay with the integrator.

- Embedding: `src/embeddings/<name>.py`. Write a vector column named `embedding`.
- Analytics: `src/analytics/<name>.py`. Read `embedding` and the encoded non-text columns, fit on the training frame, and return three prediction frames. The file defines `DEFAULT_PARA` and `PARA_RANGE`.
- Integrator: shared pipeline, dependencies, documentation, and the final assembly.

Register a model with its decorator and select that name in the existing experiment notebook. Import optional libraries inside the functions that use them, and add new dependencies to `requirements.txt`.

## Entry points

`project.ipynb` uses `RUN`:

- `rebuild` prepares EDA, cleaning, text features, and the encoded split.
- `train_embedding` fits `train_registry` and saves the models.
- `predict_embedding` loads those models and writes vectors.
- `baseline` runs tokenizer-only TF-IDF (`tfidf` / `baseline()` / `lr__tfidf`) and engineered-text TF-IDF (`tfidf_improved` / `improved_baseline()` / `lr_improved__tfidf_improved`).
- `analytics` runs every classifier in `registry` on the saved vectors. TF-IDF is outside both loops.
- `load_stacking`, `train_stacking`, and `predict_stacking` build out-of-fold scores, fit the second-level model, and write `stacking__stacking`.

A later stage reads saved files, so an earlier switch can stay off. Call `fit_phrases(df_train)` before constructing `EmbeddingModel` when the representation needs phrases.

Other entries:

- `experiments/embedding_pipeline.ipynb` trains one name from `train_registry`, then every classifier in `registry`.
- `experiments/analytics_pipeline.ipynb` runs one `register` name across `predict_registry`. `REBUILD = True` trains those embeddings; `False` loads saved vectors.
- `experiments/para_tuning/para_tuning_pipeline.ipynb` tunes the written `EMBEDDING_MODEL` and `ANALYTICS_MODEL` with k-fold. Fold vectors go under `data/processed/temp/` and do not replace the shared split.

`load_prediction_data()` reads every saved prediction. Personal notebooks pass `model_names`. The reported comparison uses the test score.

## Text and cache

`data_cleaning()` filters rows, applies the configured replacements, removes one paired outer quote, normalizes whitespace, lowercases, deduplicates reviews, and adds `days`, `usefulCount_log`, and `rating_m1`. `text_feature_engineering()` writes whitespace, contraction, and dose repairs back to `review`, then tokenizes a temporary copy and keeps negation and degree words. The main notebook enables bigrams. `baseline_text_feature_engineering()` only tokenizes `review` for the `tfidf` path. `fit_phrases(train_df)` fits phrases on the training `words` column. Details are in [docs/interface.md](docs/interface.md).

`REBUILD` in an experiment notebook only rebuilds that notebook's inputs. In `project.ipynb`, `rebuild` prepares inputs and the other `RUN` keys turn later stages on or off. A fresh checkout has no `data/processed/` cache, so rebuild once. Changes to cleaning, text features, `replacement.json`, the seed, or shared encoding require a new split and everything downstream. An embedding change requires new vectors and new predictions from the classifiers that read them. A classifier change requires new predictions for that model.

## Layout

```text
project.ipynb                     final project notebook
src/cleaning.py                   cleaning
src/text_features.py              text features and phrases
src/split.py                      hash split and training weights
src/encoder.py                    condition and drug encoding
src/embeddings/<name>.py          embeddings
src/analytics/<name>.py           classifiers
src/stacking/stacking.py          second-level model
src/evaluation.py                 scoring
src/dataOI.py                     read, write, and cache release
experiments/                      single-model notebooks
src/embeddings/models/            fitted representations
data/processed/                   local split, vectors, predictions, and OOF
```

Commit the model implementation, fitted embeddings, prediction frames, stacking outputs, and agreed dependency changes. Do not commit classifier files, split, embedding frames, temp caches, or notebook outputs.
