# BERT setup

Use Python 3.12, PySpark 4.2.0, and Java 21. `requirements.txt` already includes the BERT libraries. `experiments/bert_requirements.txt` includes that file, repeats the BERT pins, and adds `ipykernel`. On a compatible GPU, install the CUDA wheel first:

```bash
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/bert_requirements.txt
python -m pip check
```

A CPU environment can install `requirements.txt` directly. The notebook kernel and `PYSPARK_PYTHON` must use that same Python. Set `JAVA_HOME` to Java 21. If `SPARK_HOME` is set, point it at this environment's PySpark. Windows also needs the Hadoop native tools.

`experiments/bert_pipeline.ipynb` uses `local[4]` and an `8g` driver. It starts with `REBUILD=True`, `MODEL_NAME="bert"`, `FIT_ENCODER=True`, and `RUN_TEAM_ANALYTICS=False`. For a new training run, pick a fresh `MODEL_NAME`, such as `bert_final`, and keep the registry key `bert`. After the model changes, regenerate its embedding frames and the predictions that read them. `RESUME=True` continues an interrupted run whose local checkpoints are still present.

Prediction rejects a model directory without a valid `completion.json`. Finish or resume that run before exporting vectors.
