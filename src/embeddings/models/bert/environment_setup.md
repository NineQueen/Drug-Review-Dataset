# BERT setup

Use Python 3.12, PySpark 4.2.0, and Java 21. `requirements.txt` already includes the BERT libraries. `experiments/bert_requirements.txt` includes that file, repeats the BERT pins, and adds `ipykernel`. Install a CUDA wheel before that file when the machine has a compatible GPU:

```bash
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/bert_requirements.txt
python -m pip check
```

CPU environments can install `requirements.txt` directly. The notebook kernel and `PYSPARK_PYTHON` use the same Python. Set `JAVA_HOME` to Java 21. If `SPARK_HOME` is set, point it at that environment's PySpark. Windows also needs the Hadoop native tools.

`experiments/bert_pipeline.ipynb` uses `local[*]` and a `4g` driver. It starts with `REBUILD=True`, `MODEL_NAME="bert"`, `FIT_ENCODER=True`, and `RUN_TEAM_ANALYTICS=False`. For a new training run, choose a fresh `MODEL_NAME`, such as `bert_final`, and keep the registry key `bert`. After the model changes, regenerate its embedding frames and downstream predictions. `RESUME=True` applies only to an interrupted run whose local checkpoints are still present. A completed model does not need optimizer checkpoints.
