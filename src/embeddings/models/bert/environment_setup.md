# BERT setup and execution

Use Python 3.12, PySpark 4.2.0 and Java 21, matching the shared README. `experiments/bert_requirements.txt` inherits the shared dependencies and adds BERT's libraries.

For CUDA 12.8-compatible GPUs:

```bash
python -m pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/bert_requirements.txt
python -m pip check
```

Choose the PyTorch wheel appropriate for the machine. CPU environments can install the dependency file directly. A uv-managed environment can use `uv pip install --python <python-path> ...` and `uv pip check --python <python-path>`.

The notebook kernel and `PYSPARK_PYTHON` must use the same Python. Set `JAVA_HOME` to Java 21. If `SPARK_HOME` is set, point it at that environment's PySpark installation. Windows also needs Hadoop native tools configured.

Open `experiments/bert_pipeline.ipynb` inside the repository. It defaults to rebuilding shared inputs from the raw CSV, then checking whether the included fitted BERT can be reused. For cached inputs, set `REBUILD=False`. Source data, shared preprocessing and seed must match the saved input fingerprints.

For the group's later training run, choose a fresh `MODEL_NAME`, for example `bert_final`, and keep the registry key `bert`. This prevents overwriting the existing model. Set `FIT_ENCODER=True` to train. After a model changes, regenerate its embedding frames and downstream predictions. The shared notebooks can also use the registered `bert` train/predict functions directly.

The fitted package includes encoder, tokenizer, projection, training parameters, input fingerprints and a completion marker. Inference and completed-run reuse do not require optimizer checkpoints or experiment records. `RESUME=True` applies only to an interrupted run whose local checkpoints remain available.
