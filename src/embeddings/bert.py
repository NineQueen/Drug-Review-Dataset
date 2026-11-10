"""Small BERT trained from random weights on patient reviews.

Dependencies extend the shared environment in experiments/bert_requirements.txt:
    torch==2.9.1  transformers==4.57.6  accelerate==1.15.0
    tokenizers==0.22.2  safetensors  pyarrow>=18  scikit-learn  matplotlib
Use a CUDA-enabled PyTorch wheel supporting the local GPU. No model or
tokenizer is downloaded. The registry works without importing these libraries.

The personal runner is experiments/bert_pipeline.ipynb. Train/predict also work
through the unchanged shared runner with embedding_name = "bert".
"""

from dataclasses import asdict, dataclass
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
import html
import importlib.metadata
import json
import re
import time
import unicodedata

from src.embeddings.base import EmbeddingModel
from src.paths import EMBEDDING_MODELS_DIR


DEFAULT_POOLING = "layers_concat"
DEFAULT_WHITEN_DIMENSION = 512


@dataclass
class BertTrainingConfig:
    seed: int = 42
    vocab_size: int = 8000
    max_length: int = 256
    hidden_size: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    intermediate_size: int = 1024
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = 6000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    mlm_probability: float = 0.15
    eval_every: int = 500
    eval_max_rows: int = 2048
    mlm_holdout_fraction: float = 0.1
    patience: int = 4
    device: str = "auto"

    def validate(self):
        for key in ("vocab_size", "max_length", "hidden_size", "num_hidden_layers",
                    "num_attention_heads", "intermediate_size", "batch_size",
                    "gradient_accumulation_steps", "max_steps", "eval_every",
                    "eval_max_rows", "patience"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.vocab_size < 32 or self.max_length < 8:
            raise ValueError("Use vocab_size >= 32 and max_length >= 8")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if not 0 < self.mlm_probability <= 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Invalid masking probability or warmup ratio")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid learning rate or weight decay")
        if not 0 < self.mlm_holdout_fraction < 1:
            raise ValueError("mlm_holdout_fraction must be between 0 and 1")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")


def _dependencies():
    try:
        import torch
        import transformers
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    except ImportError as exc:
        raise ImportError("BERT needs CUDA-enabled torch, transformers<5, "
                          "accelerate, and tokenizers. See bert.py's docstring.") from exc
    return torch, transformers, (Tokenizer, models, normalizers, pre_tokenizers, trainers)


def _model_directory(model_name):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", model_name) or "__" in model_name:
        raise ValueError("Use a lowercase model_name without __ or path separators")
    return EMBEDDING_MODELS_DIR / model_name


def _text_key(text):
    text = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _read_rows(frame, ordered=True):
    selected = frame.select("uniqueID", "review")
    if ordered:
        selected = selected.orderBy("uniqueID")
    rows = list(selected.toLocalIterator())
    ids = [row.uniqueID for row in rows]
    if any(value is None for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("BERT requires non-null, unique uniqueID values")
    return ids, [row.review or "" for row in rows]


def _corpus_digest(texts):
    digest = sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _split_training_rows(ids, texts, config):
    """Reserve normalized-text groups from training for MLM model selection."""
    keys = [_text_key(text) for text in texts]
    groups = sorted(set(keys), key=lambda key: sha256(
        f"{config.seed}:mlm_holdout:{key}".encode("utf-8")).digest())
    if len(groups) < 2:
        raise ValueError("BERT needs at least two distinct training text groups")
    count = max(1, int(len(groups) * config.mlm_holdout_fraction))
    held_out = set(groups[:count])
    fit_ids, fit_texts, holdout_ids, holdout_texts = [], [], [], []
    for row_id, text, key in zip(ids, texts, keys):
        if key in held_out:
            holdout_ids.append(row_id)
            holdout_texts.append(text)
        else:
            fit_ids.append(row_id)
            fit_texts.append(text)
    return fit_ids, fit_texts, holdout_ids, holdout_texts


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _train_tokenizer(texts, config, directory):
    _, transformers, stack = _dependencies()
    Tokenizer, models, normalizers, pre_tokenizers, trainers = stack
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(
        vocab_size=config.vocab_size, min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))
    # BertTokenizerFast supplies the standard CLS/SEP template and token types.
    vocab_path = Path(tokenizer.model.save(str(directory))[0])
    result = transformers.BertTokenizerFast(
        vocab_file=str(vocab_path), do_lower_case=True, model_max_length=config.max_length,
    )
    result.save_pretrained(str(directory))
    return result


def _tokenize(texts, tokenizer, max_length):
    result = []
    full_lengths = []
    for start in range(0, len(texts), 1024):
        batch = texts[start:start + 1024]
        encoded = tokenizer(batch, truncation=False, return_special_tokens_mask=True,
                            verbose=False)
        for i, token_ids in enumerate(encoded["input_ids"]):
            full_lengths.append(len(token_ids))
            # Keep the closing SEP token when truncating a long review.
            item = {key: value[i][:max_length] for key, value in encoded.items()}
            if len(token_ids) > max_length:
                item["input_ids"][-1] = tokenizer.sep_token_id
                item["special_tokens_mask"][-1] = 1
            result.append(item)
    stats = {
        "rows": len(texts), "tokens_before_truncation": sum(full_lengths),
        "mean_tokens": sum(full_lengths) / max(1, len(full_lengths)),
        "truncation_rate": sum(n > max_length for n in full_lengths) / max(1, len(full_lengths)),
    }
    return result, stats


def _mask_features(features, tokenizer, probability, generator=None):
    import torch

    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    special = batch.pop("special_tokens_mask").bool()
    eligible = batch["attention_mask"].bool() & ~special
    selected = (torch.rand(eligible.shape, generator=generator) < probability) & eligible
    # At least one target per nonempty review prevents NaN loss on small batches.
    for i in range(len(features)):
        if eligible[i].any() and not selected[i].any():
            positions = eligible[i].nonzero().flatten()
            index = torch.randint(len(positions), (1,), generator=generator).item()
            selected[i, positions[index]] = True
    labels = batch["input_ids"].clone()
    labels[~selected] = -100
    choice = torch.rand(selected.shape, generator=generator)
    batch["input_ids"][selected & (choice < 0.8)] = tokenizer.mask_token_id
    random_ids = torch.randint(len(tokenizer), selected.shape, generator=generator)
    replace_random = selected & (choice >= 0.8) & (choice < 0.9)
    batch["input_ids"][replace_random] = random_ids[replace_random]
    batch["labels"] = labels
    return batch


class _MlmCollator:
    def __init__(self, tokenizer, probability):
        self.tokenizer = tokenizer
        self.probability = probability

    def __call__(self, features):
        import torch

        if "labels" not in features[0]:
            return _mask_features(features, self.tokenizer, self.probability)
        # Evaluation masks are materialized once, independently of training RNG.
        labels = [feature["labels"] for feature in features]
        inputs = [{k: v for k, v in feature.items() if k != "labels"} for feature in features]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        width = batch["input_ids"].shape[1]
        batch["labels"] = torch.tensor([row + [-100] * (width - len(row)) for row in labels])
        return batch


def _fixed_eval(features, tokenizer, config):
    import random
    import torch

    indices = random.Random(config.seed).sample(range(len(features)),
                                               min(config.eval_max_rows, len(features)))
    generator = torch.Generator().manual_seed(config.seed + 1)
    result = []
    for index in indices:
        batch = _mask_features([features[index]], tokenizer, config.mlm_probability, generator)
        result.append({key: value[0].tolist() for key, value in batch.items()})
    if not result or not any(any(n != -100 for n in row["labels"]) for row in result):
        raise ValueError("Validation data must contain maskable review text")
    return result


def _masked_accuracy(prediction):
    import numpy as np

    predictions, labels = prediction
    keep = labels != -100
    return {"masked_accuracy": float(np.mean(predictions[keep] == labels[keep]))}


def _fit_texts(train_texts, valid_texts, directory, config, provenance, resume=False):
    torch, transformers, _ = _dependencies()
    config.validate()
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        if not all((directory / name).is_file() for name in
                   ("training_config.json", "input_manifest.json")):
            raise FileExistsError(f"{directory} has incomplete setup files; choose a fresh model_name")
        saved_config = json.loads((directory / "training_config.json").read_text(encoding="utf-8"))
        saved_input = json.loads((directory / "input_manifest.json").read_text(encoding="utf-8"))
        completed = all((directory / name).is_file() for name in
                        ("completion.json", "embedding_config.json", "tokenizer.json",
                         "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
                         "encoder/config.json", "encoder/model.safetensors"))
        if completed:
            representation = json.loads((directory / "embedding_config.json").read_text(encoding="utf-8"))
            if representation.get("postprocessing"):
                completed = (directory / representation["postprocessing"]["file"]).is_file()
        requested = asdict(config)
        # A completed encoder is portable between inference devices. Interrupted
        # training still requires identical settings, including device selection.
        comparable_saved = {k: v for k, v in saved_config.items() if not completed or k != "device"}
        comparable_requested = {k: v for k, v in requested.items() if not completed or k != "device"}
        if comparable_saved != comparable_requested or saved_input != {
                "corpus_sha256": _corpus_digest(train_texts),
                "validation_sha256": _corpus_digest(valid_texts), "provenance": provenance}:
            raise ValueError("Existing run uses different data or settings; choose a fresh model_name")
        if completed:
            print(f"Reusing completed BERT run with identical inputs and settings: {directory}")
            return json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError(f"{directory} contains an interrupted run; pass resume=True "
                                  "or choose a fresh model_name")
    elif resume:
        raise FileNotFoundError("There is no existing run to resume")
    if not train_texts or any(not text.strip() for text in train_texts):
        raise ValueError("Training reviews must be nonempty")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but is unavailable")
    use_cuda = config.device != "cpu" and torch.cuda.is_available()
    if use_cuda:
        # Check actual kernels, including backward, on newer GPU architectures.
        probe = torch.ones((16, 16), device="cuda", requires_grad=True)
        (probe @ probe).sum().backward()
        del probe
        torch.cuda.reset_peak_memory_stats()
    transformers.set_seed(config.seed)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    tokenizer = (transformers.BertTokenizerFast.from_pretrained(str(directory), local_files_only=True)
                 if resume else _train_tokenizer(train_texts, config, directory))
    training, train_stats = _tokenize(train_texts, tokenizer, config.max_length)
    validation, valid_stats = _tokenize(valid_texts, tokenizer, config.max_length)
    evaluation = _fixed_eval(validation, tokenizer, config)
    model_config = transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_length,
        pad_token_id=tokenizer.pad_token_id,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
    )
    # Construction from configuration initializes random weights, without a checkpoint.
    model = transformers.BertForMaskedLM(model_config)
    eval_every = min(config.eval_every, config.max_steps)
    arguments = transformers.TrainingArguments(
        output_dir=str(directory / "checkpoints"), max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio, lr_scheduler_type="linear",
        eval_strategy="steps", eval_steps=eval_every,
        save_strategy="steps", save_steps=eval_every, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, logging_steps=min(50, eval_every),
        seed=config.seed, data_seed=config.seed, report_to="none",
        bf16=use_cuda and torch.cuda.is_bf16_supported(), fp16=False,
        use_cpu=not use_cuda, dataloader_num_workers=0,
        optim="adamw_torch", remove_unused_columns=False,
    )
    trainer = transformers.Trainer(
        model=model, args=arguments, train_dataset=training, eval_dataset=evaluation,
        data_collator=_MlmCollator(tokenizer, config.mlm_probability),
        processing_class=tokenizer,
        preprocess_logits_for_metrics=lambda logits, labels: logits.argmax(dim=-1),
        compute_metrics=_masked_accuracy,
    )
    _write_json(directory / "training_config.json", asdict(config))
    _write_json(directory / "input_manifest.json", {
        "corpus_sha256": _corpus_digest(train_texts),
        "validation_sha256": _corpus_digest(valid_texts), "provenance": provenance,
    })
    initial = {} if resume else trainer.evaluate(metric_key_prefix="initial")
    trainer.add_callback(transformers.EarlyStoppingCallback(early_stopping_patience=config.patience))
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.remove_callback(transformers.EarlyStoppingCallback)
    final = trainer.evaluate(metric_key_prefix="validation")
    # Only the selected encoder is needed for independent downstream classifiers.
    trainer.model.bert.save_pretrained(str(directory / "encoder"), safe_serialization=True)
    summary = {
        "initialization": "random", "objective": "MLM, no NSP or rating labels",
        "pooling": "mean excluding padding, CLS and SEP; L2 normalized",
        "embedding_dimension": config.hidden_size, "vocabulary_size": len(tokenizer),
        "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "training": train_stats, "validation": valid_stats,
        "validation_source": provenance["validation_source"],
        "selection_protocol": provenance["selection_protocol"],
        "mlm_eval_rows": len(evaluation), "fixed_mask_seed": config.seed + 1,
        "corpus_sha256": _corpus_digest(train_texts), "provenance": provenance,
        "initial_metrics": initial, "selected_checkpoint_metrics": final,
        "training_metrics": result.metrics,
        "selected_checkpoint": Path(trainer.state.best_model_checkpoint).name,
        "wall_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name() if use_cuda else "cpu",
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if use_cuda else 0,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "tokenizers", "accelerate")},
    }
    representation = _fit_representation(directory, train_texts, config)
    summary["embedding_dimension"] = representation["dimension"]
    summary["pooling"] = representation["pooling"]
    summary["postprocessing"] = representation.get("postprocessing")
    summary["wall_seconds"] = time.perf_counter() - started
    # Write the completion marker after every inference artifact is saved.
    _write_json(directory / "completion.json", {
        "status": "complete", "optimizer_steps": trainer.state.global_step,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


@EmbeddingModel.train_register("bert")
def bert_train(self, model_name="bert", *, config=None, resume=False):
    """Fit and select the encoder using only rows from self.train.

    A grouped holdout inside training supplies fixed-mask MLM early stopping.
    The public validation and test frames are not read here.
    """
    config = config or BertTrainingConfig()
    if isinstance(config, dict):
        config = BertTrainingConfig(**config)
    config.validate()
    pool_ids, pool_texts = _read_rows(self.train)
    train_ids, train_texts, holdout_ids, mlm_validation = _split_training_rows(
        pool_ids, pool_texts, config)
    provenance = {
        "selection_protocol": "train_internal_group_holdout_v1",
        "validation_source": "internal_training_holdout",
        "training_pool_rows": len(pool_ids),
        "training_rows": len(train_ids), "validation_rows": len(holdout_ids),
        "training_pool_ids_sha256": _corpus_digest([str(x) for x in pool_ids]),
        "training_pool_corpus_sha256": _corpus_digest(pool_texts),
        "training_ids_sha256": _corpus_digest([str(x) for x in train_ids]),
        "validation_ids_sha256": _corpus_digest([str(x) for x in holdout_ids]),
        "split_grouping": "NFKC/HTML/case/whitespace normalized text hash",
        "external_source": None,
    }
    _fit_texts(train_texts, mlm_validation, _model_directory(model_name), config, provenance, resume)


def _load_encoder(directory, device="auto"):
    torch, transformers, _ = _dependencies()
    directory = Path(directory)
    settings = json.loads((directory / "embedding_config.json").read_text(encoding="utf-8"))
    chosen = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if chosen == "auto":
        chosen = "cpu"
    tokenizer = transformers.BertTokenizerFast.from_pretrained(str(directory), local_files_only=True)
    model = transformers.BertModel.from_pretrained(str(directory / "encoder"),
                                                  local_files_only=True, add_pooling_layer=False)
    model.to(chosen).eval()
    if settings.get("postprocessing"):
        import numpy as np
        with np.load(directory / settings["postprocessing"]["file"], allow_pickle=False) as saved:
            settings["_projection"] = {key: saved[key] for key in ("mean", "components", "scale")}
    return tokenizer, model, settings, chosen


def _fit_representation(directory, fit_texts, config):
    """Fit sentence-vector postprocessing on the MLM fitting rows only."""
    multipliers = {"mean": 1, "last2_mean": 1, "all_mean": 1, "max": 1,
                   "mean_max": 2, "embedding_last_mean": 2, "segmented_mean": 3,
                   "layers_concat": config.num_hidden_layers,
                   "embedding_layers_concat": config.num_hidden_layers + 1}
    raw_dimension = config.hidden_size * multipliers[DEFAULT_POOLING]
    settings = {"max_length": config.max_length, "dimension": raw_dimension,
                "pooling": DEFAULT_POOLING, "normalize": True, "postprocessing": None}
    _write_json(directory / "embedding_config.json", settings)
    if DEFAULT_WHITEN_DIMENSION:
        import numpy as np
        from sklearn.decomposition import PCA
        from threadpoolctl import threadpool_limits
        tokenizer, encoder, raw_settings, device = _load_encoder(directory, config.device)
        vectors = _encode_texts(fit_texts, tokenizer, encoder, raw_settings, device)
        dimension = min(DEFAULT_WHITEN_DIMENSION, raw_dimension, len(fit_texts))
        with threadpool_limits(limits=8):
            pca = PCA(n_components=dimension, whiten=False, svd_solver="covariance_eigh")
            pca.fit(vectors.astype("float64"))
        scale = np.sqrt(np.maximum(pca.explained_variance_, 0))
        scale_floor = max(float(scale.max()) * 0.01, 1e-6)
        scale = np.maximum(scale, scale_floor)
        np.savez(directory / "postprocess.npz", mean=pca.mean_, components=pca.components_, scale=scale)
        settings.update(dimension=dimension, postprocessing={
            "file": "postprocess.npz", "method": "PCA whitening with scale floor and L2 normalization",
            "input_dimension": raw_dimension, "fit_rows": len(fit_texts),
            "fit_corpus_sha256": _corpus_digest(fit_texts), "scale_floor": scale_floor,
            "explained_variance_fraction": float(pca.explained_variance_ratio_.sum()),
        })
        _write_json(directory / "embedding_config.json", settings)
    return settings


def _project_vectors(vectors, settings):
    import numpy as np

    if "_projection" not in settings:
        return vectors
    saved = settings["_projection"]
    projected = (vectors.astype("float64") - saved["mean"]) @ saved["components"].T / saved["scale"]
    projected /= np.maximum(np.linalg.norm(projected, axis=1, keepdims=True), 1e-12)
    return projected.astype("float32")


def _pool_outputs(output, mask, pooling):
    """Pool contextual tokens without padding or special tokens."""
    import torch

    hidden = output.last_hidden_state
    if pooling == "last2_mean":
        hidden = torch.stack(output.hidden_states[1:][-2:]).mean(dim=0)
    elif pooling == "all_mean":
        hidden = torch.stack(output.hidden_states[1:]).mean(dim=0)
    def mean(value):
        return (value * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    if pooling in {"mean", "last2_mean", "all_mean"}:
        return mean(hidden)
    if pooling == "embedding_last_mean":
        return torch.cat((mean(output.hidden_states[0]), mean(hidden)), dim=1)
    if pooling == "layers_concat":
        return torch.cat([mean(layer) for layer in output.hidden_states[1:]], dim=1)
    if pooling == "embedding_layers_concat":
        return torch.cat([mean(layer) for layer in output.hidden_states], dim=1)
    if pooling in {"max", "mean_max"}:
        maximum = hidden.masked_fill(~mask, -torch.inf).max(dim=1).values
        maximum = torch.where(mask.any(dim=1), maximum, torch.zeros_like(maximum))
        return maximum if pooling == "max" else torch.cat((mean(hidden), maximum), dim=1)
    if pooling == "segmented_mean":
        ranks = mask.long().cumsum(dim=1) - 1
        segment = (3 * ranks) // mask.sum(dim=1, keepdim=True).clamp(min=1)
        parts = []
        for index in range(3):
            selected = mask & (segment == index)
            parts.append((hidden * selected).sum(dim=1) / selected.sum(dim=1).clamp(min=1))
        return torch.cat(parts, dim=1)
    raise ValueError(f"Unknown BERT pooling: {pooling}")


def _encode_texts(texts, tokenizer, model, settings, device, batch_size=128):
    import numpy as np
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    vectors = []
    pooling = settings["pooling"]
    if "_projection" in settings:
        from threadpoolctl import threadpool_limits
        projection_threads = threadpool_limits(limits=8)
    else:
        projection_threads = nullcontext()
    with projection_threads, torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(texts[start:start + batch_size], padding=True,
                                truncation=True, max_length=settings["max_length"],
                                return_special_tokens_mask=True, return_tensors="pt")
            special = encoded.pop("special_tokens_mask").to(device).bool()
            inputs = {key: value.to(device) for key, value in encoded.items()}
            output = model(**inputs, output_hidden_states=pooling in {
                "last2_mean", "all_mean", "embedding_last_mean", "layers_concat", "embedding_layers_concat"})
            mask = (inputs["attention_mask"].bool() & ~special).unsqueeze(-1)
            pooled = _pool_outputs(output, mask, pooling)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=1)
            vectors.append(_project_vectors(pooled.cpu().numpy(), settings))
    result = np.concatenate(vectors) if vectors else np.empty((0, settings["dimension"]), dtype="float32")
    if not np.isfinite(result).all():
        raise ValueError("BERT produced non-finite vectors")
    return result


def _bert_encode_frames(frames, model_name, *, batch_size=128, device="auto"):
    """Encode supplied frames without retaining text columns or Spark caches."""
    import numpy as np
    import pyarrow as pa
    from pyspark.ml.functions import array_to_vector
    from pyspark.sql import functions as F

    tokenizer, model, settings, chosen = _load_encoder(_model_directory(model_name), device)
    transformed = []
    for frame in frames:
        ids, texts = _read_rows(frame, ordered=False)
        vectors = _encode_texts(texts, tokenizer, model, settings, chosen, batch_size)
        # Columnar transfer avoids a Python object for every vector element and
        # large serialized Python-RDD task closures on the 4 GB Spark driver.
        offsets = np.arange(0, vectors.size + 1, settings["dimension"], dtype="int32")
        values = pa.ListArray.from_arrays(offsets, pa.array(vectors.reshape(-1)))
        table = pa.Table.from_arrays([pa.array(ids, type=pa.int64()), values],
                                     names=["uniqueID", "_bert_values"])
        vector_frame = frame.sparkSession.createDataFrame(table).select(
            F.col("uniqueID").cast(frame.schema["uniqueID"].dataType).alias("uniqueID"),
            array_to_vector("_bert_values").alias(
                "embedding", metadata={"ml_attr": {"num_attrs": settings["dimension"]}}),
        )
        original = EmbeddingModel._model_columns(frame.drop("embedding"))
        transformed.append(original.join(vector_frame, "uniqueID", "left")
                           .select(*original.columns, "embedding"))
    return tuple(transformed)


@EmbeddingModel.predict_register("bert")
def bert_predict(self, model_name="bert"):
    """Registered interface: return the three narrow analytics input frames.

    The shared decorator accepts only the instance and model directory name.
    The runner owns persistence and cache release.
    """
    return _bert_encode_frames((self.train, self.valid, self.test), model_name)
