"""Spark random forest and validation-controlled training experiments.

Run from the repository root: python -m src.analytics.random_forest
Each completed change is recorded in results/random_forest_training.
Use --resume <experiment-directory> to continue saved trials.
"""

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


def _metrics_from_counts(counts):
    """Compute unweighted evaluation metrics from a ten-class confusion table."""
    from collections import Counter
    from math import sqrt

    support, predicted, correct = Counter(), Counter(), Counter()
    absolute_error = squared_error = 0
    for label, prediction, count in counts:
        support[label] += count
        predicted[prediction] += count
        if label == prediction:
            correct[label] += count
        absolute_error += abs(label - prediction) * count
        squared_error += (label - prediction) ** 2 * count
    total = sum(support.values())
    if not total:
        raise ValueError("Cannot evaluate an empty split")
    accuracy = sum(correct.values()) / total
    precision = sum(n * correct[k] / predicted[k] for k, n in support.items() if predicted[k]) / total
    f1 = sum(n * 2 * correct[k] / (n + predicted[k]) for k, n in support.items()) / total
    return dict(accuracy=accuracy, weighted_f1=f1, weighted_precision=precision,
                weighted_recall=accuracy, mae=absolute_error / total,
                rmse=sqrt(squared_error / total),
                prediction_counts={str(int(k) + 1): n for k, n in sorted(predicted.items())})


def _training_metrics(frame):
    # Aggregate once; avoid running forest inference separately for every metric.
    counts = [(int(r.rating_p1), int(r.prediction), r["count"])
              for r in frame.groupBy("rating_p1", "prediction").count().collect()]
    return _metrics_from_counts(counts)


def _windows_local_io_jar(directory):
    """Build a local-only Hadoop adapter; NTFS still enforces file access."""
    import os
    from pathlib import Path
    import subprocess
    import pyspark

    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "WindowsLocalFileSystem.java"
    source.write_text('''package localio;
import java.io.*;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.permission.FsPermission;
public class WindowsLocalFileSystem extends RawLocalFileSystem {
 @Override public boolean mkdirs(Path p, FsPermission permission) throws IOException {
  File f=pathToFile(p); return f.isDirectory() || f.mkdirs();
 }
 @Override public void setPermission(Path p, FsPermission permission) throws IOException { }
 @Override public FileStatus getFileStatus(Path p) throws IOException {
  File f=pathToFile(p);
  if(!f.exists()) throw new FileNotFoundException(p.toString());
  return new FileStatus(f.length(), f.isDirectory(), 1, 4096, f.lastModified(),
                        p.makeQualified(getUri(),getWorkingDirectory()));
 }
 @Override public FileStatus[] listStatus(Path p) throws IOException {
  File f=pathToFile(p);
  if(!f.isDirectory()) return new FileStatus[]{getFileStatus(p)};
  File[] files=f.listFiles();
  if(files==null) throw new IOException("Cannot list "+p);
  FileStatus[] result=new FileStatus[files.length];
  for(int i=0;i<files.length;i++) result[i]=getFileStatus(new Path(files[i].toURI()));
  return result;
 }
}''', encoding="utf-8")
    java_bin = Path(os.environ["JAVA_HOME"]) / "bin"
    jars = str(Path(pyspark.__file__).parent / "jars" / "*")
    subprocess.run([str(java_bin / "javac.exe"), "-cp", jars, "-d", str(directory), str(source)], check=True)
    jar = directory / "windows-local-io.jar"
    subprocess.run([str(java_bin / "jar.exe"), "cf", str(jar), "-C", str(directory), "localio"], check=True)
    return str(jar)


def _write_training_report(directory):
    """Write a human-readable record without changing model selection."""
    import json

    records = json.loads((directory / "trials.json").read_text(encoding="utf-8"))
    descriptions = {
        "00_original": "原始配置：100 棵树，深度 5，仅文本，不加权",
        "01_weights": "仅启用训练集类别权重",
        "02_depth10": "仅将树深从 5 增至 10",
        "03_depth16": "仅将树深从 10 增至 16",
        "04_features": "仅增加病症、日期、有用票数特征",
        "05_trees": "仅将树数从 100 增至 200",
    }
    lines = ["# 随机森林训练修改记录", "",
             "使用现有 TF-IDF 缓存和 seed=42 的既有数据划分，仅在训练集拟合，按验证集 weighted F1 选模型。",
             "每轮只改变下表所列的一项；下降也照实记录。变化量单位为百分点。",
             "表中 F1 均为 weighted F1；评估按原始类别支持度加权，不使用训练类别权重。", "",
             "| 轮次与修改 | 训练 F1 | 验证 Accuracy | 验证 F1 | 比上一轮 | 比原始配置 | 训练秒数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f'| {descriptions[r["trial"]]} | {r["train"]["weighted_f1"]:.4f} | '
                     f'{r["valid"]["accuracy"]:.4f} | {r["valid"]["weighted_f1"]:.4f} | '
                     f'{r["delta_valid_f1_previous_pp"]:+.2f} | {r["delta_valid_f1_original_pp"]:+.2f} | '
                     f'{r["fit_seconds"]:.2f} |')
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected, test = summary["selected"], summary["test"]
        lines += ["", f'验证集选定：{descriptions[selected["trial"]]}。', "",
                  f'测试 Accuracy：{test["accuracy"]:.6f}；weighted F1：{test["weighted_f1"]:.6f}；'
                  f'MAE：{test["mae"]:.6f}；RMSE：{test["rmse"]:.6f}。', "",
                  f'选定模型参数：`{json.dumps(selected["parameters"], ensure_ascii=False)}`。',
                  f'输入列：`{selected["features"]}`。', "",
                  "测试集只在选参结束后评分，未用于决定任何一轮修改或模型选择。",
                  "原始配置在训练和验证中都只预测 10 分；每轮预测分布见 trials.json。",
                  "这是 Spark 实测结果，与此前 scikit-learn 的 0.3689 Accuracy 属于不同配置。",
                  "新增辅助特征未包含 drugName_idx，以避免其高类别数超出 Spark 默认 maxBins 的限制。",
                  "usefulCount_log 使用评论发表后积累的投票信息，沿用现有数据定义。", "",
                  "模型函数的默认配置可被调用方显式传入的 feature_cols 和 rf_params 覆盖。若 notebook 仍指定 feature_cols=['embedding']，会运行纯文本配置。", "",
                  "完整参数修改及各轮指标：trials.json；可筛选对比表：progress.csv；最终测试及输入文件指纹：summary.json。"]
    (directory / "训练修改记录.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_training_experiments():
    import argparse
    import csv
    import hashlib
    import json
    import os
    from pathlib import Path
    import sys
    import time
    from pyspark.sql import SparkSession, functions as F
    from src.dataOI import DataOI
    from src.paths import ROOT, EMBEDDING_DIR, ANALYTIC_MODELS_DIR

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    parser = argparse.ArgumentParser(description="Train RF and record each validation change")
    parser.add_argument("--resume", type=Path, help="Continue an existing experiment directory")
    parser.add_argument("--tfidf-controls", action="store_true", help="Run matched regularization/class-weight controls")
    parser.add_argument("--tune-text", action="store_true", help="Compare training-importance text feature subsets")
    parser.add_argument("--calibrate", type=Path, help="Calibrate a completed experiment by validation accuracy")
    args = parser.parse_args()
    if args.tfidf_controls:
        return _tfidf_controls()
    if args.calibrate:
        return _calibrate_forest(args.calibrate)
    if args.tune_text:
        return _tune_text_features()
    out = args.resume or ROOT / "results" / "random_forest_training" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print("OUTPUT", str(out), flush=True)
    builder = (SparkSession.builder.master("local[*]").appName("RandomForestTraining")
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.default.parallelism", "4")
             .config("spark.ui.showConsoleProgress", "false"))
    if os.name == "nt":
        jar = _windows_local_io_jar(out / "local_io")
        builder = (builder.config("spark.driver.extraClassPath", jar)
                   .config("spark.executor.extraClassPath", jar)
                   .config("spark.hadoop.fs.file.impl", "localio.WindowsLocalFileSystem"))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    records = json.loads((out / "trials.json").read_text(encoding="utf-8")) if args.resume else []
    for record in records:
        estimator = RandomForestClassifier(labelCol="rating_p1", **record["parameters"])
        record["resolved_parameters"] = {p.name: value for p, value in estimator.extractParamMap().items()}
    if records:
        (out / "trials.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    try:
        # Load only train/validation during selection.
        def read_split(split):
            # Explicit part files avoid Hadoop's native Windows directory listing.
            return spark.read.parquet(*[str(p) for p in sorted(
                (EMBEDDING_DIR / f"tfidf/{split}.parquet").glob("*.parquet"))])

        train = read_split("train").cache()
        valid = read_split("valid").cache()
        print("ROWS", train.count(), valid.count(), flush=True)
        text = ["embedding"]
        # Keep Spark categorical metadata; drugName_idx exceeds the small-bin budget.
        structured = text + ["condition_vec", "days", "usefulCount_log"]
        trials = [
            ("00_original", "Original: text only, depth 5, unweighted", text, 5, False, 100),
            ("01_weights", "Only enable training class weights", text, 5, True, 100),
            ("02_depth10", "Only increase maxDepth from 5 to 10", text, 10, True, 100),
            ("03_depth16", "Only increase maxDepth from 10 to 16", text, 16, True, 100),
            ("04_features", "Only add condition, date and useful votes", structured, 16, True, 100),
            ("05_trees", "Only increase numTrees from 100 to 200", structured, 16, True, 200),
        ]
        assembled = {}
        best_model = None
        best_record = max(records, key=lambda r: r["valid"]["weighted_f1"]) if records else None
        for name, change, features, depth, balanced, trees in trials:
            if any(r["trial"] == name for r in records):
                continue
            params = dict(numTrees=trees, maxDepth=depth, seed=42)
            if balanced:
                params["weightCol"] = "weight"
            record = dict(trial=name, change=change, features=features, parameters=params)
            print("START", json.dumps(record), flush=True)
            started = time.perf_counter()
            if tuple(features) not in assembled:
                instance = AnalyticsModel("tfidf", train, valid, valid)
                tr, va, _ = instance._feature_transform(features)
                assembled[tuple(features)] = (tr, va)
            tr, va = assembled[tuple(features)]
            model = RandomForestClassifier(labelCol="rating_p1", **params).fit(tr)
            record["resolved_parameters"] = {p.name: value for p, value in model.extractParamMap().items()}
            record["fit_seconds"] = time.perf_counter() - started
            record["train"] = _training_metrics(model.transform(tr))
            record["valid"] = _training_metrics(model.transform(va))
            score = record["valid"]["weighted_f1"]
            record["delta_valid_f1_previous_pp"] = (score - records[-1]["valid"]["weighted_f1"]) * 100 if records else 0
            record["delta_valid_f1_original_pp"] = (score - records[0]["valid"]["weighted_f1"]) * 100 if records else 0
            records.append(record)
            (out / "trials.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            _write_training_report(out)
            with (out / "progress.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                fields = ["trial", "change", "fit_seconds", "train_accuracy", "train_f1", "valid_accuracy", "valid_f1", "delta_previous_pp", "delta_original_pp"]
                writer = csv.writer(stream)
                writer.writerow(fields)
                for r in records:
                    writer.writerow([r["trial"], r["change"], r["fit_seconds"], r["train"]["accuracy"], r["train"]["weighted_f1"], r["valid"]["accuracy"], r["valid"]["weighted_f1"], r["delta_valid_f1_previous_pp"], r["delta_valid_f1_original_pp"]])
            if best_record is None or score > best_record["valid"]["weighted_f1"]:
                best_model, best_record = model, record
                model.write().overwrite().save(str(out / f"checkpoint_{name}"))
            print("COMPLETE", json.dumps(record), flush=True)

        # Freeze selection before reading test features or labels.
        (out / "selected.json").write_text(json.dumps(best_record, indent=2), encoding="utf-8")
        if best_model is None:
            instance = AnalyticsModel("tfidf", train, valid, valid)
            tr, va, _ = instance._feature_transform(best_record["features"])
            assembled[tuple(best_record["features"])] = (tr, va)
            checkpoint = out / f'checkpoint_{best_record["trial"]}'
            if checkpoint.is_dir():
                from pyspark.ml.classification import RandomForestClassificationModel
                print("LOAD_SELECTED_CHECKPOINT", str(checkpoint), flush=True)
                best_model = RandomForestClassificationModel.load(str(checkpoint))
            else:
                print("REFIT_SELECTED", best_record["trial"], flush=True)
                best_model = RandomForestClassifier(labelCol="rating_p1", **best_record["parameters"]).fit(tr)
            verification = _training_metrics(best_model.transform(va))
            assert abs(verification["weighted_f1"] - best_record["valid"]["weighted_f1"]) < 1e-12
        test = read_split("test")
        from pyspark.ml.feature import VectorAssembler
        test = VectorAssembler(inputCols=best_record["features"], outputCol="features").transform(test)
        test_pred = best_model.transform(test).cache()
        test_metrics = _training_metrics(test_pred)
        assert len(test_metrics["prediction_counts"]) > 1, "Selected model still predicts only one class"
        report = dict(selected=best_record, test=test_metrics, spark_version=spark.version,
                      python=sys.executable, seed=42, selection_metric="validation weighted F1",
                      input_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (EMBEDDING_DIR / "tfidf").rglob("*.parquet") if p.is_file()})
        (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        ANALYTIC_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        best_model.write().overwrite().save(str(ANALYTIC_MODELS_DIR / "random_forest__tfidf"))
        from pyspark.ml.classification import RandomForestClassificationModel
        loaded = RandomForestClassificationModel.load(str(ANALYTIC_MODELS_DIR / "random_forest__tfidf"))
        tr, va = assembled[tuple(best_record["features"])]
        sample = va.limit(1000).cache()
        expected = best_model.transform(sample).select("uniqueID", F.col("prediction").alias("expected"))
        actual = loaded.transform(sample).select("uniqueID", "prediction")
        assert expected.join(actual, "uniqueID").filter("expected != prediction").count() == 0
        report["saved_model_reload_verified"] = True
        cols = ["uniqueID", "rating", "prediction"]
        DataOI().save_prediction_data("random_forest__tfidf", best_model.transform(tr).select(*cols),
                                     best_model.transform(va).select(*cols), test_pred.select(*cols))
        (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["status"] = "complete"
        (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        _write_training_report(out)
        print("FINAL", json.dumps(report), flush=True)
    except Exception as exc:
        (out / "failure.txt").write_text(repr(exc), encoding="utf-8")
        raise
    finally:
        spark.stop()


def _publish_tuned_pipeline(directory):
    import json
    import shutil
    from src.paths import ROOT, ANALYTIC_MODELS_DIR

    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["selected"]["trial"] == "baseline":
        return
    target = (ANALYTIC_MODELS_DIR / "random_forest__tfidf").resolve()
    stage = (ANALYTIC_MODELS_DIR / f"random_forest__tfidf_{directory.name}_staged").resolve()
    backup = (directory / "previous_published_model").resolve()
    for path in [target, stage, backup]:
        if ROOT.resolve() not in path.parents:
            raise ValueError(f"Model publication outside repository: {path}")
    if backup.exists():
        raise ValueError("Previous model backup already exists; refusing to overwrite it")
    shutil.copytree(directory / "best_pipeline", stage)
    if target.exists():
        target.rename(backup)
    stage.rename(target)


def _tune_text_features():
    import csv
    import hashlib
    import json
    import os
    import sys
    import time
    import numpy as np
    from pyspark.ml import PipelineModel
    from pyspark.ml.classification import RandomForestClassificationModel
    from pyspark.ml.feature import VectorAssembler
    from pyspark.sql import SparkSession, functions as F
    from src.dataOI import DataOI
    from src.paths import ROOT, EMBEDDING_DIR, ANALYTIC_MODELS_DIR

    previous = ROOT / "results/random_forest_training/20261107_223301"
    out = ROOT / "results/random_forest_training" / time.strftime("%Y%m%d_%H%M%S_text_selection")
    out.mkdir(parents=True)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    builder = (SparkSession.builder.master("local[*]").appName("RandomForestTextSelection")
               .config("spark.driver.memory", "4g").config("spark.default.parallelism", "4")
               .config("spark.sql.shuffle.partitions", "8").config("spark.ui.showConsoleProgress", "false"))
    if os.name == "nt":
        jar = _windows_local_io_jar(out / "local_io")
        builder = (builder.config("spark.driver.extraClassPath", jar)
                   .config("spark.executor.extraClassPath", jar)
                   .config("spark.hadoop.fs.file.impl", "localio.WindowsLocalFileSystem"))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("OUTPUT", str(out), flush=True)
    try:
        old_summary = json.loads((previous / "summary.json").read_text(encoding="utf-8"))
        # Refuse to reuse historical scores with changed inputs.
        for name, expected_hash in old_summary["input_files"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected_hash, name
        def read(split):
            return spark.read.parquet(*[str(p) for p in sorted(
                (EMBEDDING_DIR / f"tfidf/{split}.parquet").glob("*.parquet"))])
        train, valid = read("train").cache(), read("valid").cache()
        features = old_summary["selected"]["features"]
        params = old_summary["selected"]["parameters"]
        baseline_model = RandomForestClassificationModel.load(str(previous / "checkpoint_05_trees"))
        dimension = train.select("embedding").first().embedding.size
        ranking = np.argsort(-baseline_model.featureImportances.toArray()[:dimension], kind="stable")
        base = dict(trial="baseline", change="Previous 200-tree full-text configuration",
                    parameters=params, text_features=dimension,
                    train=old_summary["selected"]["train"], valid=old_summary["selected"]["valid"], fit_seconds=0)
        records, best, best_model, best_indices = [base], base, None, None
        for k in [2000, 500]:
            indices = sorted(int(i) for i in ranking[:k])
            print("START", k, "selected text features", flush=True)
            tr, va, _, stages = _slice_rf_frames([train, valid, valid], features, indices)
            started = time.perf_counter()
            model = RandomForestClassifier(labelCol="rating_p1", **params).fit(tr)
            record = dict(trial=f"top_{k}", change=f"Retain top {k} text features ranked by training-only forest importance",
                          parameters=params, text_features=k, indices=indices,
                          fit_seconds=time.perf_counter()-started,
                          train=_training_metrics(model.transform(tr)), valid=_training_metrics(model.transform(va)))
            for metric in ["accuracy", "weighted_f1"]:
                record[f"delta_{metric}_previous_pp"] = 100*(record["valid"][metric]-records[-1]["valid"][metric])
                record[f"delta_{metric}_baseline_pp"] = 100*(record["valid"][metric]-base["valid"][metric])
            records.append(record)
            (out / "trials.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            with (out / "progress.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["trial", "text_features", "fit_seconds", "train_f1", "valid_accuracy", "valid_f1", "delta_accuracy_previous_pp", "delta_accuracy_baseline_pp", "delta_f1_previous_pp", "delta_f1_baseline_pp"])
                for r in records:
                    writer.writerow([r["trial"], r["text_features"], r["fit_seconds"], r["train"]["weighted_f1"], r["valid"]["accuracy"], r["valid"]["weighted_f1"], r.get("delta_accuracy_previous_pp",0), r.get("delta_accuracy_baseline_pp",0), r.get("delta_weighted_f1_previous_pp",0), r.get("delta_weighted_f1_baseline_pp",0)])
            if record["valid"]["weighted_f1"] > best["valid"]["weighted_f1"]:
                best, best_indices = record, indices
                best_model = PipelineModel(stages=stages + [model])
                best_model.write().overwrite().save(str(out / f"checkpoint_top_{k}"))
            print("COMPLETE", json.dumps({a:b for a,b in record.items() if a != "indices"}), flush=True)
        (out / "selected.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        test = read("test")
        if best_model is None:
            best_model = PipelineModel(stages=[VectorAssembler(inputCols=features, outputCol="features"), baseline_model])
        predictions = best_model.transform(test).cache()
        test_metrics = _training_metrics(predictions)
        best_model.write().overwrite().save(str(out / "best_pipeline"))
        loaded = PipelineModel.load(str(out / "best_pipeline"))
        sample = test.limit(1000).cache()
        expected = best_model.transform(sample).select("uniqueID", F.col("prediction").alias("expected"))
        actual = loaded.transform(sample).select("uniqueID", "prediction")
        assert expected.join(actual,"uniqueID").filter("expected != prediction").count() == 0
        report = dict(status="complete", selected=best, test=test_metrics,
                      previous_test=old_summary["test"], selection_metric="validation weighted F1",
                      saved_model_reload_verified=True, input_hashes_verified=True,
                      limitation="Repeated tuning on the same validation split; test set was previously observed.")
        (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if best_indices is not None:
            selection = dict(indices=best_indices, parameters=params, features=features,
                             experiment=str(out), embedding_dimension=dimension)
            (ANALYTIC_MODELS_DIR / "random_forest__tfidf_selection.json").write_text(json.dumps(selection,indent=2), encoding="utf-8")
            cols=["uniqueID","rating","prediction"]
            DataOI().save_prediction_data("random_forest__tfidf", best_model.transform(train).select(*cols),
                                         best_model.transform(valid).select(*cols), predictions.select(*cols))
        lines=["# 随机森林继续训练记录", "", "按验证 weighted F1 选参；每轮保留相同 200 棵树、深度 16、类别权重，只改变文本特征数量。",
               "特征排名来自上一轮仅在训练集拟合的森林；训练、验证、测试划分与输入指纹保持一致。", "",
               "| 配置 | 验证 Accuracy | 验证 F1 | Accuracy 比上一轮（百分点） | F1 比上一轮（百分点） | 训练秒数 |", "|---|---:|---:|---:|---:|---:|"]
        for r in records:
            lines.append(f'| {r["trial"]} | {r["valid"]["accuracy"]:.6f} | {r["valid"]["weighted_f1"]:.6f} | {r.get("delta_accuracy_previous_pp",0):+.3f} | {r.get("delta_weighted_f1_previous_pp",0):+.3f} | {r["fit_seconds"]:.2f} |')
        lines += ["", f'选定：{best["trial"]}。测试 Accuracy={test_metrics["accuracy"]:.6f}，weighted F1={test_metrics["weighted_f1"]:.6f}。',
                  "训练耗时为 0 的 baseline 是复用已核验历史结果，并未重新拟合。",
                  "这是同一验证划分上的继续调优；测试集此前已被观察，当前比较不构成全新独立测试。", "模型管道已保存并通过重载预测核验。"]
        (out / "训练增长记录.md").write_text("\n".join(lines), encoding="utf-8")
        _publish_tuned_pipeline(out)
        print("FINAL", json.dumps({k:v for k,v in report.items() if k!="selected"}), flush=True)
    except Exception as exc:
        (out / "failure.txt").write_text(repr(exc), encoding="utf-8")
        raise
    finally:
        spark.stop()


def _write_calibration_report(directory):
    import json
    records = json.loads((directory / "trials.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    baseline = records[0]["valid"]
    lines = ["# 随机森林判类阈值增长记录", "",
             "复用已训练的 200 棵、深度 16 的随机森林，未重新拟合树。训练类别频数决定各类阈值，验证集选择阈值强度 alpha。",
             "本阶段要求验证 F1 不低于原配置，再按 Accuracy 选参；同时记录所有配置的增减。", "",
             "| alpha | 验证 Accuracy | 验证 weighted F1 | Accuracy 比上一组（百分点） | Accuracy 比原配置（百分点） | F1 比原配置（百分点） |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in records:
        v = r["valid"]
        lines.append(f'| {r["alpha"]} | {v["accuracy"]:.6f} | {v["weighted_f1"]:.6f} | '
                     f'{r["delta_accuracy_previous_pp"]:+.3f} | {(v["accuracy"]-baseline["accuracy"])*100:+.3f} | '
                     f'{(v["weighted_f1"]-baseline["weighted_f1"])*100:+.3f} |')
    test = summary["test"]
    lines += ["", f'验证集选定 alpha={summary["selected"]["alpha"]}。',
              f'选定后测试 Accuracy={test["accuracy"]:.6f}，weighted F1={test["weighted_f1"]:.6f}，MAE={test["mae"]:.6f}，RMSE={test["rmse"]:.6f}。', "",
              "此前保留 2000 和 500 个文本特征的重训均未提升，记录见相邻 20261108_011704_text_selection 目录。",
              "输入文件指纹在恢复运行前与原始实验逐一核对一致。训练/验证/测试划分保持不变。",
              "测试集此前已被观察；这些是同一划分上的继续调参成绩，不代表全新独立测试。",
              "训练耗时为 0 表示仅调整判类阈值，并非重新拟合。模型管道已保存并重新加载核验。"]
    (directory / "训练增长记录.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def _calibrate_forest(experiment):
    import csv
    import json
    import os
    import sys
    import time
    from collections import Counter
    import numpy as np
    from pyspark.ml import PipelineModel
    from pyspark.sql import SparkSession
    from src.paths import ROOT, EMBEDDING_DIR, ANALYTIC_MODELS_DIR
    from src.dataOI import DataOI

    out = ROOT / "results/random_forest_training" / time.strftime("%Y%m%d_%H%M%S_calibration")
    out.mkdir(parents=True)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    builder = (SparkSession.builder.master("local[*]").appName("RFCalibration")
               .config("spark.driver.memory","4g").config("spark.default.parallelism","4")
               .config("spark.sql.shuffle.partitions","8").config("spark.ui.showConsoleProgress","false"))
    if os.name == "nt":
        jar = _windows_local_io_jar(out / "local_io")
        builder = (builder.config("spark.driver.extraClassPath",jar).config("spark.executor.extraClassPath",jar)
                   .config("spark.hadoop.fs.file.impl","localio.WindowsLocalFileSystem"))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    print("OUTPUT", str(out), flush=True)
    try:
        def read(split):
            return spark.read.parquet(*[str(p) for p in sorted((EMBEDDING_DIR/f"tfidf/{split}.parquet").glob("*.parquet"))])
        pipe = PipelineModel.load(str(experiment / "best_pipeline"))
        model = pipe.stages[-1]
        counts = read("train").groupBy("rating_p1").count().collect()
        frequency = np.array([next(r["count"] for r in counts if r.rating_p1==k) for k in range(10)])
        weights = frequency.sum()/(10*frequency)
        valid = read("valid")
        rows = pipe.transform(valid).select("rating_p1","probability").collect()
        labels = np.array([int(r.rating_p1) for r in rows])
        probabilities = np.array([r.probability.toArray() for r in rows])
        records=[]
        for alpha in [0.0,0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.5,0.75,1.0]:
            thresholds=weights**alpha
            prediction=(probabilities/thresholds).argmax(axis=1)
            table=Counter(zip(labels.tolist(),prediction.tolist()))
            metrics=_metrics_from_counts([(a,b,n) for (a,b),n in table.items()])
            record=dict(alpha=alpha, thresholds=thresholds.tolist(), valid=metrics, fit_seconds=0,
                        delta_accuracy_previous_pp=100*(metrics["accuracy"]-records[-1]["valid"]["accuracy"]) if records else 0,
                        delta_f1_previous_pp=100*(metrics["weighted_f1"]-records[-1]["valid"]["weighted_f1"]) if records else 0)
            records.append(record)
            print("VALID",json.dumps(record),flush=True)
        eligible=[r for r in records if r["valid"]["weighted_f1"] >= records[0]["valid"]["weighted_f1"]]
        best=max(eligible,key=lambda r:(r["valid"]["accuracy"],r["valid"]["weighted_f1"]))
        (out/"trials.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
        (out/"selected.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
        model.setThresholds(best["thresholds"])
        verification=_training_metrics(pipe.transform(valid))
        assert abs(verification["accuracy"]-best["valid"]["accuracy"])<1e-12
        test=read("test")
        test_metrics=_training_metrics(pipe.transform(test))
        pipe.write().overwrite().save(str(out/"best_pipeline"))
        loaded=PipelineModel.load(str(out/"best_pipeline"))
        assert abs(_training_metrics(loaded.transform(test))["accuracy"]-test_metrics["accuracy"])<1e-12
        cols=["uniqueID","rating","prediction"]
        DataOI().save_prediction_data("random_forest__tfidf",pipe.transform(read("train")).select(*cols),pipe.transform(valid).select(*cols),pipe.transform(test).select(*cols))
        (ANALYTIC_MODELS_DIR/"random_forest__tfidf_calibration.json").write_text(json.dumps(best,indent=2),encoding="utf-8")
        report=dict(status="complete",selected=dict(trial="calibrated",**best),test=test_metrics,
                    selection_metric="validation accuracy subject to weighted F1 >= baseline",saved_model_reload_verified=True,
                    source_experiment=str(experiment),operation="Reuse trained trees; adjust thresholds only")
        (out/"summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        with (out/"progress.csv").open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.writer(stream); writer.writerow(["alpha","valid_accuracy","valid_f1","delta_accuracy_previous_pp","delta_f1_previous_pp","fit_seconds"])
            for r in records: writer.writerow([r["alpha"],r["valid"]["accuracy"],r["valid"]["weighted_f1"],r["delta_accuracy_previous_pp"],r["delta_f1_previous_pp"],0])
        _write_calibration_report(out)
        _publish_tuned_pipeline(out)
        print("FINAL",json.dumps(report),flush=True)
    finally:
        spark.stop()


def _tfidf_controls():
    import csv
    import hashlib
    import json
    import os
    import sys
    import time
    from pyspark.sql import SparkSession
    from pyspark.ml.classification import RandomForestClassificationModel
    from pyspark.ml.feature import VectorAssembler
    from src.paths import ROOT, EMBEDDING_DIR

    out=ROOT/'results/tfidf_rf_controls'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True)
    os.environ.setdefault('PYSPARK_PYTHON',sys.executable)
    os.environ.setdefault('SPARK_LOCAL_IP','127.0.0.1')
    builder=(SparkSession.builder.master('local[*]').appName('TFIDFRFStrictControls')
             .config('spark.driver.memory','4g').config('spark.default.parallelism','4')
             .config('spark.sql.shuffle.partitions','8').config('spark.ui.showConsoleProgress','false'))
    if os.name=='nt':
        jar=_windows_local_io_jar(out/'local_io')
        builder=(builder.config('spark.driver.extraClassPath',jar).config('spark.executor.extraClassPath',jar)
                 .config('spark.hadoop.fs.file.impl','localio.WindowsLocalFileSystem'))
    spark=builder.getOrCreate(); spark.sparkContext.setLogLevel('ERROR')
    print('OUTPUT',str(out),flush=True)
    try:
        previous=ROOT/'results/random_forest_training/20261107_223301'
        old=json.loads((previous/'summary.json').read_text(encoding='utf-8'))
        for name,digest in old['input_files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        def read(split):
            return spark.read.parquet(*[str(p) for p in sorted((EMBEDDING_DIR/f'tfidf/{split}.parquet').glob('*.parquet'))])
        assembler=VectorAssembler(inputCols=old['selected']['features'],outputCol='features')
        train,valid=assembler.transform(read('train')).cache(),assembler.transform(read('valid')).cache()
        common=dict(numTrees=100,seed=42,maxMemoryInMB=64,featureSubsetStrategy='auto',subsamplingRate=1.0)
        specs=[('weak_unweighted',False,10,5),('weak_weighted',True,10,5),
               ('strong_unweighted',False,8,20),('strong_weighted',True,8,20)]
        protocol=dict(features=old['selected']['features'],seed=42,configs=specs,common=common,
                      evaluation='All four configurations frozen before scoring any test rows',
                      class_weights='n/(10*class_count) on train only; no resampling',
                      weaker_regularization='maxDepth=10,minInstancesPerNode=5',
                      stronger_regularization='maxDepth=8,minInstancesPerNode=20',
                      default_model_unchanged=True)
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
        records=[]
        for name,balanced,depth,leaf in specs:
            params=dict(common,maxDepth=depth,minInstancesPerNode=leaf)
            if balanced: params['weightCol']='weight'
            print('START',name,json.dumps(params),flush=True)
            started=time.perf_counter()
            model=RandomForestClassifier(labelCol='rating_p1',**params).fit(train)
            seconds=time.perf_counter()-started
            record=dict(name=name,balanced=balanced,regularization='strong' if depth==8 else 'weak',
                        parameters=params,fit_seconds=seconds,reused_checkpoint=False,
                        train=_training_metrics(model.transform(train)),valid=_training_metrics(model.transform(valid)))
            model.write().save(str(out/name))
            records.append(record)
            (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print('COMPLETE',json.dumps(record),flush=True)
        test=assembler.transform(read('test')).cache()
        for record in records:
            model=RandomForestClassificationModel.load(str(out/record['name']))
            prediction=model.transform(test)
            record['test']=_training_metrics(prediction)
            prediction.select('uniqueID','rating','prediction').write.parquet(str(out/f'{record["name"]}_test_predictions.parquet'))
        with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.writer(stream)
            writer.writerow(['configuration','balanced','regularization','split','accuracy','weighted_precision','weighted_f1','mae','rmse','fit_seconds'])
            for r in records:
                for split in ['train','valid','test']:
                    m=r[split]
                    writer.writerow([r['name'],r['balanced'],r['regularization'],split,m['accuracy'],m['weighted_precision'],m['weighted_f1'],m['mae'],m['rmse'],r['fit_seconds']])
        by_name={r['name']:r for r in records}
        differences=[]
        for before,after,effect in [('weak_unweighted','strong_unweighted','regularization_without_weights'),
                                    ('weak_weighted','strong_weighted','regularization_with_weights'),
                                    ('weak_unweighted','weak_weighted','weights_with_weak_regularization'),
                                    ('strong_unweighted','strong_weighted','weights_with_strong_regularization')]:
            for split in ['valid','test']:
                row=dict(effect=effect,before=before,after=after,split=split)
                for metric in ['accuracy','weighted_precision','weighted_f1','mae','rmse']:
                    row[metric]=by_name[after][split][metric]-by_name[before][split][metric]
                differences.append(row)
        (out/'differences.json').write_text(json.dumps(differences,indent=2),encoding='utf-8')
        (out/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        (out/'complete.json').write_text(json.dumps(dict(status='passed',inputs_sha256_verified=True,protocol=protocol),indent=2),encoding='utf-8')
        print('STRICT_CONTROLS_COMPLETE',str(out),flush=True)
    except Exception as exc:
        (out/'failure.txt').write_text(repr(exc),encoding='utf-8')
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    _run_training_experiments()
