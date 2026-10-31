# 模型接入与职责说明

本项目预测 1–10 分的十分类 `rating`。AI 先读根目录 [AGENTS.md](../AGENTS.md)，再按当前分配的模型文件完成任务。本文记录现有代码的接口，不要求模型成员重构公共流程。

## 1. 职责与最小改动范围

embedding 成员只实现 `src/embeddings/<name>.py`：选择原评论或 token，拟合表示，保存自身模型，输出向量。analytics 成员只实现 `src/analytics/<name>.py`：选择自身特征、拟合分类器、保存自身模型、返回预测。模型内部需要的辅助函数、缩放或降维放在该模型文件里。

集成人负责 `base.py`、`__init__.py`、路径、I/O、预处理、划分、编码、评分、共享 notebook、依赖与最终材料。成员发现接口或共享逻辑需要变化时，说明具体原因和影响，由集成人统一处理；不要顺手修改其他模型或清洗规则。

成员只需在现成实验入口中设置自己的注册名和重建开关，不需要改 notebook 的公共循环。依赖的输入不存在时先按 README 准备数据，不能通过临时改共享路径、标签或划分来让单个模型跑通。姓名和具体模型分工由小组另行确定，本文不代为分配。

## 2. 数据流与输入

主流程依次执行：原始 CSV → EDA → 清洗 → 文本特征 → ID 哈希划分和训练权重 → 类别编码 → embedding 训练 / 变换 → analytics 训练 / 预测 → 评分。所有训练函数先执行完，再执行对应预测阶段。

`split_and_weight` 用 `pmod(xxhash64(uniqueID, SEED), 10000)`，桶号小于 8000 为训练，其余为测试，比例约 80/20。固定数据、ID 类型、Spark 版本和 seed 后，归属不受顺序或分区影响。默认 `SEED = 42`。预处理后样本集合变化时，需要统一重建和记录新基准，不能继续混用旧结果。

传给 embedding 的 `self.train` / `self.test` 是 Spark DataFrame，保留以下列：

- `uniqueID`：整数型样本 ID，一行一条评论，必须保留，不能重编。
- `drugName`、`condition`、`review`：字符串。`review` 已经过共享清洗，不是未经处理的原始 CSV 文本。
- `words`、`words_stem`：`array<string>`，分别为未词干化 token 和 TF-IDF 使用的词干。其他表示可直接读取 `review` 或 `words`。
- `days`：整数，距 1970-01-01 的天数；`usefulCount_log`：double，`log1p(usefulCount)`。投票是在评论发表后累积的，解释模型应用场景时需说明输入条件。
- `rating`：原始整数评分 1–10，仅作真值；`rating_p1`：整数分类标签 0–9。
- `condition_idx`、`condition_vec`：类别索引与 Spark ML one-hot 向量。模型可使用 `condition_vec`。
- `drugName_idx`、`drug_mean_rating`：辅助类别索引与平滑的目标均值编码。`drugName_idx` 不是连续数值特征，不能直接作为线性模型的有序数值。
- 训练集另有 double 列 `weight = n / (k * class_count)`，测试集没有。它只能作为训练权重，不能作为输入特征。

公共 encoder 只在训练集拟合，使用 `handleInvalid="keep"`。药物编码使用 `TargetEncoder(targetType="continuous", labelCol="rating", smoothing=10.0)`，未知药物映射到训练全局均值。模型任务沿用此接口；调整编码方法属于公共预处理任务。

## 3. 注册规则

每种模型一个顶层 `.py` 文件，train 与 predict 使用同一个独有注册名。名称用小写字母、数字或单下划线，不含 `__` 或路径分隔符。函数名也要区分，例如 `my_embedding_train`、`my_embedding_predict`，避免互相覆盖基类上的同名属性。

**模型模块从对应的 `base` 导入基类**，因为自动发现发生在包初始化过程中；notebook 才使用 `from src.embeddings import EmbeddingModel` / `from src.analytics import AnalyticsModel`。

导入模块时只定义函数、类和常量。不要在模块顶层训练、下载权重、写文件或创建 SparkSession。torch、transformers 等可选模型库可放在实际使用它们的函数中导入，并把新增依赖交给集成人登记。这可以保留现有自动注册，同时避免其他模型被无关依赖阻断。

下面仅示意接口，不能直接替代模型实现。完整参考是 `src/embeddings/tfidf.py` 和 `src/analytics/logistic_regression.py`。

## 4. Embedding 接口

```python
from pyspark.sql import DataFrame
from src.embeddings.base import EmbeddingModel
from src.paths import EMBEDDING_MODELS_DIR

@EmbeddingModel.train_register("my_embedding")
def my_embedding_train(self) -> None:
    ...

@EmbeddingModel.predict_register("my_embedding")
def my_embedding_predict(self) -> tuple[DataFrame, DataFrame]:
    ...
```

- `train(self)`：只在 `self.train` 上拟合，模型保存到 `EMBEDDING_MODELS_DIR / "my_embedding"`。加载预训练模型不等于允许在测试集做适配或调参。
- `predict(self)`：加载自身模型，对 `self.train`、`self.test` 变换，并按这个顺序返回两个 DataFrame。不要在预测阶段重新拟合词表或任何学习得到的变换。
- 返回表保留全部原有列和样本，并新增 `embedding`。其类型必须为 Spark ML `VectorUDT`，例如 DenseVector 或 SparseVector；数组 / Python list 应在模型内部转为向量。
- 同一模型的训练 / 测试向量维数一致，不能有 null、NaN 或无限值。不可为了处理失败样本而静默删行或改变 ID；模型内部必须给出保持样本覆盖的处理方式。
- 模型专用的文本处理可以放在自己文件里，但不覆盖公共 `review`、`words` 或 `words_stem`。

## 5. Analytics 接口

```python
from pyspark.sql import DataFrame
from src.analytics.base import AnalyticsModel
from src.paths import ANALYTIC_MODELS_DIR

@AnalyticsModel.train_register("my_classifier")
def my_classifier_train(self, embedding_model: str) -> None:
    ...

@AnalyticsModel.predict_register("my_classifier")
def my_classifier_predict(self, embedding_model: str) -> tuple[DataFrame, DataFrame]:
    ...
```

`embedding_model` 是表示的注册名字符串，例如 `tfidf`，不是模型对象。实例保存 `self.train_embedding` / `self.test_embedding`，其中包含上一节的列和 `embedding`。

训练与预测使用同一份特征列表；需要组合列时调用 `train_df, test_df = self._feature_transform(features)`，得到新的 `features` 列。辅助函数不修改实例保存的原 DataFrame。仅声明 `FEATURES` 常量不会自动组装向量，需实际调用该函数；模型也可以在自己文件里实现合适的其他组装方式。

分类器在训练集用 `labelCol="rating_p1"` 拟合。不得把 `rating`、`rating_p1` 或 `weight` 拼入输入特征。若加权，用分类器的权重参数读取训练集 `weight`。缩放器、降维器等也只在训练集拟合，并与模型一起保存以供预测复用。

模型保存到 `ANALYTIC_MODELS_DIR / f"my_classifier__{embedding_model}"`，不同表示不能共用同一模型目录。预测按 train / test 顺序返回，每个表恰好三列：

- `uniqueID`：保留原 ID。
- `rating`：保留 1–10 的原评分。
- `prediction`：double 类型、取整数值的 0–9 类别编号。

不要在分类器里加 1，不要返回概率向量或连续回归值代替类别。MAE / RMSE 只是分类结果的补充评分，不表示需要实现另一个回归任务。

## 6. 保存、评分与实验入口

模型 owner 只保存自己的拟合模型。Spark 保存前只创建模型目录的父目录；训练 / 测试 DataFrame 由现有 runner 通过 `DataOI` 写入。

路径约定：

- `data/processed/split/{train,test}.parquet`：已编码 split。
- `data/processed/embeddings/<embedding>/{train,test}.parquet`：向量表。
- `data/processed/predictions/<analytics>__<embedding>/{train_predictions,test_predictions}.parquet`：预测表。
- `src/embeddings/models/<embedding>/` 与 `src/analytics/models/<analytics>__<embedding>/`：拟合产物。

组合名按 `analytics__embedding` 排列，例如 `lr_weighted__tfidf`。评分表保留既有列名 `Embedding_Model` 和 `Analytic_Model`，不要自行重命名。`Scorer` 创建 `rating_pred = prediction + 1`，再与 `rating` 比较。Accuracy 与 support-weighted Precision / Recall / F1 越大越好；MAE / RMSE 用原始评分单位，越小越好。相同组合重加时只替换该组合，其他组合保留。

`embedding_pipeline.ipynb` 只训练选中的 `embedding_name`，设为 `tfidf` 即可检查 baseline 表示的流程，再运行已有 analytics；`analytics_pipeline.ipynb` 只训练选中的 `analytics_name`，使用本机已准备的所有 embedding。个人 runner 只把当前 `test_pred_dict` 交给 Scorer。主 runner 做全量集成，不改成针对个人模型的特例。

`REBUILD` 仅控制输入准备，具体阶段见 README。新环境需要先生成本地缓存；有缓存时可以只重跑模型阶段。现有加载器枚举本地目录，半成品目录需清理或补跑。模型变化后重建受影响的下游结果，避免把旧预测当作新模型表现。

## 7. 交接前自查

1. 只改分配的模型文件；可选依赖和共享变更需求已经说明。
2. train / predict 成对注册，包导入时能发现；导入本身不触发训练。
3. 对同一输入，embedding 保留样本及列、向量维数一致；analytics 返回恰好三列，标签与预测尺度正确。
4. 拟合仅使用训练数据，预测加载相匹配的模型，保存路径不覆盖他人产物。
5. 小样本能跑通该模型的训练 / 预测 / 评分；记录实际跑过的组合、参数、seed、依赖和结果。未跑的实验不能写成已通过。
6. 拟合模型可交接；`data/processed/` 不提交。提交 notebook 前清空输出，未经用户要求不自行 commit 或 push。

新增或修改模型模块后，最简单的刷新方式是重启 kernel 后从初始化运行。不要只 reload 基类：注册表与已导入的旧模块可能不同步。调试时 `importlib.reload` 接收模块而不是类，旧实例也不会自动变成新类。
