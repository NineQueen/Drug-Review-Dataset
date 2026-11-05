# 模型接入与职责说明

本项目预测 1–10 分的十分类 `rating`。AI 先读根目录 [AGENTS.md](../AGENTS.md)，再按当前分配的模型文件完成任务。本文记录现有代码的接口，不要求模型成员重构公共流程。

## 1. 职责与最小改动范围

embedding 成员只实现 `src/embeddings/<name>.py`：选择原评论或 token，拟合表示，保存自身模型，输出向量。analytics 成员只实现 `src/analytics/<name>.py`：选择自身特征，在同一个注册函数里拟合分类器并返回预测。模型内部需要的辅助函数、缩放或降维放在该模型文件里。

集成人负责 `base.py`、`__init__.py`、路径、I/O、预处理、划分、编码、评分、共享 notebook、依赖与最终材料。成员发现接口或共享逻辑需要变化时，说明具体原因和影响，由集成人统一处理；不要顺手修改其他模型或清洗规则。

成员只需在现成实验入口中设置自己的注册名和重建开关，不需要改 notebook 的公共循环。依赖的输入不存在时先按 README 准备数据，不能通过临时改共享路径、标签或划分来让单个模型跑通。姓名和具体模型分工由小组另行确定，本文不代为分配。

## 2. 数据流与输入

主流程依次执行：原始 CSV → EDA → 清洗 → 文本特征 → ID 哈希划分和训练权重 → 类别编码 → embedding 训练 / 变换 → analytics → 评分。embedding 先跑完已注册的训练函数，再跑对应的预测函数。每个 analytics 只调用一次已注册函数，这次调用同时拟合并返回预测。

清洗保持整行过滤、已配置 condition / HTML 替换、成对的外层包装引号移除、普通空白合并、小写化、确定性 review 去重及数值衍生列。包装引号只在两端同时是同一个标记时去掉一对：先是 `"`，再是 `&quot;`；只有一侧有引号时保留。普通空白在这次判断前、以及 HTML 解码后再各合并一次。字符与 Unicode 空白、缩写展开、剂量空格，以及后续文本处理，统一放在文本特征阶段。这些已实现的通用修复先写回 `review`，再复制临时 `review_clean` 准备 token；分词、停用词和词干不覆盖 `review`。停用词保留否定词。主流程调用 `text_feature_engineering` 时打开 bigrams。`baseline_text_feature_engineering` 只对 `review` 做 Tokenizer，用于单独的 baseline TF-IDF。文本阶段不改变行、ID 或标签，也不二次去重或新增永久文本列。后续新增规则只面向普遍问题，不维护数据集专属错拼或逐条评论例外。规范化碰撞需在比较模型前单独检查，不能把 token 或词干相同直接当成重复评论。

划分用 `pmod(xxhash64(uniqueID, SEED), 10000)`。桶号小于 7000 为训练，7000 至 7999 为验证，8000 及以上为测试，比例约 70/10/20。默认 `SEED = 42`。固定数据、ID 类型、Spark 版本和 seed 后，归属不受顺序或分区影响。不要改成 `randomSplit`。预处理后样本集合变化时，需要统一重建和记录新基准，不能继续混用旧结果。

模型接收三份数据，构造时按 train、valid、test 传入。

- `train`：拟合词表、IDF、编码器和分类器。类别权重 `weight` 只写在这份数据上。
- `valid`：验证集。列与训练集相同，但没有 `weight`。不在这里拟合，也不把这里的指标当作最终分数。用它检查已经拟合的模型。
- `test`：测试集。不参与拟合，也不用来选择模型。`Scorer` 只对测试预测评分。

embedding 上的三份表是 `self.train`、`self.valid`、`self.test`。它们保留以下列：

- `uniqueID`：整数型样本 ID，一行一条评论，必须保留，不能重编。
- `drugName`、`condition`、`review`：字符串。`review` 保存清洗以及字符、空白、缩写和剂量修复后的完整评论。分词和词干不覆盖此列。
- `words`、`words_featured`：`array<string>`。`words` 是未词干化 token，`words_featured` 是 TF-IDF 使用的词干。其他表示可直接读取 `review` 或 `words`。
- `days`：整数，距 1970-01-01 的天数；`usefulCount_log`：double，`log1p(usefulCount)`。投票是在评论发表后累积的，解释模型应用场景时需说明输入条件。
- `rating`：原始整数评分 1–10，仅作真值；`rating_p1`：整数分类标签 0–9。
- `condition_idx`、`condition_vec`：类别索引与 Spark ML one-hot 向量。模型可使用 `condition_vec`。
- `drugName_idx`：训练集拟合的药物类别索引。当前不加权 baseline 和加权 `lr_improved` 把它和 `embedding`、`condition_vec`、`days`、`usefulCount_log` 一起组装进特征。当前 encoder 不生成 `drug_mean_rating`。
- 训练集另有 double 列 `weight = n / (k * class_count)`。验证集和测试集没有这一列。它只能作为训练权重，不能作为输入特征。加权 LR 读取这一列；不加权 baseline 不读取。

公共 encoder 只在训练集拟合，使用 `handleInvalid="keep"`。它拟合 `condition` 的 StringIndexer 与 one-hot，以及 `drugName` 的 StringIndexer，再变换 validation 和 test，使三份数据的类别索引和向量维数一致。调用形式是 `encoder(train, valid, test)`。修改这个公共编码后必须统一重建 split、embedding、分类器和预测；模型负责人不自行修改它。

## 3. 注册规则

每种模型一个顶层 `.py` 文件。注册名用小写字母、数字或单下划线，不含 `__` 或路径分隔符。函数名不要重复，避免互相覆盖基类上的同名属性。

embedding 的 train 与 predict 使用同一个注册名，例如 `my_embedding_train` 和 `my_embedding_predict`。analytics 只注册一个函数：普通分类器用 `@AnalyticsModel.register("my_classifier")`，必需的不加权 baseline 用 `@AnalyticsModel.baseline_register()`。runner 对普通分类器调用 `AnalyticsModel.registry[key](instance)`，对 baseline 调用 `instance.baseline()`。

**模型模块从对应的 `base` 导入基类**，因为自动发现发生在包初始化过程中；notebook 才使用 `from src.embeddings import EmbeddingModel` / `from src.analytics import AnalyticsModel`。

导入模块时只定义函数、类和常量。不要在模块顶层训练、下载权重、写文件或创建 SparkSession。torch、transformers 等可选模型库可放在实际使用它们的函数中导入，并把新增依赖交给集成人登记。这可以保留现有自动注册，同时避免其他模型被无关依赖阻断。

下面仅示意接口，不能直接替代模型实现。完整参考是 `src/embeddings/tfidf.py` 和 `src/analytics/logistic_regression.py`。

## 4. Embedding 接口

```python
from pyspark.sql import DataFrame
from src.embeddings.base import EmbeddingModel
from src.paths import EMBEDDING_MODELS_DIR

@EmbeddingModel.train_register("my_embedding")
def my_embedding_train(self, model_name: str = "my_embedding") -> None:
    model = ...  # fit on self.train
    # save to EMBEDDING_MODELS_DIR / model_name

@EmbeddingModel.predict_register("my_embedding")
def my_embedding_predict(self, model_name: str = "my_embedding") -> tuple[DataFrame, DataFrame, DataFrame]:
    model = ...  # load EMBEDDING_MODELS_DIR / model_name
    # return transformed train, valid, test
```

`model_name` 是磁盘目录名，默认可与注册名相同；基线可以传入另一个名字，例如 `"baseline"`。

- `train`：只在 `self.train` 上拟合，并保存到 `EMBEDDING_MODELS_DIR / model_name`。加载预训练模型不等于允许在验证集或测试集做适配或调参。
- `predict`：从该目录加载已保存的模型，变换 `self.train`、`self.valid`、`self.test`，按这个顺序返回。不要在预测阶段重新拟合词表或任何学习得到的变换。
- 返回的三份表都保留原有列和样本，并新增 `embedding`。其类型必须为 Spark ML `VectorUDT`，例如 DenseVector 或 SparseVector；数组 / Python list 应在模型内部转为向量。
- 同一模型在训练、验证、测试上的向量维数一致，不能有 null、NaN 或无限值。不可为了处理失败样本而静默删行或改变 ID；模型内部必须给出保持样本覆盖的处理方式。
- 模型专用的文本处理可以放在自己文件里，但不覆盖公共 `review`、`words` 或 `words_featured`。

## 5. Analytics 接口

```python
from pyspark.sql import DataFrame
from src.analytics.base import AnalyticsModel

@AnalyticsModel.register("my_classifier")
def my_classifier(self) -> tuple[DataFrame, DataFrame, DataFrame]:
    train_df, valid_df, test_df = self._feature_transform(features)
    model = ...  # fit on train_df only
    # return train, valid, and test predictions

@AnalyticsModel.baseline_register()
def my_baseline(self) -> tuple[DataFrame, DataFrame, DataFrame]:
    ...  # the required unweighted baseline; called as instance.baseline()
```

构造 `AnalyticsModel` 时传入 embedding 名称和三份向量表。`embedding_model` 是表示的注册名字符串，例如 `tfidf`，不是模型对象，并保存在 `self.embedding_model`。三份表分别是 `self.train_embedding`、`self.valid_embedding`、`self.test_embedding`，其中包含上一节的列和 `embedding`。一个实例对应一个 embedding 名称；换表示时另建一个实例。

不要把训练和预测拆成两个函数。注册函数在训练集上拟合，并立刻按 train、valid、test 返回预测。runner 调用 `AnalyticsModel.registry["my_classifier"](instance)`；baseline 调用 `instance.baseline()`。

需要组合列时调用 `train_df, valid_df, test_df = self._feature_transform(features)`，得到新的 `features` 列。辅助函数不修改实例保存的原 DataFrame。仅声明 `FEATURES` 常量不会自动组装向量，需实际调用该函数；模型也可以在自己文件里实现合适的其他组装方式。

分类器在训练集用 `labelCol="rating_p1"` 拟合。不得把 `rating`、`rating_p1` 或 `weight` 拼入输入特征。若加权，用分类器的权重参数读取训练集 `weight`。缩放器、降维器等也只在训练集拟合。

返回的三份预测按 train、valid、test 顺序排列，每个表恰好三列：

- `uniqueID`：保留原 ID。
- `rating`：保留 1–10 的原评分。
- `prediction`：double 类型、取整数值的 0–9 类别编号。

不要在分类器里加 1，不要返回概率向量或连续回归值代替类别。MAE / RMSE 只是分类结果的补充评分，不表示需要实现另一个回归任务。

## 6. 保存、评分与实验入口

模型 owner 只保存自己的拟合模型。Spark 保存前只创建模型目录的父目录；训练、验证、测试 DataFrame 由现有 runner 通过 `DataOI` 写入。

路径约定：

- `data/processed/split/{train,valid,test}.parquet`：已编码的训练、验证、测试 split。
- `data/processed/embeddings/<embedding>/{train,valid,test}.parquet`：向量表。
- `data/processed/predictions/<analytics>__<embedding>/{train_predictions,valid_predictions,test_predictions}.parquet`：预测表。
- `src/embeddings/models/<embedding>/` 与 `src/analytics/models/<analytics>__<embedding>/`：拟合产物。

组合名按 `analytics__embedding` 排列，例如 `lr_improved__tfidf`。评分表保留既有列名 `Embedding_Model` 和 `Analytic_Model`，不要自行重命名。`Scorer` 创建 `rating_pred = prediction + 1`，再与 `rating` 比较。Accuracy 与 support-weighted Precision / Recall / F1 越大越好；MAE / RMSE 用原始评分单位，越小越好。相同组合重加时只替换该组合，其他组合保留。

`embedding_pipeline.ipynb` 只训练选中的 `embedding_name`，设为 `tfidf` 即可检查 baseline 表示的流程，再对 `AnalyticsModel.registry` 里的分类器各调用一次。`analytics_pipeline.ipynb` 只运行选中的 `analytics_name`，这个名字是 `register` 的注册名，使用本机已准备的所有 embedding。两个入口都按 train、valid、test 评分。验证分数用来检查模型，最终比较看测试分数。主 runner 做全量集成，不改成针对个人模型的特例。不加权 baseline 由 `baseline()` 调用，不在 `registry` 里。

`REBUILD` 仅控制输入准备，具体阶段见 README。新环境需要先生成本地缓存；有缓存时可以只重跑模型阶段。现有加载器枚举本地目录，半成品目录需清理或补跑。模型变化后重建受影响的下游结果，避免把旧预测当作新模型表现。

## 7. 交接前自查

1. 只改分配的模型文件；可选依赖和共享变更需求已经说明。
2. embedding 的 train / predict 成对注册；analytics 只注册一个函数。包导入时能发现；导入本身不触发训练。
3. 对同一输入，embedding 在 train、valid、test 上保留样本及列，向量维数一致；analytics 的三份预测表各有恰好三列，标签与预测尺度正确。
4. 拟合仅使用训练数据。embedding 保存到自己的目录，预测从该目录加载并按 train、valid、test 返回。analytics 在同一个函数里拟合并按这个顺序返回三份预测。验证集不参与拟合，测试集不用于挑选模型。保存路径不覆盖他人产物。
5. 小样本能跑通该模型并完成评分；记录实际跑过的组合、参数、seed、依赖和结果。未跑的实验不能写成已通过。
6. 拟合模型可交接；`data/processed/` 不提交。提交 notebook 前清空输出，未经用户要求不自行 commit 或 push。

新增或修改模型模块后，最简单的刷新方式是重启 kernel 后从初始化运行。不要只 reload 基类：注册表与已导入的旧模块可能不同步。调试时 `importlib.reload` 接收模块而不是类，旧实例也不会自动变成新类。
