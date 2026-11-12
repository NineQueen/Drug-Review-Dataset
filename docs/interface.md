# 模型接入

预测 1–10 分的十分类 `rating`。本文是现有代码的接口。模型成员按分配的文件实现，不重构公共流程。姓名和模型分工由小组确定。

## 1. 改动范围

embedding 成员只改 `src/embeddings/<name>.py`：读取已有文本列，拟合表示，保存自己的模型，输出 `embedding`。analytics 成员只改 `src/analytics/<name>.py`：读取 `embedding` 和已经编码的非文本列，在一个注册函数里拟合分类器并返回三份预测。分类器里只对已经组装好的特征向量做训练集上的缩放或降维。

集成人维护 `base.py`、`__init__.py`、路径、I/O、预处理、划分、编码、评分、共享 notebook、依赖和最终材料。接口需要变化时，说明原因和影响，由集成人改。输入不存在时按 README 准备数据。

## 2. 数据

主流程是：原始 CSV、EDA、清洗、文本特征、哈希划分和训练权重、类别编码、embedding、analytics、评分。已注册 embedding 先训练再预测。每个普通分类器调用一次注册函数，这次调用同时拟合并返回预测。

会改变预测的量只从训练集得到，包括词表、IDF、编码器、缩放器、降维器、分类器、阈值和选出的特征子集。分类器函数内部不读取验证集或测试集来拟合、校准或选参。函数返回后可以给验证预测打分。最终比较用测试预测。

清洗保留整行过滤、已配置的 condition / HTML 替换、成对的外层包装引号、普通空白合并、小写化、确定性 review 去重和数值列。包装引号只在两端是同一个标记时去掉一对：先是 `"`，再是 `&quot;`。只有一侧有引号时保留。普通空白在这次判断前、以及 HTML 解码后再各合并一次。

`text_feature_engineering` 把字符与 Unicode 空白、缩写展开和剂量空格写回 `review`，再复制临时列做 token。分词、停用词和词干不覆盖 `review`。停用词保留 `not`、`no`、`nor`、`never`、`neither`、`without`、`cannot`、`but`、`only`、`very`、`too`、`more`、`most`。主流程打开 bigrams。文本阶段不删行、不改 ID 或标签，也不二次去重。`baseline_text_feature_engineering` 只对 `review` 做 Tokenizer，给单独的 TF-IDF 路径用。新增文本规则只处理普遍问题。规范化碰撞在比较模型前单独检查。

`fit_phrases(train_df)` 只在训练集 `words` 上拟合，参数固定为 `min_count=20`、`scoring="npmi"`、`threshold=0.5`、`delimiter="_"`。它返回冻结规则，不另存短语目录，也不接收参数字典。`apply_phrases` 在新表上把 `words_nostop` 换成短语 token。构造 `EmbeddingModel(..., phrases)` 后得到 `train_phrase`、`valid_phrase`、`test_phrase`。`train_expanded` 把训练集词列与 WebMD 评论合并，并去掉验证集和测试集里已经出现的 `review`。`train_expanded_phrase` 套用同一份短语；没传入短语时不能读取。GloVe 在 `train_expanded_phrase` 上训练，并在三份短语表上预测。TF-IDF 读取 `self.train` 的 `words_featured`。

划分用 `pmod(xxhash64(uniqueID, seed), 10000)`。`split` 的正整数比例有几项就切几份，例如 `(8, 2)` 或 `(7, 1, 2)`。默认 `(7, 1, 2)`：桶号小于 7000 为训练，7000 至 7999 为验证，8000 及以上为测试。不能整除 10000 时，余数进最后一份。某一份的实际占比偏离请求占比超过该占比的一半时，`split` 报错；这表示这个 seed 已经切过这些行，桶发生了重叠。`weight` 给一张表写入归一化平方根权重 `n / (sqrt(class_count) * Σ sqrt(class_count))`，样本权重之和为 `n`，算完后卸掉输入表的缓存，通常只调用在拟合用的那一份上。种子只在 `src/seed.py`：`SEED = 42`，`seed(level)` 是 `SEED + level`。全表用 `seed(0)`，训练池上的 K 折用 `seed(1)`，折内早停用 `seed(2)`。固定数据、ID 类型、Spark 版本、seed 和比例后，归属不受行顺序影响。预处理后的样本集合变了，就统一重建下游结果。

`KFold(model_name, df, k, prepare_function, SEED=seed(1))` 用 `split(df, SEED, (1,) * k)`。每一折把其余份拼成拟合表，留出一份作验证表，调用 `prepare_function(fit, holdout)`，再把返回的两张表写到 `data/processed/temp/<model_name>/fold_<i>/`。每一折的新缓存放在 `data_oi.scope()` 里，写出后释放。`scope()` 卸掉块内新建的 DataFrame 缓存，包括 SQL 登记。`next_fold()` 读取下一折并卸掉上一折。`reset()` 回到第一折。`KFold.open(model_name, k)` 只读取已有目录。默认种子是 `seed(1)`，因为传入的表通常已经用 `seed(0)` 做过 7:1:2。K 折只有拟合和留出两份；embedding 和分类器的第三份都用拟合表的一行。

embedding 上的三份表是 `self.train`、`self.valid`、`self.test`。

- `uniqueID`：整数 ID，一行一条评论，原样保留。
- `drugName`、`condition`、`review`：字符串。`review` 是清洗和文本修复后的评论。
- `words`、`words_nostop`、`words_featured`：`array<string>`。`words` 未词干化，`words_nostop` 去掉停用词并保留上面列出的否定词和程度词，`words_featured` 是 TF-IDF 用的词干。analytics 不读取这些列。
- `days`：距 1970-01-01 的天数。`usefulCount_log`：`log1p(usefulCount)`。有用票在评论发表后累积。
- `rating`：原始评分 1–10。`rating_p1`：分类标签 0–9。
- `condition_idx`、`condition_vec`：类别索引和 one-hot。
- `drugName_grouped`、`drugName_idx`、`drugName_vec`：训练集里出现至少 10 次的药名保留，其余并入 `__other__`。分类器使用已经生成的 `drugName_vec`，不另算药名均分或其他由标签得到的列。
- 训练集另有 `weight`。验证集和测试集没有这一列。它只作训练权重，不作输入特征。不加权 `baseline()` 不读取它；加权 `improved_baseline()` 读取它。

公共 encoder 只在第一张表上拟合，`handleInvalid="keep"`，再变换后面每一张表。后面的表可以不传。模型负责人不改这个编码。

## 3. 注册

每种模型一个顶层 `.py` 文件。注册名用小写字母、数字或一个下划线，不含 `__` 或路径分隔符。函数名不要重复。

embedding 的 train 与 predict 使用同一个注册名。TF-IDF 用 `baseline_train_register` 和 `baseline_predict_register`，调用 `baseline_train()` 与 `baseline_predict()`，不进入 `train_registry` 或 `predict_registry`。`baseline_train(model_name, para=None)` 里传入的键覆盖 CountVectorizer 的 `inputCol`、`outputCol`、`vocabSize`、`minDF`，缺的键保留这些默认值。普通分类器用 `register`。不加权 baseline 用 `baseline_register`，调用 `instance.baseline()`，不接收 `para`。加权 LR 用 `improved_baseline_register`，调用 `instance.improved_baseline(para=None)`；传入的键覆盖 `DEFAULT_PARA`，缺的键保留默认值。这两个函数都不在 `registry` 里。

主 notebook 的 baseline 段把只做分词的 TF-IDF 存为 `tfidf`，只调用 `baseline()`；把带文本特征的 TF-IDF 存为 `tfidf_improved`，只调用 `improved_baseline()`。共享 runner 对普通分类器调用 `AnalyticsModel.registry[key](instance)`，因此使用默认 `para`。个人调参可以传入同一形状的 `para`。

模型模块从 `src.embeddings.base` 或 `src.analytics.base` 导入基类。notebook 使用 `from src.embeddings import EmbeddingModel` 和 `from src.analytics import AnalyticsModel`。模块顶层不训练、不下载、不写文件、不创建 SparkSession。torch、transformers 等放在使用它们的函数里导入。

## 4. Embedding

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

`model_name` 是磁盘目录名。`baseline_predict()` 不传名字时目录名是 `baseline`；主 notebook 传入 `tfidf` 和 `tfidf_improved`。GloVe、Word2Vec 和 BERT 的 `train` / `predict` 还可以传 `model_dir`；省略时仍是 `EMBEDDING_MODELS_DIR / model_name`。

构造函数是 `EmbeddingModel(train, valid, test, phrases=None)`。

- `train` 只在 `self.train` 上拟合，保存到 `EMBEDDING_MODELS_DIR / model_name`。
- `predict` 从该目录加载模型，按 train、valid、test 返回。预测阶段不重新拟合词表。
- 返回表保留全部样本，并带有 `embedding`。注册好的 `predict` 随后只留下 `uniqueID`、`rating`、`rating_p1`、训练集上的 `weight`、`condition_idx`、`condition_vec`、`days`、`usefulCount_log`、`drugName_idx`、`drugName_vec` 和 `embedding`。`embedding` 是 Spark ML `VectorUDT`。三份表的向量维数一致，不含 null、NaN 或无限值。失败样本不删行、不改 ID。
- 模型自己的文本处理可以放在该文件里，但不覆盖公共 `review`、`words`、`words_nostop` 或 `words_featured`。

## 5. Analytics

`DEFAULT_PARA` 是这一组构造参数，所有 embedding 共用。`PARA_RANGE` 是同一批键的候选列表。runner 不读取 `PARA_RANGE`，也不规定搜索过程。两个字典都不放 embedding 名称、特征列、标签、路径或流程开关。是否使用 `weight` 由模型写死；要用时，把权重参数留在 `DEFAULT_PARA` 里。

```python
from pyspark.sql import DataFrame
from src.analytics.base import AnalyticsModel

FEATURES = ["embedding", "condition_vec", "days", "usefulCount_log", "drugName_vec"]

DEFAULT_PARA = {"regParam": 0.1, "maxIter": 100}
PARA_RANGE = {"regParam": [0.01, 0.1, 0.3, 1.0], "maxIter": [50, 100]}

@AnalyticsModel.register("my_classifier")
def my_classifier(self, para: dict = DEFAULT_PARA) -> tuple[DataFrame, DataFrame, DataFrame]:
    train_df, valid_df, test_df = self._feature_transform(FEATURES)
    model = ...  # construct with **para; fit on train_df
    # return train, valid, and test predictions

@AnalyticsModel.baseline_register()
def baseline(self) -> tuple[DataFrame, DataFrame, DataFrame]:
    ...  # unweighted baseline; called as instance.baseline()

@AnalyticsModel.improved_baseline_register()
def improved_baseline(self, para: dict | None = None) -> tuple[DataFrame, DataFrame, DataFrame]:
    ...  # weighted LR; passed keys override DEFAULT_PARA
```

当前 logistic regression 和 SVM 的 `FEATURES` 含 `drugName_vec`。random forest 不含这一列。同一个分类器对所有 embedding 使用同一份特征列。

构造 `AnalyticsModel` 时传入 embedding 名称和三份向量表。`embedding_model` 是注册名字符串，保存在 `self.embedding_model`。可读列是 `embedding`、`condition_idx`、`condition_vec`、`days`、`usefulCount_log`、`drugName_idx`、`drugName_vec`。可以少用其中几列。换表示时另建实例，仍然调用同一个函数。

特征列写成模块常量，并调用 `self._feature_transform(FEATURES)`。只声明 `FEATURES` 不会自动组装。分类器在训练集上用 `labelCol="rating_p1"` 拟合。缩放器和降维器只作用于已组装的 `features`，并且只在训练集拟合。注册函数不保存分类器，也不为缺列或缺文件增加分支。

三份预测按 train、valid、test 排列。每份含 `uniqueID`、`rating`（1–10）、`prediction`（double，取值 0–9）和 `probability`（长度为 10 的 `array<double>`，下标 `i` 是类别 `i`）。逻辑回归、随机森林、XGBoost 和 MLP 写入类别概率。SVM 的同一列是 10 个 one-vs-rest 间隔分数。`Scorer` 只使用 `prediction`，计算 `rating_pred = prediction + 1`。MAE / RMSE 用原始评分单位。

## 6. 保存、评分和入口

embedding 只保存自己的拟合表示。Spark 保存前只创建模型目录的父目录。DataFrame 由 runner 通过 `DataOI` 写入。

- `data/processed/split/{train,valid,test}.parquet`
- `data/processed/embeddings/<embedding>/{train,valid,test}.parquet`
- `data/processed/predictions/<analytics>__<embedding>/{train_predictions,valid_predictions,test_predictions}.parquet`
- `src/embeddings/models/<embedding>/`

普通组合名是 `analytics__embedding`，例如 `random_forest__glove`。主 notebook 的 baseline 段写出 `lr__tfidf` 和 `lr_improved__tfidf_improved`。评分表的列名是 `Embedding_Model` 和 `Analytic_Model`。行按这两个名字的字典序排列。指标是 accuracy、support-weighted precision 和 F1；support-weighted recall 与 accuracy 相同，不再单列。相同组合重加时只替换该组合。

共享的 `train_register` 把已注册 embedding 的训练调用包在 `data_oi.scope()` 中，训练返回后释放新建缓存，保留进入作用域前已有的输入缓存。主入口调用 `load_prediction_data()`，一次读取全部已保存预测。传入 `model_names` 时只读取列出的组合；个人入口用这个参数，避免混入历史实验。

`embedding_pipeline.ipynb` 训练选中的 `embedding_name`，名字来自 `train_registry` / `predict_registry`，再对 `AnalyticsModel.registry` 各调用一次。`analytics_pipeline.ipynb` 运行选中的 `register` 名，并遍历 `predict_registry`，因此不含 TF-IDF。`REBUILD = True` 时训练并保存这些表示，`False` 时调用 `load_embedding_data(name)`。两个入口都评 train、validation、test；汇报比较用 test。

`experiments/para_tuning/para_tuning_pipeline.ipynb` 用 `getattr` 取写下的 `EMBEDDING_MODEL` 和 `ANALYTICS_MODEL`。`REBUILD = True` 时先取默认训练池，再用 `seed(1)` 做 K 折。`prepare_fold` 完成 `weight`、`encoder`、`fit_phrases` 和这个 embedding，返回拟合向量和留出向量。表示写到 `{EMBEDDING_MODEL}_para_{fold}`，折表写到 `data/processed/temp/{EMBEDDING_MODEL}_para/fold_<i>/`。`REBUILD = False` 时用 `KFold.open`。参数循环每次 `reset()` 后 `next_fold()`。参数按留出折的 support-weighted F1 平均值最大化，与评分表的 F1 相同。写入 `{EMBEDDING_MODEL}_{ANALYTICS_MODEL}.json` 时同时保存 `weighted_f1`。文件里已有分数时，只有新分数更高才覆盖；没有这个分数的已有文件保持原样。`load_analytics_para` 要求文件带有 `weighted_f1`，返回前去掉它和两个名字字段，所以返回值只含构造参数。向量写出前先卸掉编码表和扩展语料，否则 scope 会把还没写完的向量一起释放。

`REBUILD` 的各阶段见 README。向量按注册名读取，目录里要同时有 train、valid、test。模型变化后重建受影响的下游结果。

## 7. Stacking

`StackingModel` 在 `src/stacking/stacking.py`，不进入 embedding 或 analytics 注册表。

`load_data()` 读取 encoder 之前的 `data/processed/split/train_raw.parquet`。`KFold` 使用 `seed(1)`。每一折先 `weight` 再 `encoder`。只有 GloVe 在该折的 fit 上调用 `fit_phrases`。外层循环是 `predict_registry`，内层是 `registry`，所以未注册的模型不会进入。TF-IDF 和 logistic regression 不在这两个表里。每个分类器调用 `load_analytics_para(embedding, analytics)`，和主 notebook 一样。一次 embedding 的训练、预测和分类器写出放在同一个 `DataOI.scope()` 里，写出后退出。GloVe 内部缓存的完整帧和投影后的帧一起释放。折的 fit 和 holdout 在进入这个 scope 之前已经缓存，不会被卸掉。折内表示把 `model_dir` 设为 `data/processed/stacking/models`，写到 `stacking_f<fold>_<embedding>/`，分类器跑完由 `DataOI` 删除。全局的 `bert`、`glove`、`word2vec` 仍在 `src/embeddings/models`。留出折只保留 `uniqueID`、`rating`、`probability`，由 `DataOI` 合并到 `data/processed/stacking/oof/<analytics>__<embedding>/`。`manifest.json` 由 `DataOI` 读写。有序逻辑回归写到 `src/stacking/ordinal.json`。

`train()` 按 `manifest.json` 里的顺序拼接这些长度为 10 的输出，得到 `features` 和 `rating_p1`。二层只有有序逻辑回归：一条线性分数加一组递增阈值。分数在折外样本上标准化，阈值 0 自由，其余间隔用 softplus 保证递增，类别概率由累计概率的差分得到。`predict()` 读取同名的全量预测，返回 train、valid、test。每份含 `uniqueID`、`rating`、`prediction` 和 `probability`。`Scorer` 仍用 `prediction + 1`。主 notebook 把这一份预测写到 `stacking__stacking`。

## 8. 交接前核对

1. 只改了分配的模型文件；可选依赖和共享变更已经说明。
2. embedding 的 train / predict 成对注册；analytics 只注册一个函数。导入包能发现它，导入本身不训练。
3. 同一输入上，embedding 保留样本且向量维数一致；analytics 的三份预测各有三列。
4. 会改变预测的量只从训练集得到。验证集和测试集不参与拟合。
5. 小样本跑通过，并记录实际跑过的组合、参数、seed、依赖和结果。
6. `data/processed/` 不提交。提交 notebook 前清空输出。未经要求不 commit 或 push。

新增或修改模型模块后，重启 kernel 再从初始化运行。
