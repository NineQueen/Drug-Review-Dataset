# DSAI4205 Big Data Analytics — UCI Drug Review

用 PySpark 把药品评论的 `rating` 分成 1–10 十类。共享实现在 `src/`，集成入口是 `project.ipynb`。指定 baseline 是 TF-IDF 加不加权 Logistic Regression，另有一份加权 LR。列、注册和输出约定见 [docs/interface.md](docs/interface.md)。AI 的修改边界见 [AGENTS.md](AGENTS.md)。

## 环境与数据

已验证的组合是 Python 3.12、PySpark 4.2.0、Java 21。把 `JAVA_HOME` 指到 Java 21。`DataEnv` 只是本机环境名。Jupyter 要选装了依赖的同一个 Python。

```bash
conda activate DataEnv
python -m pip install -r requirements.txt
python -m nltk.downloader averaged_perceptron_tagger_eng wordnet
```

没有这个环境时，先执行 `conda create -n DataEnv python=3.12`。编辑器提示缺少 `ipykernel` 时，在该环境里安装它。

交付的 Word2Vec 使用 NLTK 3.9.3，并核对名词词形还原所用资源的指纹。上面的下载步骤在本机运行一次；Spark Python worker 也要能读取这些资源。升级 NLTK 或替换资源后，需要重新训练对应的 Word2Vec，不能直接复用旧模型。

源文件是 `data/raw/drugsComTrain_raw.csv`，替换表是 `replacement.json`。源数据和拟合好的 embedding 随仓库交接。`data/processed/` 是本机缓存，不进 Git。分类器不另存模型文件。从仓库根目录启动 notebook。入口也会向上寻找 `src/` 和 `project.ipynb`，所以从本仓库的 `experiments/` 启动可以导入共享代码。

`project.ipynb` 使用 `local[4]` 和 `16g` driver。实验 notebook 在自己的第一个单元格里设置 master 和内存。SparkSession 由入口创建一次。改过 driver memory，或 JVM 退出并出现 `Connection refused` 之后，重启 kernel。改过已导入的 `src/` 函数后也要重启；只重新执行 import 会继续用旧模块。

## 各自改什么

成员实现分配给自己的一个模型文件。基类、包入口、公共预处理和主 notebook 的循环由集成人维护。

- embedding：`src/embeddings/<name>.py`，输出向量列 `embedding`。
- analytics：`src/analytics/<name>.py`。读取 `embedding` 和已编码的非文本列，在训练集上拟合并返回三份预测。文件给出 `DEFAULT_PARA` 和 `PARA_RANGE`。
- 集成人：公共流程、依赖、文档和最终汇总。共享数据或接口要改时，由集成人协调。

模型用装饰器注册。在已有实验 notebook 里选择自己的注册名。可选依赖放在使用它的函数里导入，并交给集成人写入 `requirements.txt`。

## 运行入口

- `project.ipynb`：EDA、预处理，然后训练 `train_registry` 里的 embedding 并评分 `registry` 里的分类器。TF-IDF 不在这个循环里。`BASELINE = True` 时，只做分词的 TF-IDF 存为 `tfidf`，只调用 `baseline()`，预测存为 `lr__tfidf`；带文本特征的 TF-IDF 存为 `tfidf_improved`，只调用 `improved_baseline()`，预测存为 `lr_improved__tfidf_improved`。需要短语的表示在构造 `EmbeddingModel` 前调用 `fit_phrases(df_train)`。
- `experiments/embedding_pipeline.ipynb`：选择一个 `train_registry` 里的 `embedding_name`，再跑 `registry` 里的分类器。
- `experiments/analytics_pipeline.ipynb`：选择一个 `register` 注册名。`REBUILD = True` 时训练已注册 embedding；`False` 时按注册名读取已有向量。
- `experiments/para_tuning/para_tuning_pipeline.ipynb`：按写下的 `EMBEDDING_MODEL` 和 `ANALYTICS_MODEL` 做 K 折调参。折向量写到 `data/processed/temp/`，不覆盖公共 split。

主入口和个人入口只评这次运行的预测。`load_prediction_data(model_names)` 按本轮模型名读取；不传名字时仍可读取全部历史预测。最终比较看 test。

## 清洗和文本的边界

`data_cleaning()` 做行过滤、已配置的替换、成对外层引号、空白、小写、确定性去重，并生成 `days`、`usefulCount_log`、`rating_p1`。`text_feature_engineering()` 把空白、缩写和剂量修复写回 `review`，再在临时列上分词；停用词步骤保留否定词和程度词。主 notebook 打开 bigrams。`baseline_text_feature_engineering()` 只对 `review` 做 Tokenizer，给上面的 `tfidf` 路径用。`fit_phrases(train_df)` 只在训练集 `words` 上拟合短语，不写回公共文本列。具体规则见 [docs/interface.md](docs/interface.md)。

文本修复用通用规则，不建数据集专属错拼表。修复发生在去重之后，比较模型前单独检查因此产生的文本碰撞。

## 缓存

`REBUILD` 只决定是否重建这个入口需要的输入，不决定模型是否训练。

- 主 notebook 和 embedding 实验：`True` 从 CSV 做到编码后的 train、valid、test；`False` 读取已有 split。默认比例是 `(7, 1, 2)`。`weight` 只加在拟合用的那一份上。
- analytics 实验：`True` 还会训练每一个已注册 embedding；`False` 按注册名读取向量。
- 调参实验：`True` 在默认训练池上做 K 折，折向量写到 `data/processed/temp/{EMBEDDING_MODEL}_para/fold_<i>/`；`False` 用 `KFold.open` 读取。
- 首次拉取没有 `data/processed/`，先重建。之后只改自己的模型时，可以复用匹配的输入缓存。
- 改了清洗、文本特征、`replacement.json`、seed 或公共编码之后，重建 split 和全部下游结果。只改一个 embedding 时，更新它的向量和依赖它的分类器。只改分类器时，更新该模型的预测。
- 主 notebook 复用 split 时，可以从划分单元格继续。直接 Run All 仍会执行原始数据和 EDA。

现有 runner 默认会训练 embedding。只有表示与当前数据、编码和参数匹配时，才可以在个人实验里跳过对应的 `train` 调用。把 `REBUILD` 改成 `False` 不会跳过训练。

## 目录

```text
AGENTS.md                         AI 的修改边界
project.ipynb                     公共集成入口
src/cleaning.py                   数据清洗
src/text_features.py              文本特征与短语
src/split.py                      哈希划分与训练权重
src/encoder.py                    condition / drug 编码
src/embeddings/<name>.py          各 embedding
src/analytics/<name>.py           各分类器
src/evaluation.py                 评分
src/dataOI.py                     读取、保存和缓存释放
src/paths.py                      路径常量
replacement.json                  condition 全名与 HTML 实体
experiments/                      单模型入口
final/final_submission.ipynb      最终汇总入口
src/embeddings/models/            拟合好的表示
data/processed/                   本地 split、向量和预测
```

提交时带上自己的模型实现和 embedding 拟合产物，以及集成人确认过的依赖改动。分类器不提交模型、阈值或搜索文件。`data/processed/` 不提交。提交 notebook 前清除输出。
