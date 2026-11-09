# DSAI4205 Big Data Analytics — UCI Drug Review

用 PySpark 将药品评论的 `rating` 分为 1–10 共十个类别。共享实现位于 `src/`，`project.ipynb` 负责集成运行。当前 baseline 为 TF-IDF + 不加权 Logistic Regression，另有加权 LR 对照；高级模型与最终报告在下一阶段完成。

## 先看哪份说明

- 人和 AI 的文件职责边界：[AGENTS.md](AGENTS.md)。AI 开始任务前先读这一份。
- 模型接入、列类型、注册和输出约定：[docs/interface.md](docs/interface.md)。

## 环境与数据

已验证的运行组合为 Python 3.12、PySpark 4.2.0、Java 21。使用可用的 Java 21，并将 `JAVA_HOME` 指向其安装目录。`DataEnv` 是本机环境名称，组员可以用其他名称；Jupyter 必须选择安装了依赖的同一 Python 环境。

```bash
conda activate DataEnv
python -m pip install -r requirements.txt
```

没有此环境时，可先创建 `conda create -n DataEnv python=3.12`。Notebook 需要可用的 Python kernel；若编辑器提示缺少 `ipykernel`，在该环境安装它。

源文件为 `data/raw/drugsComTrain_raw.csv`，配置为 `replacement.json`。源数据和拟合好的 embedding 计划随仓库交接，目录迁移提交时应一起包含；`data/processed/` 是本机生成的缓存，不随 Git 拉取。分类器不要求另存模型文件。先确认 CSV 文件存在，再进行首跑。

推荐从仓库根目录启动 notebook。现有入口也会从当前目录向上寻找 `src/` 和 `project.ipynb`，因此从本仓库的 `experiments/` 启动同样可以导入共享代码；从仓库外启动时应把 kernel 工作目录改到仓库内。

## 组员如何独立工作

**原则上，成员只需实现分配给自己的一个模型文件。** 不需要改基类、包入口、公共预处理或主 notebook 的循环。

- embedding 负责人：修改 `src/embeddings/<自己的模型>.py`，生成统一的向量列 `embedding`。
- analytics 负责人：修改 `src/analytics/<自己的模型>.py`。注册函数只读取 `embedding` 和已经编码好的非文本列，在训练集上拟合并返回预测。文件开头给出 `DEFAULT_PARA` 和 `PARA_RANGE`；函数接收可缺省的 `para`，缺省时使用 `DEFAULT_PARA`。具体格式见 [docs/interface.md](docs/interface.md)。
- 集成人：维护公共流程、依赖、文档、基准结果与最终汇总。共享数据或接口需要变更时，由集成人协调。

模型通过装饰器自动注册。运行时在已有实验 notebook 里选择自己的注册名即可；选择参数属于运行配置，不需要改公共调用逻辑。新模型的可选依赖放在其函数内部导入，并把依赖交给集成人登记，避免影响其他成员导入自己的模型。

## 运行入口

- `project.ipynb`：完整集成，先 EDA 和预处理，再训练所有已注册 embedding、生成向量，再对每个 analytics 注册函数调用一次并评分。不加权 baseline 单独调用 `baseline()`。
- `experiments/embedding_pipeline.ipynb`：选择一个 `embedding_name`，运行该表示及 `registry` 中的 analytics；设为 `tfidf` 即可检查 baseline 表示的流程。
- `experiments/analytics_pipeline.ipynb`：选择一个 `analytics_name`，这个名字是 `register` 的注册名，运行该分类器与本机已准备的 embedding 组合。

个人实验分别评分本次运行的 train、validation、test 预测，不混入其他历史缓存；最终比较看 test。主 notebook 仍按已有公共循环做全量集成。TF-IDF + 不加权 LR 是指定 baseline；自动跑出的其他 TF-IDF 组合不计入高级模型要求。

embedding 和 analytics 都接收 train、validation、test。会改变预测的量只从 train 得到。validation 和 test 在分类器函数内部都不参与拟合或改写预测；函数返回后可以查看验证分数，最终比较使用 test。embedding 的 `train` 保存模型，`predict` 加载该模型并按这个顺序返回三份向量。analytics 在同一个注册函数里拟合并返回三份预测，不保存分类器文件。共享 runner 调用 `registry[key](instance)`，因此使用默认 `para`。

## 数据清洗与文本特征的边界

`data_cleaning()` 保留整行过滤、已配置的 condition / HTML 替换、成对的外层包装引号移除、普通空白合并、小写化和确定性 review 去重，并生成 `days`、`usefulCount_log`、`rating_p1`。包装引号只在两端同时存在时去掉一对：先是解析后的 `"`，再是两端都有的 `&quot;`。只有一侧有引号时保留。普通空白在识别包装前、以及 HTML 解码后再各合并一次。

`text_feature_engineering()` 把字符与 Unicode 空白规范化、缩写展开和剂量空格写回 `review`，再复制临时列 `review_clean` 做分词。停用词步骤保留否定词，词干写入 `words_featured`；主 notebook 调用时打开 bigrams。token 处理不覆盖 `review`。函数返回既有列，不保留 `review_clean`，不删行、不改 ID 或评分，也不二次去重。`baseline_text_feature_engineering()` 只把 `review` 交给 Tokenizer，供单独的 baseline TF-IDF 路径使用。

后续若再增加通用文本修复，仍应先写回 `review`，再建立临时 token 列。只处理普遍模式，不建立数据集专属错拼表，也不逐条手工改正全部评论。这些修复发生在清洗去重之后，可能使不同记录得到相同文本或 token；比较模型前应单独统计这种碰撞，不自动合并记录或改动划分。

## 首跑与缓存运行

所有入口使用 `REBUILD`，含义是**是否重建本入口需要的输入数据**，不控制后面的模型训练。

- 主 notebook / embedding 实验中，`True` 从 CSV 完成清洗、文本特征、约 70/10/20 的哈希划分、类别权重和编码，并保存 train、valid、test；`False` 读取已有 split。
- analytics 实验中，`True` 还会训练现有 embedding 并保存其向量表；`False` 读取已有 embedding 表。
- 首次拉取没有 `data/processed/`，需先重建。后续只改自己的模型时，可复用匹配的输入缓存。
- 主 notebook 复用 split 时，运行初始化单元格后，可从划分/编码单元格继续；原始数据和 EDA 单元格可跳过。直接 Run All 仍会执行原始数据与 EDA 检查。
- 修改清洗、文本特征、`replacement.json`、哈希 seed 或公共编码后，需要重建 split 及全部下游结果。只改 embedding 时，需要更新该 embedding 及依赖它的分类器；只改分类器时，更新该模型和预测即可。
- 修改已经被 kernel 导入的 `src/` 函数后，先重启 kernel，再设置 `REBUILD = True` 重建；仅重新执行 `from ... import ...` 会复用已加载的模块，可能继续调用旧实现。
- 本地半成品或不再使用的缓存目录可以清理后重跑；加载器会检查对应目录的 train、valid、test 是否齐全。请勿把不同输入版本的预测混在同一轮比较中。

拟合好的 embedding 可以随代码交接，但现有 runner 默认仍会执行训练。仅当该表示与当前数据、编码和参数匹配时，才可在个人实验中保留上下文初始化并跳过相应训练调用，直接执行预测；仅把 `REBUILD` 改成 `False` 不会跳过训练。分类器每次由注册函数拟合，并只留下 runner 写入的预测表。

## 目录

```text
AGENTS.md                            AI 工作边界和约定
project.ipynb                        公共集成入口
src/cleaning.py                      数据清洗
src/text_features.py                 文本特征
src/split.py                         按 ID 哈希划分及训练集权重
src/encoder.py                       condition / drug 编码
src/embeddings/<name>.py             各 embedding 的独立实现
src/analytics/<name>.py              各分类器的独立实现
src/evaluation.py                    统一评分
src/dataOI.py                       数据读取与保存
src/paths.py                        路径常量
replacement.json                     condition 全名映射与 HTML 实体
experiments/                         单模型运行入口
final/final_submission.ipynb         最终汇总位置，目前为空
src/embeddings/models/               拟合好的表示模型
data/processed/                     本地生成的 split / embedding / prediction
```

## 常见运行问题与 Git

SparkSession 由入口创建一次，模型文件不要另建会话。当前 driver memory 为 `4g`；修改内存后或出现 JVM 退出导致的 `Connection refused`，重启 kernel 后从初始化开始。新增或修改已注册模型后也推荐重启 kernel，避免旧类与注册表残留。

提交时包含自己的模型实现、embedding 的拟合产物，以及集成人已确认的依赖改动；共享入口由集成人维护。分类器不提交额外的模型、阈值或搜索文件。生成的大表 `data/processed/` 不提交。提交 notebook 前清除输出；notebook 冲突标记可能藏在 JSON 的输出字段里。注册函数假定 runner 给出的输入合法，不为缺列或可选文件增加分支。
