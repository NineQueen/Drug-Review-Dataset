# DSAI4205 Big Data Analytics — UCI Drug Review

用 PySpark 将药品评论的 `rating` 分为 1–10 共十个类别。共享实现位于 `src/`，`project.ipynb` 负责集成运行。当前 baseline 为 TF-IDF + 不加权 Logistic Regression，另有加权 LR 对照；高级模型与最终报告在下一阶段完成。

## 先看哪份说明

- 人和 AI 的文件职责边界：[AGENTS.md](AGENTS.md)。AI 开始任务前先读这一份。
- 模型接入、列类型、注册和输出约定：[docs/interface.md](docs/interface.md)。
- 预处理负责人的剩余任务：[docs/todo.md](docs/todo.md)。模型负责人不必顺手完成此清单。

## 环境与数据

已验证的运行组合为 Python 3.12、PySpark 4.2.0、Java 21。使用可用的 Java 21，并将 `JAVA_HOME` 指向其安装目录。`DataEnv` 是本机环境名称，组员可以用其他名称；Jupyter 必须选择安装了依赖的同一 Python 环境。

```bash
conda activate DataEnv
python -m pip install -r requirements.txt
```

没有此环境时，可先创建 `conda create -n DataEnv python=3.12`。Notebook 需要可用的 Python kernel；若编辑器提示缺少 `ipykernel`，在该环境安装它。

源文件为 `data/raw/drugsComTrain_raw.csv`，配置为 `replacement.json`。源数据和拟合模型计划随仓库交接，目录迁移提交时应一起包含；`data/processed/` 是本机生成的缓存，不随 Git 拉取。先确认 CSV 文件存在，再进行首跑。

推荐从仓库根目录启动 notebook。现有入口也会从当前目录向上寻找 `src/` 和 `project.ipynb`，因此从本仓库的 `experiments/` 启动同样可以导入共享代码；从仓库外启动时应把 kernel 工作目录改到仓库内。

## 组员如何独立工作

**原则上，成员只需实现分配给自己的一个模型文件。** 不需要改基类、包入口、公共预处理或主 notebook 的循环。

- embedding 负责人：修改 `src/embeddings/<自己的模型>.py`，生成统一的向量列 `embedding`。
- analytics 负责人：修改 `src/analytics/<自己的模型>.py`，使用现成向量训练分类器并返回预测。
- 集成人：维护公共流程、依赖、文档、基准结果与最终汇总。共享数据或接口需要变更时，由集成人协调。

模型通过装饰器自动注册。运行时在已有实验 notebook 里选择自己的注册名即可；选择参数属于运行配置，不需要改公共调用逻辑。新模型的可选依赖放在其函数内部导入，并把依赖交给集成人登记，避免影响其他成员导入自己的模型。

## 运行入口

- `project.ipynb`：完整集成，先 EDA 和预处理，再训练所有已注册 embedding、生成向量，再训练所有 analytics 组合并评分。
- `experiments/embedding_pipeline.ipynb`：选择一个 `embedding_name`，运行该表示及已有 analytics 组合。
- `experiments/analytics_pipeline.ipynb`：选择一个 `analytics_name`，运行该分类器与本机已准备的 embedding 组合。
- `experiments/embedding_baseline.ipynb`：默认选择 `tfidf`，用于检查 baseline 表示的流程。

个人实验只评分本次运行产生的预测，不混入其他历史缓存。主 notebook 仍按已有公共循环做全量集成。TF-IDF + LR 是指定 baseline；自动跑出的其他 TF-IDF 组合不计入高级模型要求。

## 首跑与缓存运行

所有入口使用 `REBUILD`，含义是**是否重建本入口需要的输入数据**，不控制后面的模型训练。

- 主 notebook / embedding 实验中，`True` 从 CSV 完成清洗、文本特征、哈希划分、类别权重和编码，并保存 split；`False` 读取已有 split。
- analytics 实验中，`True` 还会训练现有 embedding 并保存其向量表；`False` 读取已有 embedding 表。
- 首次拉取没有 `data/processed/`，需先重建。后续只改自己的模型时，可复用匹配的输入缓存。
- 主 notebook 复用 split 时，运行初始化单元格后，可从划分/编码单元格继续；原始数据和 EDA 单元格可跳过。直接 Run All 仍会执行原始数据与 EDA 检查。
- 修改清洗、文本特征、`replacement.json`、哈希 seed 或公共编码后，需要重建 split 及全部下游结果。只改 embedding 时，需要更新该 embedding 及依赖它的分类器；只改分类器时，更新该模型和预测即可。
- 本地半成品或不再使用的缓存目录可以清理后重跑；加载器目前会检查对应目录的 train/test 是否齐全。请勿把不同输入版本的预测混在同一轮比较中。

拟合模型可以随代码交接，但现有 runner 默认仍会执行训练。仅当模型与当前数据、编码和参数匹配时，才可在个人实验中保留上下文初始化并跳过相应训练调用，直接执行预测；仅把 `REBUILD` 改成 `False` 不会跳过训练。

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
src/analytics/models/                拟合好的分类器
data/processed/                     本地生成的 split / embedding / prediction
```

## 常见运行问题与 Git

SparkSession 由入口创建一次，模型文件不要另建会话。当前 driver memory 为 `4g`；修改内存后或出现 JVM 退出导致的 `Connection refused`，重启 kernel 后从初始化开始。新增或修改已注册模型后也推荐重启 kernel，避免旧类与注册表残留。

提交时只包含自己的模型实现、该模型的拟合产物和集成人已确认的依赖改动；共享入口由集成人维护。生成的大表 `data/processed/` 不提交。提交 notebook 前清除输出；notebook 冲突标记可能藏在 JSON 的输出字段里。当前不要求各成员增加复杂的缓存版本或容错框架。
