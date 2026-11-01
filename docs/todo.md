# 文本特征工程待办与清洗边界

本清单由公共预处理负责人维护，模型负责人不需要顺手完成。AI 的改动边界见 [AGENTS.md](../AGENTS.md)，模型接口见 [interface.md](interface.md)。

当前边界：data cleaning 恢复原 TODO 1（旧 A2）引入前的处理方式；原 TODO 1 起的额外字符、空白及后续文本工作统一放在 text feature engineering。B1–B8 的实现、独立检查、实际 Spark 文本流程验收及规范化碰撞审计已完成；B10 的可选 bigram 实现与仓库外实际 Spark 样例、全量检查也已完成，检查记录见 B10。尚未执行 B9 的输入缓存重建与下游重新训练，也未比较分类效果。

本轮用户任务为完成文本特征工程，包含新增的 B10 可选词组实现及正确性检查。简单 baseline / improved baseline 的分支设计、结果版本管理及训练不属于本轮范围。B9 留作后续集成运行提醒，不要求当前文本特征负责人修改 baseline 或执行训练；B10 的分类收益仍需后续验证。

后续只修复较普遍、可用通用规则描述的问题。不维护数据集专属错拼表，不逐条编写评论例外，不要求人工穷举全部异常形式。有歧义时保留原表达，使用全量基础统计与代表性样例验证支持的规则。

## A. Data cleaning 保持原有流程

- [x] 整行删除缺失 `condition` 或该字段混入网页提示的记录。
- [x] 保留 `replacement.json` 中现有 condition 名称映射与 HTML 实体替换。
- [x] 恢复原有外层双引号移除、首尾空格处理、普通空白合并及小写化。
- [x] 按此阶段的 `review` 去重：优先保留最大 `usefulCount`，相同时保留最大 `uniqueID`。
- [x] 保留 `usefulCount_log`、`days`、`rating_p1` 的生成及既有返回接口。

本轮不扩展 condition 或 drugName 的修复范围。额外字符规范化不再影响此阶段的去重键；旧版清洗后的行数和字符检查快照不能直接作为新版验收结果。

## B. Text feature engineering

本阶段将所有已实现的通用文本修复先写回 `review`，使所有读取完整评论的模型共享修复结果；之后才建立临时 `review_clean`，准备 token 并生成 `words`、`words_stem`。当前已接入字符与空白规范化、明确缩写展开及常见剂量空格规范化，并改用显式模式的 `RegexTokenizer` 提取完整 `words`；临时列上的旧剂量合并及非字母数字替换块已移除。自定义 `StopWordsRemover` 保留否定词，在临时 `words_nostop` 上过滤普通停用词，再生成 `words_stem`；词干函数跳过空 token，保留含数字 token。通用文本修复不以现有 `normalize_review_text()` 为边界。token 处理不覆盖修复后的完整评论；保留行、`uniqueID` 和原始评分，不二次去重。对外继续返回既有列，不永久保留 `review_clean` 或 `words_nostop`，原始 CSV 文件保持原样。

- [x] **B1. 迁移现有字符与空白规范化（原 TODO 1／旧 A2）**
  - `normalize_review_text()` 已从 `src/cleaning.py` 搬到 `src/text_features.py`，在剂量处理和分词前调用。
  - Before/After character repair 统计一并迁移，检查对象为直接更新的 `review`。
  - 本项只表示既有实现已搬运；迁移未新增字符规则，也不表示后续规则复核或全量验收已完成。

- [x] **B2. 复核通用字符规范化规则**
  - 梳理引号、撇号、Unicode 空白及格式控制字符的通用处理，保护正常字母、重音、数值、单位与有意义符号。
  - 复核现有规则的必要性及误改风险；后续调整采用普适模式，不为个别字形或单条评论增加补丁。
  - 记录每类规则的影响数量；规则残留为零只说明该规则的处理结果。
  - 2026-11-01 完成独立全量盘点与 Java 正则对照；撤去特定字形替换，保留有歧义字符，并采用通用控制字符处理。初次独立检查未执行 Spark；后续实际 Spark 文本流程验收结果见 B8。
  - 独立复现清洗后共 111,169 条评论；新版字符规则改变 1,117 条，空评论与二次执行变化均为 0；244 个 Java 正则对照案例及 7 组保留性样例通过。按规则执行顺序的修改评论数：撇号 696、双引号 64、零宽空格 20、LRM 2、非空白控制字符 2、Unicode 空白合并 19；各项可重叠，未含最后 trim 单独造成的修改。

- [x] **B3. 常见缩写及其空格处理（原 TODO 2）**
  - 将这部分通用修复结果写回 `review`，放在临时 token 列创建之前。
  - 在删除标点或分词前处理常见、含义明确的英文缩写，如 `didn't → did not`、`can't → can not`。
  - 对常见合法缩写形式中的多余空格采用有边界的规则，不无条件删除所有撇号两侧空格。
  - 对有歧义的后缀保留原表达，保护所有格与医学名称；不扩展为错拼纠正或逐条上下文判定。
  - `expand_review_contractions()` 在字符规范化之后、临时 token 列创建之前更新 `review`。先整理支持形式的空格，再展开明确的否定缩写和 `'m` / `'re` / `'ve` / `'ll` / `let's`；当前采用 `rf"\b{pattern}\b"` 的简洁单词边界，不再使用原有自定义左右边界常量。避免将 `tablet's` 的结尾当成 `let's`，或将 `i 'might'` 的开头当成 `i 'm`；不额外保护罕见连写复合缩写。
  - 保留 `'s`、`'d`、`ain't`、所有格、普通引号及 `dont` / `cant` 等无撇号形式，不按评论增加例外。
  - 2026-11-01 当前版本的 11 组独立 Python 正则样例检查通过，覆盖缩写空格、否定展开、所有格/医学名称/歧义后缀保留及普通引号表达。此前自定义边界版本的全量数量和 39 组样例结果不作为当前版本验收依据；后续实际 Spark 验收确认当前缩写规则改变 63,216 条评论，详见 B8。

- [x] **B4. 剂量表达规范化（原 TODO 3）**
  - 在 `review` 上完成保持原数值含义的通用剂量格式规范化，再准备 token；调整现有临时列上的剂量合并步骤，避免重复处理。
  - 配合分词规则识别常见整数、小数与单位，使 `0.5 mg` 与 `0.5mg` 表示一致，且区别于 `5mg`。
  - 保留原数值与单位，不猜测或纠正剂量。
  - 已实现 `normalize_dose_spacing()`：规范常见整数/小数与 `mg` / `mcg` / `ml` 之间的空格，在缩写展开之后、创建 `review_clean` 之前写回 `review`；临时列上的旧剂量合并已移除。不换算单位，不改变数字。
  - 2026-11-01 当前函数的 12 组独立正则样例及重复执行检查通过，覆盖整数、小数、三种单位、已合并形式、多空格、多剂量、范围外单位保留和 null。实际 Spark 执行结果见 B8；缓存重建仍待 B9。
  - `review` 中的 `0.5mg` 已保留；B5 已移除破坏小数点的旧标点规则，独立样例及 B8 实际 Spark 样例均确认 `0.5mg` 与 `5mg` 可作为不同的完整 token 保留。

- [x] **B5. 调整分词与标点处理（原 TODO 4）**
  - 替换直接删除所有非字母数字字符的做法，配置有效 token 规则。
  - 配合缩写和剂量处理，保护否定表达、小数及需要保留的撇号；不只更换分词工具名称而沿用默认行为。
  - 已保留临时文本副本，移除旧的非字母数字替换块；使用 `RegexTokenizer(gaps=False, minTokenLength=1, toLowercase=False)`，依次匹配完整剂量、普通小数、含必要词内撇号的字母数字词，并保留正常 Unicode 字母与组合标记。未匹配的标点不进入 `words`，不需要先破坏性替换标点。
  - 2026-11-01 从当前源码读取分词配置和模式，结合 B2–B4 辅助函数完成 14 组独立 Python `regex` 样例检查，覆盖 `0.5mg` / `5mg`、`.5mg`、三种剂量单位、否定句、词内撇号、重音/组合标记、字母数字词、单独标点、空串及完整缩写/剂量处理链；结果符合预期。源码语法与差异空白检查通过。后续实际 Spark DataFrame 验收见 B8。
  - 独立样例已检查 `words` 中的 `0.5mg` / `5mg` 区别及完整性；B6/B7 已接入自定义停用词过滤和数字 token 保护，实际 Spark 验收已完成。

- [x] **B6. 调整停用词处理（原 TODO 5）**
  - 已按用户确认的方案，由“全部保留停用词”调整为“删除普通停用词、保留否定词”。从默认英文停用词表中排除 `not`、`no`、`nor`、`never`、`neither`、`without`，再传给 `StopWordsRemover(stopWords=custom_stop_words)`。
  - Pipeline 顺序为 `RegexTokenizer → StopWordsRemover`。完整 `words` 保留，过滤结果写入临时 `words_nostop`，`stem_words()` 读取该临时列。最终返回列仍为原有接口。
  - 2026-11-01 当前自定义过滤及词干逻辑的 11 组独立 Python 样例通过，覆盖普通停用词删除、否定词保留、小数剂量、含数字词、仅停用词/仅标点/空文本及词干函数的 null/空 token 处理。后续实际 Spark 验收已完成，见 B8。
  - 尚未比较该方案与“全部保留”的分类效果；如需选择或调整词表，使用训练部分的验证数据，不用测试集调参。

- [x] **B7. 核对词干与中间结果（原 TODO 6）**
  - 保留未词干化的 `words`，继续为 TF-IDF 提供 `words_stem`。
  - 保留 PorterStemmer，过滤空 token，检查数字及剂量 token 是否需要跳过词干化。
  - 统计空结果并明确处理方式，保持既有列类型与样本覆盖。
  - 已在 `stem_words()` 中跳过空 token，让含数字的 token 原样保留，其余词使用 PorterStemmer，不按各剂量或单位逐条设例外。null 输入返回空数组；空数组自然返回空数组。
  - 2026-11-01 从当前源码读取 B2–B7 规则，以独立 CSV 清洗复现、Python 正则与实际 NLTK 词干函数检查 111,169 条评论：空 `review` 0 条，空 `words` 1 条，空 `words_nostop` / `words_stem` 各 3 条，词干数组内部空 token 0 个。三条空词干评论为 `a+++`（158432）、`i`（172398）、`-`（180087）。未删行、未填入占位词；当前 TF-IDF 可将空数组表示为零文本向量。
  - 空结果统计是开发阶段的独立检查，不要求把聚合打印加入正式函数或 `project.ipynb`；未新增检查文件。上述独立结果与 B8 实际 Spark 结果一致；运行时列类型、ID/标签集合和全量行数均已通过 B8 核对。

- [x] **B8. 验证流程及规范化碰撞（原 TODO 7）**
  - 以下 2026-11-01 验收结果对应默认 unigram 流程；B10 启用 bigram 的补充检查另记在 B10。
  - 用代表性样例检查通用字符、缩写、否定句、小数剂量、引号和空结果。
  - 检查 `not effective` 与 `effective`、`0.5mg` 与 `5mg` 能否区分。
  - 汇总空文本、空 token、列类型、ID 和标签等基础统计；确认本阶段保持行、ID 和标签，`review` 包含全部已实现的通用修复并保留完整评论，不被 token 准备覆盖。
  - 单独统计不同 `review` 变成相同规范文本或 token 的情况，并检查其跨 split 分布。此项是审计，不自动删行、合并或改变划分；token／词干相同不直接定义为重复评论。
  - 2026-11-01 后台直接调用现有 `DataOI.load_raw_data()`、`data_cleaning()`、`text_feature_engineering()`，未修改源码或 notebook，未保存模型或 processed 缓存，未新增项目检查文件。
  - 实际运行环境：Python 3.11.18、PySpark 4.2.0、Java 17.0.18，`local[*]` / driver 4g；检查进程使用 default parallelism 2、shuffle partitions 8 控制资源，不改变源码中的 ID 哈希划分策略。修正后台启动方式的标准 JVM 参数和进程依赖路径后完成运行。
  - 实际 Spark 六条样例验证否定词、小数剂量、含数字 token 及空数组处理。另用小样本 CountVectorizer（仅此检查设 minDF=1）+ IDF 验证空数组可产生零文本向量，未保存拟合模型，未训练分类器。
  - 全量实际 Spark：原始 161,297 行，清洗后及文本处理后均为 111,169 行，唯一 ID 为 111,169；处理前后 ID、药物、condition、days、usefulCount_log、rating、rating_p1 的多重集合完全相同，缺失/新增记录均为 0。输出列保持原接口，`words` / `words_stem` 均为 `array<string>`，无永久临时列。
  - 空 `review` 0 条，空 `words` 1 条，空 `words_stem` 3 条，词干数组内部含空元素的行数 0，评分与 rating_p1 对应错误 0。字符规则改变 1,117 条评论，重复执行变化 0；缩写规则改变 63,216 条评论。
  - 用现有 `xxhash64(uniqueID, SEED)` 策略和 notebook 的 SEED=42 审计：train 88,948 行，test 22,221 行。未改变 seed、ID 类型、划分或去重规则。
  - 按“不同清洗后原文得到相同结果”统计：完整 `review` 碰撞 7 组/14 行，其中跨 split 3 组、评分不一致 2 组；`words` 碰撞 75 组/174 行，其中跨 split 29 组、评分不一致 37 组；`words_stem` 碰撞 115 组/435 行，其中跨 split 59 组、评分不一致 65 组。三层结果有重叠，不可相加；这些碰撞不直接定义为重复记录，已保留所有样本。
  - Arrow 关闭时完成全量检查与碰撞审计，随后在相同环境中开启 Arrow，六条样例和全量样本完整性/类型/空结果检查再次通过。最终不需要为此关闭 Arrow 或修改词干函数；不要求向 notebook 添加输出模块。

- [ ] **B9. 后续集成人：重建输入与下游结果（原 TODO 8，非本轮范围）**
  - 本项尚未执行；不影响 B1–B8 及 B10 可选文本特征实现与检查的完成状态。后续由集成/训练负责人使用当前函数重建输入及模型，不在本轮引入简单 baseline / improved baseline 的对照分支。
  - 重启已导入旧函数的 kernel，使用 `REBUILD = True` 重跑清洗、文本特征、划分与编码，更新 `data/processed/split/{train,test}.parquet`。
  - 重新拟合 CountVectorizer、IDF、分类器并更新相应 embedding 与 prediction 缓存。现有入口不单独保存 cleaned／engineered Parquet。
  - `REBUILD` 只控制输入准备，不控制是否训练。词表、IDF 及其他学习型变换只在训练集拟合。
  - 保持现有 ID 哈希与 seed 策略；记录恢复清洗边界后的样本数和保留 ID 变化，所有模型使用同一版输入比较。

- [x] **B10. 实现可选 bigram 特征（原 TODO 9）**
  - `text_feature_engineering(df, use_bigrams=False)` 默认保持现有 unigram 行为；显式传入 `use_bigrams=True` 时，将 bigram 追加到 `words_stem`。当前 `project.ipynb` 未传入该参数，仍使用默认行为。
  - unigram 继续读取过滤普通停用词后的 `words_nostop`；bigram 从完整 `words` 先做相同词干处理，再用 `NGram(n=2)` 生成，保留删除停用词之前的相邻关系。例如 `not very effective` 生成 `not veri` / `veri effect`，不会人为生成 `not effect`。
  - `F.concat` 保留原有 unigram、词组顺序及重复出现的词组；少于两个 token 时不产生 bigram。最终返回列保持既有接口，不返回 `_ngram_words` 或 `_bigrams`；`review` 和 `words` 不因该开关改变。
  - 2026-11-02 在 repo 外复制当前源码和原始数据进行实际 Spark 检查，未修改源码或 notebook，未在 repo 内新增检查文件、模型或 processed 缓存。16 组样例在默认与启用两种模式下均通过，覆盖否定表达、原始相邻关系、缩写、小数剂量、Unicode、空输入、单 token、仅停用词及重复词组；列类型与其他字段保持一致。
  - 小样本使用固定词表的 CountVectorizerModel 验证含空格的词组作为单个特征计数，重复词组按出现次数计数，`0.5mg` 与 `5mg` 可区分；未拟合或保存分类器。
  - 全量启用 `use_bigrams=True`：清洗后与文本处理后均为 111,169 行、111,169 个唯一 ID；ID、药物、condition、days、usefulCount_log、rating、rating_p1 的多重集合一致，缺失/新增记录均为 0。共生成 9,462,248 个 bigram（按出现次数计，非不同词组数），每行数量均为 `max(size(words) - 1, 0)`，不符合预期的行数为 0；空 `words_stem` 仍为 3 行，词干数组内空元素为 0，输出列及数组类型符合原接口。
  - 检查环境为 Python 3.11.18、PySpark 4.2.0、Java 17.0.18，`local[*]` / driver 4g，Arrow 开启；检查进程使用 default parallelism 2 / shuffle partitions 8。源码快照 SHA-256 为 `5c8b4f5998a5fe72eb28c49ce01d6a55224ae3facf723e356f8b018d9213dd8f`，检查前后源码及 `project.ipynb` 校验值相同。
  - 尚未比较 unigram 与 unigram + bigram 的分类效果，不表示已证明提升。当前 TF-IDF 使用 `vocabSize=5000` / `minDF=5`，启用后单词与词组共用词表容量；后续比较需保持同版数据与划分，所有学习型变换仅在训练部分拟合，方案选择使用训练部分的验证数据，不使用测试集调参。

本轮文本特征工作顺序：B2 → B3 → B4／B5 → B6 → B7 → B8 → B10；实现及已记录的检查已完成。B9 为后续集成与训练事项，B10 的分类效果比较也留待后续；当前未执行模型重新训练。
