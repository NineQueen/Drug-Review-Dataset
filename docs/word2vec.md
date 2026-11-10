# Word2Vec embedding 交接

注册名为 `word2vec`，兼容当前 `EmbeddingModel(train, valid, test, phrases=None)` 和现有共享循环。只读取完整、未词干化的 `words`，只在 `self.train` 拟合，不使用 WebMD、短语或停用词过滤。输入的公共文本列不会被修改。

## 默认方案

| 配置 | 默认值 |
|---|---|
| vector_size | 300 |
| window_size | 5 |
| min_count | 5 |
| max_iter | 10 |
| step_size | 0.025 |
| seed | 42 |
| num_partitions | 1 |
| max_sentence_length | 1000 |
| pooling | meanmax |

词模型300维；评论向量为平均和逐坐标最大值各300维的拼接，共600维。均值分母是全部输入token数，包括OOV；OOV贡献0，不参与最大值，避免把负的最大值截成0。重复词重复计入。空/全部OOV输入输出定长零向量，不额外归一化。超过序列上限的评论由Spark在训练时分块；预测聚合整条评论。

## 重新训练默认配置

在现有 `experiments/embedding_pipeline.ipynb` 选择 `embedding_name = "word2vec"`，不改公共循环即可运行。模型模块不创建SparkSession，入口按README创建会话。

```python
from src.embeddings import EmbeddingModel

model = EmbeddingModel(df_train, df_valid, df_test)
model.word2vec_train()  # 真正重新拟合，覆盖自己的word2vec模型目录
train_embedding, valid_embedding, test_embedding = model.word2vec_predict()
```

输出由基类裁剪为analytics约定列，并保留ID、标签和样本；`embedding` 是Spark ML `VectorUDT`。不依赖预测输出中仍有 `words` 或 `review`。

`REBUILD=False`只控制输入缓存，不会跳过训练。若要直接使用交付模型，在个人入口保留模型实例与Spark初始化，跳过 `word2vec_train()`，直接执行 `word2vec_predict()`。共享主入口默认仍先训练。

## 自定义配置

训练函数支持 `config` 字典，只覆盖提供的项目；原默认字典和传入字典都不被修改。

```python
model.word2vec_train(
    model_name="word2vec_custom",
    config={"vector_size": 200, "window_size": 10,
            "max_iter": 20, "pooling": "meanmax"},
)
custom_train, custom_valid, custom_test = model.word2vec_predict("word2vec_custom")
# 每条向量400维；其他参数沿用默认值。
```

可覆盖的键就是上表九项。`pooling`支持 `p01`（d维原生平均）与 `meanmax`（2d维）；拼错键、错误类型、非正训练长度/轮数、非法学习率或聚合方式明确报错。`min_count`允许非负整数，seed为64位整数。新模型名使用小写字母、数字和单下划线，不含路径分隔符或 `__`，实验请使用独立名字。

实际完整配置写入模型目录的 `embedding_config.json`。预测只接收 `model_name`，自动读取该模型的保存配置，不需要新版预测包装器传递额外kwargs；以后修改代码默认值也不会改变已有模型的聚合方式。该文件必须随模型交付，没有静默回退到当前默认值的分支。

## 拟合产物

`src/embeddings/models/word2vec/` 包含原生Spark模型、`embedding_config.json` 和 `provenance.json`。这次交付的原生Spark文件按字节复制自 `w2v_t20261108_225759_d300`，不是重新训练得到的替代品。

历史训练：提交 `46b1a03` 对应原训练集77,800条、6,702,823个token，词表14,785。原模型Spark4.2.0，Python3.12.9/Java17环境；最新README要求的Java21兼容检查另见验证记录。输入Parquet指纹和原生模型文件SHA256见 `provenance.json`。

历史验证集由既有不加权 `lr_baseline` 联合文本与元数据评估：accuracy **40.7284%**，weighted F1 **0.355288**，MAE **1.739302**，RMSE **2.938868**。这是旧提交与固定输入下的记录，不是新共享管线重建后的评分，不能保证重训逐位一致或稳定超过其他模型。没有在本交付中加入analytics代码、分类器文件或预测表。

重新训练会更新原生模型和实际配置，旧模型的 `provenance.json` 不会被沿用。若共享清洗、输入或参数改变，需要更新对应向量和下游结果。公共新增停用词保护不直接改变本模型读取的 `words`，但输入版本仍需核对。

模型目录内 `.gitattributes` 对该目录设置 `-text`，保留Spark元数据、Parquet及Hadoop CRC的原始字节。提交其他新模型时应保留训练函数生成的同类属性文件，不手动删减模型内部文件。

## 检查

按根README准备Python3.12/PySpark4.2/Java21及依赖，创建可用环境后，从仓库根目录运行：

```bash
python experiments/word2vec_delivery_check.py
```

该脚本只训练临时小样本模型，检查注册、默认600维、自定义p01/meanmax维度、配置持久化、重复训练、仅train拟合词表、空/OOV、负坐标最大值、窄输出与Parquet保存加载，不改交付模型。结果和临时文件位于 `tmp/`，不提交。可用 `--reference-repo <旧实验仓库>` 额外比对历史600维向量及新版分词样本。

交付验证的实际结果记录于本目录 `word2vec_validation.json`。它只验证embedding和接口，不重新训练analytics；依赖全部来自现有requirements.txt，无新增第三方依赖。
