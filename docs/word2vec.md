# Word2Vec

注册名是 `word2vec`。它读取未词干化的 `words`，只在 `self.train` 上拟合，不使用 WebMD、短语或停用词过滤，也不改公共文本列。

## 默认配置

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

词向量 300 维。`meanmax` 把全部输入 token 的平均和已知词的逐坐标最大值拼成 600 维；均值分母包括 OOV，OOV 对均值贡献 0，不参与最大值。空评论和全部 OOV 的评论输出定长零向量。`p01` 是 `vector_size` 维平均。超过 `max_sentence_length` 的评论在训练时由 Spark 分块；预测聚合整条评论。

在 `experiments/embedding_pipeline.ipynb` 里设 `embedding_name = "word2vec"`。`REBUILD = False` 只跳过输入重建，不跳过训练。要直接用已交付模型时，保留会话和模型实例，跳过 `word2vec_train()`，调用 `word2vec_predict()`。

```python
model = EmbeddingModel(df_train, df_valid, df_test)
model.word2vec_train()
train_embedding, valid_embedding, test_embedding = model.word2vec_predict()
```

`config` 只覆盖给出的键，不修改默认字典。可覆盖的键就是上表九项。未知键、非法类型、非正的训练长度或轮数、非法学习率、非法 `pooling` 会报错。`min_count` 可以是 0。`seed` 是 64 位整数。实验用单独的 `model_name`。

```python
model.word2vec_train(
    model_name="word2vec_custom",
    config={"vector_size": 200, "window_size": 10, "max_iter": 20, "pooling": "meanmax"},
)
```

实际配置写入模型目录的 `embedding_config.json`。预测只接收 `model_name`，并读取这份保存配置。没有这个文件时不回退到当前代码默认值。

## 当前交付

`src/embeddings/models/word2vec/` 含 Spark 模型、`embedding_config.json` 和 `provenance.json`。目录里的 `.gitattributes` 对该目录设置 `-text`。原生 Spark 文件按字节复制自 `w2v_t20261108_225759_d300`。`provenance.json` 记录当时的输入指纹和文件 SHA256：训练集 77,800 条、6,702,823 个 token、词表 14,785，来自提交 `46b1a03`。那次验证用的是当时的不加权 logistic baseline，accuracy 40.7284%，weighted F1 0.355288，MAE 1.739302，RMSE 2.938868。这是旧输入上的记录。重新训练会更新模型和 `embedding_config.json`，不沿用旧的 `provenance.json`。

交付前的接口和小样本检查记在 `docs/word2vec_validation.json`。自检脚本不在仓库里。重新训练用 `experiments/embedding_pipeline.ipynb`。这份记录不是当前共享管线的分类器评分。
