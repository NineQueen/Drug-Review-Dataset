# 进度

当前代码里的表示是 word2vec、BERT、GloVe，另有单独的 TF-IDF baseline。分类器是不加权 LR、加权 LR、SVM 和 random forest。MLP 和 stacking 仍是后面的工作，代码里还没有。

## 分工

每人只改自己的模型，并在 `experiments/` 里用自己的入口。embedding 的 train 和 predict 都要注册：train 拟合并保存模型，predict 从该目录变换三份数据。analytics 用一个函数拟合并返回预测，不另存分类器。

- BERT：fyc
- Word2Vec：lyt
- GloVe：zhx
- 线性 SVM：zty
- random forest：yyt
- MLP：zkw，尚未接入
- 最后的 K 折和 stacking：zhx

TF-IDF 不进 embedding 循环。`tfidf` 只配 `baseline()`，`tfidf_improved` 只配 `improved_baseline()`。

## 要留下来的比较

1. 只做分词的 TF-IDF + 不加权 LR。
2. 文本特征后的 TF-IDF + 加权 LR。
3. 各分类器自己的提升：权重、正则，以及实际跑过的参数。每一步记下数据和来源。
4. 调参前后的 test 分数。
5. stacking 做完之后，再看它有没有增益。

调参用 `experiments/para_tuning/para_tuning_pipeline.ipynb`。一个 embedding 和分类器的组合对应一份参数，写到 `{EMBEDDING_MODEL}_{ANALYTICS_MODEL}.json`。图用 `Scorer` 的结果。

## 日期

- 11 月 11 日：模型敲定。embedding 训练完成；analytics 接口固定，权重和正则的对比留有记录。
- 11 月 12 日：开始调参，并做可视化。
- 11 月 14 日：汇总结果，决定还要不要改模型，并收齐讲故事用的数据。
- 11 月 15 日：代码冻结，开始写 report 和 ppt。
- 11 月 22 日：提交。
