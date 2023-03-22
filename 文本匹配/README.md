题目链接：https://aistudio.baidu.com/aistudio/competition/detail/45/0/task-definition
文本匹配的二分类

simple_xgb.ipynb: 使用人工设计的特征输入XGBoost模型，人工输入特征包括句子的长度、中文句子分词后的长度、基于TF-idf句子向量的内积相似度（归一化）、句子之间相同字的大小、相同token的大小。
simple_xgb.py: 输入python simple_xgb.py --dataset dataset_name 即可开始训练，并生成结果保存为tsv文件。

bq_corpus,lcqmc,paws-x-zh 为数据集

Reference：

1. https://coggle.club/blog/30days-of-ml-202201
2. https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb