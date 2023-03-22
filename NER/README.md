### NER

项目链接：https://www.datafountain.cn/competitions/529/datasets

数据背景和介绍详见链接。 任务分为两个：NER和情感分类

1. NER
   官方给的baseline为BiLSTM—CRF。

具体效果如下表，效果由严格匹配的f1值度量与分类结果的f1值取平均

| NER             | 超参数设置               | f1 score |
|-----------------|---------------------|----------|
| BiLSTM-CRF      | 无                   | 0.10268  |
| BERT-BiLSTM-CRF | BATCH_SIZE=4        | 0.261014 |
| BERT-BiLSTM-CRF | BATCH_SIZE=1        | 0.302745 |
| BERT-BiLSTM-CRF | BATCH_SIZE=1,加入对抗训练 | 0.35199  |

2. 情感分析
   官方给了两个baseline，一个为基于TF-idf的线性回归分类器，另一个为Bert

具体效果如下：

| SA                      | 超参数设置                   | f1 score(加上NER最好的结果） |
|-------------------------|-------------------------|----------------------|
| Bert                    | 无                       | 0.61326              |
| RoBerta_BiGRU_Attention | BATCH_SIZE=4 + EDA      | 0.582776             |
| RoBerta_BiGRU_Attention | BATCH_SIZE=4 +EDA +K折交叉 | 0.615176             |

2023.3.14限制过拟合的瓶颈，无论怎么优化过拟合都没有好的效果，做了包括对抗训练、数据增强（同义词随机替换等）、标签平滑和K折交叉验证，验证集准确率可以到达99%，可是测试集表现与bert
baseline一样。

运行环境：Colab平台 GPU免费版 学校服务器 配置 2080Ti

代码reference：

1. https://blog.csdn.net/weixin_44750512/article/details/128460220?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-128460220-blog-125541213.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EYuanLiJiHua%7EPosition-3-128460220-blog-125541213.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=6
2. https://gitcode.net/mirrors/hemingkx/cluener2020/-/tree/main/BERT-LSTM-CRF

思路讲解reference：

1. https://www.datafountain.cn/competitions/529/submits?view=submit-records