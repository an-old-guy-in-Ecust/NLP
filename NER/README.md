### NER

项目链接：https://www.datafountain.cn/competitions/529/datasets

数据背景和介绍详见链接。 任务分为两个：NER和情感分类

1. NER
   官方给的baseline为BiLSTM—CRF。这里采用BERT-BiLSTM-CRF模型
2. 情感分析
   官方给了两个baseline，一个为基于TF-idf的线性回归分类器，另一个

具体效果如下表，效果由严格匹配的f1值度量与分类结果的f1值取平均

| NER        | 情感分类 | 超参数设置 | f1 score|
|------------|------| ---- | ----|
| BiLSTM-CRF | LR   | 无|0.10268|
| BiLSTM-CRF | BERT |无特殊设置|0.35841|
| BERT-BiLSTM-CRF| BERT|BATCH_SIZE=4|0.41976|
| BERT-BiLSTM-CRF|BERT|BATCH_SIZE=1|0.56504|