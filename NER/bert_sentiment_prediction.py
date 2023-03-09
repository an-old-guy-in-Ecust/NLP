#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import warnings

warnings.filterwarnings('ignore')

SEED = 2021
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# In[2]:


import pandas as pd

# 数据加载
train_data = pd.read_csv('./train_data_public.csv')
test_data = pd.read_csv('./test_public.csv')

X_train = list(train_data['text'])

# 使用其中一部分作为验证
X_test = list(train_data['text'][800:1000])

y_train = list(train_data['class'])
y_test = list(train_data['class'][800:1000])

# 实际的测试集
test_data_sent = list(test_data['text'])  # 列表格式
text_all = X_train + test_data_sent
text_all

# In[3]:


# 转换为tensor
total_targets = torch.tensor(y_train)
total_targets.shape

# In[4]:


# 从预训练模型中加载bert-base-chinese
# [UNK] 特征  [CLS]起始 [SEP]结束
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir="/root/bert/transformer_file/")
tokenizer

# In[5]:


# print(pos_text[2])
X_train[2]

# In[6]:


# 进行分词
print(tokenizer.tokenize(X_train[2]))

# In[7]:


# bert编码，会增加起始[CLS] 和 结束[SEP]标记
print(tokenizer.encode(X_train[2]))

# In[8]:


# 将bert编码转换为 字
print(tokenizer.convert_ids_to_tokens(tokenizer.encode(X_train[2])))
print(tokenizer.convert_ids_to_tokens([101, 2456, 6392, 7213, 6121, 2990, 7583, 2523, 2714, 4638, 100, 100, 102]))

# In[9]:


print(tokenizer.convert_ids_to_tokens([0]))

# In[10]:


# 在的编码为1762，开始[CLS]编码为101，结束[SEP]编码为102
tokenizer.encode('在')


# In[11]:


# 将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
def convert_text_to_token(tokenizer, sentence, limit_size=126):
    tokens = tokenizer.encode(sentence[:limit_size])  # 直接截断
    # 补齐（pad的索引号就是0）
    if len(tokens) < limit_size + 2:
        tokens.extend([0] * (limit_size + 2 - len(tokens)))
    return tokens


# 对每个句子进行编码
input_ids = [convert_text_to_token(tokenizer, x) for x in X_train]
# 放到tensor中
input_tokens = torch.tensor(input_ids)
print(input_tokens.shape)  # torch.Size([10000, 128])

# In[12]:


input_tokens[1]


# In[13]:


# 建立mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        # 如果有编码（>0）即为1, pad为0
        seq_mask = [float(x > 0) for x in seq]
        atten_masks.append(seq_mask)
    return atten_masks


# 生成attention_masks
atten_masks = attention_masks(input_ids)
# 将atten_masks放到tensor中
attention_tokens = torch.tensor(atten_masks)
print(attention_tokens)
print(attention_tokens.size())

# In[14]:


print('input_tokens:\n', input_tokens)  # shape=[10000, 128]
print('total_targets:\n', total_targets)  # shape=[10000, 1]
print('attention_tokens:\n', attention_tokens)  # shape=[10000, 128]
print('input_tokens:\n', input_tokens)  # shape=[10000, 128]
print(input_tokens.shape)

# In[15]:


from sklearn.model_selection import train_test_split

# 使用random_state固定切分方式，切分 train_inputs, train_labels, train_masks,
train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_tokens, total_targets, random_state=2021,
                                                                        test_size=0.2)
train_masks, test_masks, _, _ = train_test_split(attention_tokens, input_tokens, random_state=2021, test_size=0.2)
print(train_inputs.shape, test_inputs.shape)  # torch.Size([8000, 128]) torch.Size([2000, 128])
print(train_masks.shape, test_masks.shape)  # torch.Size([8000, 128])和train_inputs形状一样

print(train_inputs[0])
print(train_masks[0])

# In[16]:


# 使用TensorDataset对tensor进行打包
train_data = TensorDataset(train_inputs, train_masks, train_labels)
# 无放回地随机采样样本元素
train_sampler = RandomSampler(train_data)
# 对于训练集，random sampler, shuffle
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)

# In[17]:


# 查看dataloader内容
for i, (train, mask, label) in enumerate(train_dataloader):
    # torch.Size([16, 128]) torch.Size([16, 128]) torch.Size([16, 1])
    print(train)
    print(mask)
    print(label)
    print(train.shape, mask.shape, label.shape)
    break
print('len(train_dataloader)=', len(train_dataloader))  # 500

# In[18]:


# 加载预训练模型， num_labels表示3个分类，好评和差评, 会下载模型 大约412M
model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=3)
# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model需要放到GPU中
model.to(device)

# In[19]:


# 定义优化器 AdamW， eps默认就为1e-8（增加分母的数值，用来提高数值稳定性）
# optimizer = AdamW(model.parameters(), lr = LEARNING_RATE, eps = EPSILON)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
# AdamW优化器 性能强于Adam
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=EPSILON)
"""
from torch import optim
# 定义优化器
optimizer = optim.Adam(model.parameters())
"""

# In[20]:


epochs = 2
# training steps 的数量: [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * epochs

# 设计 learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


# # # 模型训练、评估

# In[21]:


# 分类结果评估
def class_acc(preds, labels):  # preds.shape=(16, 2) labels.shape=torch.Size([16, 1])
    # eq里面的两个参数的shape=torch.Size([16]) 
    correct = torch.eq(torch.max(preds, dim=1)[1], labels.flatten()).float()
    if 0:
        print('binary acc ********')
        print('preds = ', preds)
        print('labels = ', labels)
        print('correct = ', correct)
    acc = correct.sum().item() / len(correct)
    return acc


# In[22]:


import time
import datetime


# 时间格式化
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))  # 返回 hh:mm:ss 形式的时间


# In[23]:


def train(model, optimizer):
    # 记录当前时刻
    t0 = time.time()
    # 统计m每个batch的loss 和 acc
    avg_loss, avg_acc = [], []

    # 开启训练模式
    model.train()
    for step, batch in enumerate(train_dataloader):
        # 每隔40个batch 输出一下所用时间.
        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
        # 从batch中取数据，并放到GPU中
        b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
            2].long().to(device)
        # 前向传播，得到output
        output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        # 得到loss和预测结果logits  = [0.95, 0.5]
        loss, logits = output[0], output[1]
        # 记录每次的loss和acc
        avg_loss.append(loss.item())
        # 评估acc
        acc = class_acc(logits, b_labels)
        avg_acc.append(acc)
        # 清空上一轮梯度
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 大于1的梯度将其设为1.0, 以防梯度爆炸
        clip_grad_norm_(model.parameters(), 1.0)
        # 更新模型参数
        optimizer.step()
        # 更新learning rate
        scheduler.step()
    # 统计平均loss和acc
    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc


# In[24]:


# 模型评估
def evaluate(model):
    avg_acc = []
    # 表示进入测试模式
    model.eval()

    with torch.no_grad():
        for batch in test_dataloader:
            # 从batch中取数据，并放到GPU中
            b_input_ids, b_input_mask, b_labels = batch[0].long().to(device), batch[1].long().to(device), batch[
                2].long().to(device)
            # 前向传播，得到output
            output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            # 统计当前batch的acc
            acc = class_acc(output[0], b_labels)
            avg_acc.append(acc)
    # 统计平均acc
    avg_acc = np.array(avg_acc).mean()
    return avg_acc


# In[25]:


# 训练 & 评估
for epoch in range(epochs):
    # 模型训练
    train_loss, train_acc = train(model, optimizer)
    print('epoch={},训练准确率={}，损失={}'.format(epoch, train_acc, train_loss))
    # 模型评估
    test_acc = evaluate(model)
    print("epoch={},测试准确率={}".format(epoch, test_acc))

# In[26]:


from tqdm import tqdm


def predict(sen):
    # 将sen 转换为id
    input_id = convert_text_to_token(tokenizer, sen)
    # print(input_id)
    # 放到tensor中
    input_token = torch.tensor(input_id).long().to(device)  # torch.Size([128])
    # 统计有id的部分，即为 1(mask)，并且转换为float类型
    atten_mask = [float(i > 0) for i in input_id]
    # 将mask放到tensor中,并且放到device
    attention_token = torch.tensor(atten_mask).long().to(device)  # torch.Size([128])
    # 转换格式 size= [1,128]， torch.Size([128])->torch.Size([1, 128])否则会报错
    attention_mask = attention_token.view(1, -1)

    output = model(input_token.view(1, -1), token_type_ids=None, attention_mask=attention_mask)
    return torch.max(output[0], dim=1)[1]


sentiment_result = []
for text in tqdm(test_data_sent):
    label = predict(text)
    sentiment_result.append(label.item())
# sentiment_result


# In[27]:


# 读取BIO_anno预测结果，拼接上class预测结果
result = pd.read_csv('./lstm_crf_baseline1.csv')
result['class'] = sentiment_result
result['class'].value_counts()
result.to_csv('./lstm_crf_baseline1-final.csv', index=False)
