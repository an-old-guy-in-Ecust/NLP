import argparse
from nltk.corpus import stopwords
import jieba
import xgboost as xgb
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name for dataset: bq_corpus, lcqmc, paws-x-zh', type=str, default='lcqmc')
args = parser.parse_args()
print(args)


def load_dataset(dataname):
    train = pd.read_csv(dataname + '/train.tsv',
                        sep='[\f\r\t\v]', names=['query1', 'query2', 'label'])

    valid = pd.read_csv(dataname + '/dev.tsv',
                        sep='\t', names=['query1', 'query2', 'label'])

    test = pd.read_csv(dataname + '/test.tsv',
                       sep='\t', names=['query1', 'query2', 'label'])

    return train, valid, test


stops = set(stopwords.words('chinese'))


# A、B共有的字符数
def word_match_share(row):
    q1words = {}
    q2words = {}
    if row['query1'] is not None:
        for word in row['query1']:
            if word not in stops:
                q1words[word] = 1
    if row['query2'] is not None:
        for word in row['query2']:
            if word not in stops:
                q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q2words) + len(q1words))
    return R


def token_match_share(row):
    q1words = {}
    q2words = {}
    if row['query1'] is not None:
        q1 = jieba.lcut(row['query1'])
        for word in q1:
            if word not in stops:
                q1words[word] = 1
    if row['query2'] is not None:
        q2 = jieba.lcut(row['query2'])
        for word in q2:
            if word not in stops:
                q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q2words) + len(q1words))
    return R


# 防止只有一个单词
tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")


def tf_idf(row):
    if row['query1'] is None or row['query2'] is None:
        return np.array([0, 0])
    q1_words = jieba.lcut(row['query1'])
    q2_words = jieba.lcut(row['query2'])
    document_1 = ' '.join(q1_words)
    document_2 = ' '.join(q2_words)
    vector = tfidf_model.fit_transform([document_1, document_2]).toarray()
    return np.array(vector)


def tf_idf_q1(row):
    vector = tf_idf(row)
    return vector[0]


def tf_idf_q2(row):
    vector = tf_idf(row)
    return vector[1]


train, valid, test = load_dataset(args.dataset)
S = StandardScaler()
for index, dataset in enumerate([train, valid, test]):
    if index != 2:
        dataset.dropna(axis=0, subset='label', inplace=True)
    dataset["characters_of_query1"] = dataset.apply(lambda x: len(x["query1"]) if x['query1'] is not None else 0,
                                                    axis=1)  # 有可能为空
    dataset["characters_of_query2"] = dataset.apply(lambda x: len(x["query2"]) if x['query2'] is not None else 1,
                                                    axis=1)
    dataset["word_match_share"] = dataset.apply(word_match_share, axis=1)
    dataset["token_match_share"] = dataset.apply(token_match_share, axis=1)
    dataset["tf_idf_q1"] = dataset.apply(tf_idf_q1, axis=1)
    dataset["tf_idf_q2"] = dataset.apply(tf_idf_q2, axis=1)
    dataset['dist'] = dataset.apply(lambda row: np.sum(np.multiply(row['tf_idf_q1'], row['tf_idf_q2'])), axis=1)
    if index == 0:
        dataset['dist'] = S.fit_transform(dataset['dist'][:, np.newaxis])
    else:
        dataset['dist'] = S.transform(dataset['dist'][:, np.newaxis])
columns = ["characters_of_query1", "characters_of_query2", "word_match_share", "token_match_share", "dist"]
train_feature_data = train[columns]
train_target_data = train["label"]
valid_feature_data = valid[columns]
valid_target_data = valid["label"]
test_feature_data = test[columns]

d_train_data = xgb.DMatrix(train_feature_data.values, label=train_target_data.values)
d_eval_data = xgb.DMatrix(valid_feature_data.values, label=valid_target_data.values)
params = {'max_depth': 4, 'objective': 'binary:logistic', 'eval_metric': ['logloss', 'auc'], "eta": 0.02}
watchlist = [(d_train_data, 'train'), (d_eval_data, 'valid')]
print('train begin')
bst = xgb.train(params, d_train_data, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
print('train done')
d_test_data = xgb.DMatrix(test_feature_data)
predict_test_data = bst.predict(d_test_data)
predict_test_data = [0 if data < 0.5 else 1 for data in predict_test_data]
sub = pd.DataFrame(columns=['index', 'prediction'])
sub['index'] = test_feature_data.index
sub['prediction'] = predict_test_data
sub.to_csv('{}.tsv'.format(args.dataset), index=False, sep='\t')
print('save done')
