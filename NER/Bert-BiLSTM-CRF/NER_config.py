from transformers import BertConfig
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ner_config = BertConfig.from_pretrained('bert-base-chinese')
ner_config.num_classes = 12
ner_config.clip_grad = 5
ner_config.epoch_num = 2
ner_config.min_epoch_num = 1
ner_config.patience = 0.0002
ner_config.patience_num = 1
ner_config.learning_rate = 1e-5
ner_config.batch_size = 1
ner_config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
START_TAG, STOP_TAG = '[CLS]', '[SEP]'
tag_to_ix = {"[PAD]": 0, "O": 1, "B-BANK": 2, "I-BANK": 3, "B-PRODUCT": 4, 'I-PRODUCT': 5,
             'B-COMMENTS_N': 6, 'I-COMMENTS_N': 7, 'B-COMMENTS_ADJ': 8,
             'I-COMMENTS_ADJ': 9, STOP_TAG: 10, START_TAG: 11}
idx_to_tag = {idx: tag for idx, tag in enumerate(tag_to_ix)}
