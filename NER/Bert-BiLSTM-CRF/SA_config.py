from transformers import BertConfig
import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sa_config = BertConfig.from_pretrained('bert-base-chinese')
sa_config.num_classes = 3
sa_config.clip_grad = 5
sa_config.num_heads = 3
sa_config.epoch_num = 1
sa_config.min_epoch_num = 1
sa_config.patience = 0.0002
sa_config.patience_num = 1
sa_config.learning_rate = 1e-5
sa_config.dropout_prob = 0.1
sa_config.batch_size = 4
sa_config.alpha = torch.tensor([0.27, 0.7, 0.03])
sa_config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
