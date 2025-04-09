import torch
import os
class Config(object):
    def __init__(self,):
        self.model_name = "RCML"
        self.pretrain_path = './Model/pretrain/'
        DATASET = 'IoT'
        record_path = './Model/record/'
        log_path = './Model/log/'
        loss_path = f'./Model/loss/{DATASET}/'
        save_path = f'./Model/save/{DATASET}/'
        dirs = [self.pretrain_path, record_path, log_path, loss_path, save_path]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.dataset = '1t-30n-toniot'
        self.train_path = '../data/toniot/1t-30n/' + '{}-train.txt'.format(self.dataset)
        self.val_path = '../data/toniot/1t-30n/' + '{}-val.txt'.format(self.dataset)
        self.test_path = '../data/toniot/1t-30n/' + '{}-test.txt'.format(self.dataset)
        self.num_classes = 10
        self.class_list = ["Benign", "injection", "MITM", "backdoor", "DDoS", "DoS", "runsomware", "scanning", "XSS",
                           "password"]
        
        self.save_path = save_path
        self.record_path = record_path
        self.loss_path = loss_path
        self.log_path = log_path
        self.vocab_path = './Config/vocab.txt'
        self.n_vocab = 261
        self.length_emb_dim = 1501
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.require_improvement = 10000

        self.bert_dim = 128  # It must be consistent with the settings in pretrain_config.json
        self.num_layers = 2
        self.middle_fc_size = 2048
