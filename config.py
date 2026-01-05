import torch
import os
class Config(object):
    def __init__(self,):
        self.model_name = "RCML"
        self.pretrain_path = './Model/pretrain/'
        record_path = './Model/record/'
        log_path = './Model/log/'
        loss_path = './Model/loss/'
        save_path = './Model/save/'
        dirs = [self.pretrain_path, record_path, log_path, loss_path, save_path]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        self.dataset='1t-30n-IoT'
        self.train_path = './data/1t-30n-IoT-train.txt'
        self.val_path = './data/1t-30n-IoT-val.txt'
        self.test_path = './data/1t-30n-IoT-test.txt'
        self.num_classes = 5
        self.class_list = ["Benign", "ackflooding", "synflooding", "bruteforce", "portscan"]

        self.pretrainModel_json = '{}/config.json'.format(self.pretrain_path)
        self.pretrainModel_path = '{}/1t-30n-IoT_100.pth'.format(self.pretrain_path)

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
