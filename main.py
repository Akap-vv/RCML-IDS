# coding: UTF-8
import time
import random
import torch
import numpy as np
from train_eval import train,test
import argparse
from config import Config
from utils import build_dataset, build_iterator
import PEAN_model
from torch.nn.parallel import DistributedDataParallel as DDP

## 'Usage: python main.py  --batch 128 --pad_num 30 --pad_len 100 --pad_len_seq 30 --mode 'train' --imploss --feature 'raw+length' --embway pretrain'
parser = argparse.ArgumentParser(description='Traffic Classification')
parser.add_argument('--t', type=int, default=1, help='the time window')
parser.add_argument('--pad_num', type=int, default=30, help='the padding size of packet num')
parser.add_argument('--pad_len', default=100, type=int, help='the padding size(length) of each packet')
parser.add_argument('--pad_len_seq', default=30, type=int, help='the padding size of packet length sequence')
parser.add_argument('--emb', default=128, type=int, help='the emb size of bytes')
parser.add_argument('--load', default=False, action="store_true", help='whether train on previous model')
parser.add_argument('--batch', default=32, type=int, help='batch_size')
parser.add_argument('--feature', default='raw+length', type=str, help='length / raw /raw+length')
parser.add_argument('--method', default='trf', type=str, help='lstm / trf (Sequential Layer)')
parser.add_argument('--embway', default='random', type=str, help='random / pretrain (for raw)')
parser.add_argument('--imploss', default=False, action="store_true", help='whether to use improved loss')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--length_emb_size', default=32, type=int, help='len emb size')
parser.add_argument('--lenhidden', default=128, type=int, help='len hidden size')
parser.add_argument('--embhidden', default=1024, type=int, help='emb hidden size')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--trf_heads', default=8, type=int, help='transformers heads number')
parser.add_argument('--trf_layers', default=2, type=int, help='transformers layers')
parser.add_argument('--mode', default='train', type=str, help='train/test')
parser.add_argument('--epoch', default='100', type=int, help='epoch')
parser.add_argument('--cuda', default='0', type=int, help='GPU number')
args = parser.parse_args()


def get_model(config):
    return PEAN_model.PEAN(config).to(config.device)
    # return DDP(PEAN_model.PEAN(config).cuda())

def get_config():
    config = Config()
    config.t = args.t
    config.pad_num = args.pad_num
    config.pad_length = args.pad_len
    config.pad_len_seq = args.pad_len_seq
    config.mode = args.mode
    config.embedding_size = args.emb
    config.batch_size = args.batch
    config.load = args.load
    config.lenlstmhidden_size = args.lenhidden
    config.emblstmhidden_size = args.embhidden
    config.feature = args.feature
    config.method = args.method
    config.embway = args.embway
    config.length_emb_size = args.length_emb_size
    config.imploss = args.imploss
    config.learning_rate = args.lr
    config.seed = args.seed
    config.trf_heads = args.trf_heads
    config.trf_layers = args.trf_layers
    config.num_epochs = args.epoch
    config.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    # import pdb;pdb.set_trace()
    if args.mode == "test":
        config.load = True
        # config.num_epochs = 0
    name = "{}t_{}_lr{}_ep{}".format(config.t,config.feature,config.learning_rate,config.num_epochs)
    if "raw" in config.feature:
        name += "_{}_{}_{}_{}".format(config.embway, config.method,config.pad_num, config.pad_length)
    if "length" in config.feature:
        name += "_{}".format(config.pad_len_seq)
    if config.method == "trf":
        if config.trf_heads == 8 and config.trf_layers == 2:
            pass
        else:
            name += "_{}_{}".format(config.trf_heads, config.trf_layers)
    if config.imploss:
        name += "_imploss"
    config.loss_path = config.loss_path + name    # record loss
    config.save_path = config.save_path + name     # record saved model
    # import pdb;pdb.set_trace()
    if config.mode == 'train':
        print("\nModel save at: ", config.save_path)
    elif config.mode == "test":
        config.save_path = config.save_path + '_test_acc95.73.ckpt'
        print("\nModel load from: ", config.save_path)
    from transformers import BertTokenizer
    config.tokenizer = BertTokenizer(vocab_file=config.vocab_path, max_seq_length=config.pad_num - 2, max_len=config.pad_num)

    return config

def prepare_data(mode):
    print("----------------------------\n")

    msg = "Iput Feature: {}\nRandom Seed: {}\n".format(config.feature, config.seed)
    if "raw" in config.feature:
        msg += "Sequential use: {}\n".format(config.method)
        msg += "Embedding way: {}(hidden:{})\n".format(config.embway, config.emblstmhidden_size)
        if config.method == "pretrain":
            msg += "Bert Size: {}\n".format(config.bert_dim)
        else:
            msg += "Embedding Size: {}\n".format(config.embedding_size)
        msg += "Pad_num: {}\n".format(config.pad_num)
        msg += "Pad_len: {}\n".format(config.pad_length)

    if "length" in config.feature:
        msg += "Length use: lstm(emb: {}, hidden:{})\n".format(config.length_emb_size, config.lenlstmhidden_size)
        msg += "Pad_len_seq: {}\n".format(config.pad_len_seq)

    if config.method == "trf":
        msg += "trf heads:{}\n".format(config.trf_heads)
        msg += "trf_layers: {}\n".format(config.trf_layers)

    msg += "Use Improved loss: {}\n".format(config.imploss)
    msg += "Learning Rate: {}\n".format(config.learning_rate)
    msg += "Batch Size:{}\n".format(config.batch_size)

    print(msg)
    print("----------------------------\n")
    print("Loading data...")

    if mode == 'train':
        train_data = build_dataset(config)
        print("train_set: {}".format(len(train_data)))
        return train_data
    elif mode == 'val':
        config.mode = 'val'
        val_data = build_dataset(config)
        print("val_set: {}".format(len(val_data)))
        return val_data
    elif mode == 'test':
        config.mode = 'test'
        test_data = build_dataset(config)
        print("test_set: {}".format(len(test_data)))
        return test_data

if __name__ == '__main__':
    config = get_config()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    if config.mode == "train":
        train_data = prepare_data(mode='train')
        val_data = prepare_data(mode='val')
        test_data = prepare_data(mode='test')
        model = get_model(config)

        # print(model.parameters, "\n")
        print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

        import time
        start_train_time = time.strftime('%m-%d_%H.%M', time.localtime())

        train_iter = build_iterator(train_data, config)
        test_iter = build_iterator(test_data, config)
        dev_iter = build_iterator(val_data, config)

        acc_, loss_, f1_, fpr_, tpr_, ftf_ = train(config, model, train_iter, dev_iter, test_iter, start_train_time)

    elif config.mode == "test":
        test_data = prepare_data(mode='test')
        model = get_model(config)
        best_model_path = './Model/save/1t_raw+length_lr0.001_ep10_pretrain_trf_30_100_30_imploss_val_acc65.29.ckpt'
        test_iter = build_iterator(test_data, config)
        test_acc, test_loss, f1, fpr, tpr, ftf = test(config, model, test_iter, temp_save_path=best_model_path)
        print('acc:%.4f' % test_acc, 'F1-macro:%.4f' % f1, \
              'TPR:%.4f' % tpr, 'FPR:%.4f' % fpr, \
              'FTF:%.4f' % ftf, 'loss:%.6f' % test_loss, "\n")

