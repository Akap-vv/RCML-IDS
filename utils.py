import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pickle
import json

UNK, PAD, CLS, SEP = '[UNK]', '[PAD]', '[PKT]', '[SEP]'

def build_dataset(config):
    def load_dataset(path, pad_num=10, pad_length=100, pad_len_seq=10):
        cached_dataset_file = './DataCache/{}_{}_{}_{}_{}.txt'.format(config.dataset,pad_num, pad_length, pad_len_seq,config.mode)
        if os.path.exists(cached_dataset_file):
            print("Loading features from cached file {}".format(cached_dataset_file))
            with open(cached_dataset_file, "rb") as handle:
                contents = pickle.load(handle)
            return contents
        else:
            print("Creating ",config.mode," dataset....")
            contents = []
            with open(path, 'r') as f:
                for line in tqdm(f):
                    if not line or line=='\n':
                        continue
                    item = line.split('\t')
                    flow = item[0:-2]  # packets
                    if len(flow) > pad_num:
                        flow = flow[0: pad_num]
                    length_seq = item[-2].strip().split(' ')
                    length_seq = list(map(int, length_seq))
                    length_seq = [1500 if x > 1500 else x for x in length_seq]
                    label = int(item[-1].rstrip())

                    traffic_bytes_idss = []
                    for packet in flow:
                        traffic_bytes = config.tokenizer.tokenize(packet)
                        if len(traffic_bytes) <= pad_length - 2:
                            traffic_bytes = [CLS] + traffic_bytes + [SEP]
                        else:
                            traffic_bytes = [CLS] + traffic_bytes
                            traffic_bytes[pad_length - 1] = SEP
                        traffic_bytes_ids = config.tokenizer.convert_tokens_to_ids(traffic_bytes)

                        if pad_length:
                            if len(traffic_bytes) < pad_length:
                                traffic_bytes_ids += ([0] * (pad_length - len(traffic_bytes)))
                            else:
                                traffic_bytes_ids = traffic_bytes_ids[:pad_length]
                        traffic_bytes_idss.append(traffic_bytes_ids)

                    if pad_len_seq:
                        if len(length_seq) < pad_len_seq:
                            length_seq += [0] * (pad_len_seq - len(length_seq))
                        else:
                            length_seq = length_seq[:pad_len_seq]

                    if pad_num:
                        if len(traffic_bytes_idss) < pad_num:
                            len_tmp = len(traffic_bytes_idss)
                            traffic_bytes_ids = [1] + [0] * (pad_length - 2) + [2]
                            for i in range(pad_num - len_tmp):
                                traffic_bytes_idss.append(traffic_bytes_ids)
                        else:
                            traffic_bytes_idss = traffic_bytes_idss[:pad_num]

                    contents.append((traffic_bytes_idss, length_seq, label))
            print("Saving dataset cached file {}".format(cached_dataset_file))
            with open(cached_dataset_file, "wb") as handle:
                pickle.dump(contents, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return contents
    if config.mode == 'train':
        train = load_dataset(config.train_path, config.pad_num, config.pad_length, config.pad_len_seq)
    elif config.mode == 'test':
        train = load_dataset(config.test_path, config.pad_num, config.pad_length, config.pad_len_seq)
    elif config.mode == 'val':
        train = load_dataset(config.val_path, config.pad_num, config.pad_length, config.pad_len_seq)
    return train

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, pad_num, pad_length, pad_len_seq):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.pad_num = pad_num
        self.pad_length = pad_length
        self.pad_len_seq = pad_len_seq

    def _to_tensor(self, datas):
        # datas: batch_size * contents
        # contents: traffic_bytes_idss, length_seq, int(label)
        traffic_bytes_idss = torch.LongTensor([_[0] for _ in datas])[:,:self.pad_num,:self.pad_length].to(self.device)
        length_seq = torch.LongTensor([_[1] for _ in datas])[:,:self.pad_len_seq]
        length_seq = torch.reshape(length_seq, (-1, self.pad_len_seq, 1)).to(self.device)
        label = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        return (traffic_bytes_idss, length_seq), label

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.pad_num, config.pad_length, config.pad_len_seq)
    return iter


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


