import sys
import os
import glob
from collections import OrderedDict
import time
# import pyshark
import numpy as np
from scapy.all import PcapReader, IP, TCP, UDP,Ether
import socket
import pickle
import random
import hashlib
import argparse
import ipaddress
import json
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Process, Manager, Value, Queue


#### Dataset path
data_path = '../IoT-dataset/'
file_name = ["benign-dec.pcap", "mirai-ackflooding.pcap","dos-synflooding.pcap","mirai-hostbruteforce.pcap","scan-hostport.pcap"]

class_name = []
for fn in file_name:
    class_name.append(fn.split('.')[0])
file_path = [data_path + i for i in file_name]

SEED = 222
random.seed(SEED)
np.random.seed(SEED)
TRAIN_SIZE = 0.9


class packet_features:
    def __init__(self):
        self.id_fwd = (0, 0)  # 2-tuple_ip_pairs: src_ip_addr, dst_ip_addr
        self.id_bwd = (0, 0)  # 2-tuple_ip_pairs: src_ip_addr, dst_ip_addr
        self.features_list = {
            'raw_bytes':"",
            'pkt_len': 0,
        }

    def __str__(self):
        return "{} -> {}".format(self.id_fwd, self.features_list)

def parse_packet(packet,max_bytes):
    if not packet.haslayer(IP):
        return None

    pf = packet_features()
    tmp_id = [0, 0]

    try:

        pf.features_list['pkt_len'] = int(packet[IP].len)  # packet length
        tmp_id[0] = str(packet[IP].src)  # int(ipaddress.IPv4Address(pkt.ip.src))
        tmp_id[1] = str(packet[IP].dst)  # int(ipaddress.IPv4Address(pkt.ip.dst))

        ## 匿名化MAC地址和IP地址（默认只使用IPv4)
        if packet.haslayer(Ether):
            packet[Ether].dst = '00:00:00:00:00:00'
            packet[Ether].src = "00:00:00:00:00:00"
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"

        ### 限制最多保存前max_bytes个字节
        str_bytes = bytes(packet).hex()[:max_bytes*2]
        pf.features_list['raw_bytes']  = " ".join(str_bytes[i: i + 2] for i in range(0, len(str_bytes), 2))

        pf.id_fwd = (tmp_id[0], tmp_id[1])
        pf.id_bwd = (tmp_id[1], tmp_id[0])

        return pf

    except AttributeError as e:
        # ignore packets that aren't TCP/UDP or IPv4
        return None

# Offline preprocessing of pcap files for model training, validation and testing
def process_pcap(pcap_file, max_flow_len,max_bytes, labelled_flows, time_window,label_id):
    start_time = time.time()
    temp_dict = OrderedDict()
    start_time_window = -1
    label = label_id
    i = 0

    pcap_name = pcap_file.split("/")[-1]
    print("Processing file: ", pcap_name)

    pr = PcapReader(pcap_file)
    while True:
        try:
            pkt = pr.read_packet()
        except EOFError:
            break
        i += 1
        if i % 1000 == 0:
            # import pdb;pdb.set_trace()
            print(pcap_name + " packet #", i, ' label:', label)

        # start_time_window is used to group packets/flows captured in a time-window
        if start_time_window == -1 or float(pkt.time) > start_time_window + time_window:
            start_time_window = float(pkt.time)

        pf = parse_packet(pkt,max_bytes)
        temp_dict = store_packet(pf, temp_dict, start_time_window, max_flow_len, label)

    # import pdb;pdb.set_trace()
    apply_labels(temp_dict, labelled_flows)
    print('Completed file {} in {} seconds.'.format(pcap_name, time.time() - start_time))



def store_packet(pf,temp_dict,start_time_window, max_flow_len, label):#按时间窗口和id对数据包分组
    if pf is not None:
        if pf.id_fwd in temp_dict and start_time_window in temp_dict[pf.id_fwd] and \
                len(temp_dict[pf.id_fwd][start_time_window]) < max_flow_len:
            temp_dict[pf.id_fwd][start_time_window].append(pf.features_list)
        elif pf.id_bwd in temp_dict and start_time_window in temp_dict[pf.id_bwd] and \
                len(temp_dict[pf.id_bwd][start_time_window]) < max_flow_len:
            temp_dict[pf.id_bwd][start_time_window].append(pf.features_list)
        else:
            if pf.id_fwd not in temp_dict and pf.id_bwd not in temp_dict:
                temp_dict[pf.id_fwd] = {start_time_window: [pf.features_list], 'label': label}
            elif pf.id_fwd in temp_dict and start_time_window not in temp_dict[pf.id_fwd]:
                temp_dict[pf.id_fwd][start_time_window] = [pf.features_list]
            elif pf.id_bwd in temp_dict and start_time_window not in temp_dict[pf.id_bwd]:
                temp_dict[pf.id_bwd][start_time_window] = [pf.features_list]
    return temp_dict

def trans_to_txt(flows):
    txt_flows = []
    all_labels = []
    for f in flows:
        f_label = f[1]['label']
        for flow_key, packet_list in f[1].items():
            if flow_key != 'label':
                sample_byte_str = ''
                sample_length_str = ''
                sample_time_str = ''
                for i in packet_list:
                    sample_byte_str += str(i['raw_bytes']) +'\t'
                    sample_length_str += str(i['pkt_len']) + ' '
                sample_all_str = sample_byte_str + sample_length_str.rstrip(' ') +'\t' +str(f_label)
                txt_flows.append(sample_all_str)
                all_labels.append(f_label)
    return txt_flows,all_labels

def apply_labels(flows, labelled_flows):
    for five_tuple, flow in flows.items():
        labelled_flows.append((five_tuple, flow))

# returns the total number of flows
def count_flows(preprocessed_flows):
    class_num_dict = {}
    # total_flows = len(preprocessed_flows)

    for flow in preprocessed_flows:
        f_label = flow[1]['label']
        flow_fragments = len(flow[1]) - 1

        if class_name[f_label] in class_num_dict:
            class_num_dict[class_name[f_label]] += flow_fragments
        else:
            class_num_dict[class_name[f_label]] = flow_fragments

    return class_num_dict


# balance the dataset based on the number of benign and malicious fragments of flows
def balance_dataset(flows,class_num_dict,per_num = 0):
    new_flow_list = []
    # class_num_dict = count_flows(flows)
    balan_num_dict = {}
    class_ip_dict ={}
    # import pdb;pdb.set_trace()
    min_fragments = min(list(class_num_dict.values())) if per_num==0 else per_num

    random.shuffle(flows)

    for flow in flows:
        f_label = flow[1]['label']
        flow_fragments = len(flow[1]) - 1

        if class_name[f_label] in balan_num_dict and (balan_num_dict[class_name[f_label]] < min_fragments):
            balan_num_dict[class_name[f_label]] += flow_fragments
            new_flow_list.append(flow)
            class_ip_dict[class_name[f_label]].append((flow[0], flow_fragments))
        elif class_name[f_label] not in balan_num_dict:
            balan_num_dict[class_name[f_label]] = flow_fragments
            new_flow_list.append(flow)
            class_ip_dict[class_name[f_label]] = [(flow[0], flow_fragments)]

        # import pdb; pdb.set_trace()

    return new_flow_list, balan_num_dict, class_ip_dict


def train_test_split(flow_list, train_size=0.8, shuffle=True):

    train_examples = int(len(flow_list) * train_size)

    if shuffle == True:
        random.shuffle(flow_list)
    train_list = flow_list[:train_examples]
    test_list = flow_list[train_examples:]

    return train_list, test_list


# def main(argv):
argv = sys.argv
command_options = " ".join(str(x) for x in argv[1:])

help_string = 'Usage: python data_process.py -n 30 -p 100 -w 1 -i IoT -s 1000'
manager = Manager()

parser = argparse.ArgumentParser(
    description='Dataset parser',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p', '--bytes_per_packet', nargs='+', type=str,
                    help='Bytes per packet')
parser.add_argument('-n', '--packets_per_flow', nargs='+', type=str,
                    help='Packet per flow sample')
parser.add_argument('-s', '--samples', default=float('inf'), type=int,
                    help='Number of training samples per class ')
parser.add_argument('-i', '--dataset_id', nargs='+', type=str,
                    help='String to append to the names of output files')
parser.add_argument('-w', '--time_window', nargs='+', type=str,
                    help='Length of the time window')

args = parser.parse_args()
MAX_FLOW_LEN = 100
TIME_WINDOW = 1
MAX_BYTES = 200

if args.packets_per_flow is not None:
    max_flow_len = int(args.packets_per_flow[0])
else:
    max_flow_len = MAX_FLOW_LEN

if args.time_window is not None:
    time_window = float(args.time_window[0])
else:
    time_window = TIME_WINDOW

if args.bytes_per_packet is not None:
    max_bytes = int(args.bytes_per_packet[0])
else:
    max_bytes = MAX_BYTES

if args.dataset_id is not None:
    dataset_id = str(args.dataset_id[0])

    output_folder = './data'
    if os.path.exists(output_folder) == False:
        os.mkdir(output_folder)
    start_time = time.time()
    process_list = []
    flows_list = []
    filelist = file_path

    for i,file in enumerate(filelist):
        # import pdb;pdb.set_trace()
        label_id = i
        try:
            flows = manager.list()
            p = Process(target=process_pcap, args=(file, max_flow_len,max_bytes, flows, time_window,label_id))
            process_list.append(p)
            flows_list.append(flows)
        except FileNotFoundError as e:
            continue

    for p in process_list:
        p.start()

    for p in process_list:
        p.join()

    np.seterr(divide='ignore', invalid='ignore')
    try:
        preprocessed_flows = list(flows_list[0])
    except:
        print("ERROR: No traffic flows. \nPlease check that the dataset folder name (" + args.dataset_folder[
            0] + ") is correct and \nthe folder contains the traffic traces in pcap format (the pcap extension is mandatory)")
        exit(1)

    # concatenation of the features
    for results in flows_list[1:]:
        preprocessed_flows = preprocessed_flows + list(results)

    process_time = time.time() - start_time

    filename = str(int(time_window)) + 't-' + str(max_flow_len) + 'n-' + dataset_id + '-preprocess'
    output_file = output_folder + '/' + filename
    output_file = output_file.replace("//", "/")  # remove double slashes when needed

    with open(output_file + '.data', 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(preprocessed_flows, filehandle)

    ### 统计每类样本数量
    class_num = count_flows(preprocessed_flows)
    print("\n class         |    sample_num " )
    # import pdb;pdb.set_trace()
    for cn, sn in class_num.items():
        print(f" {cn}         |      {sn} ")
    print("process_time:" + str(process_time)+ " |\n")

    ###根据指定的样本数量平衡数据集
    balanced_flows, balanced_num_dict, class_ip_dict = balance_dataset(preprocessed_flows,class_num,args.samples)
    print(balanced_num_dict)
    if len(balanced_flows) == 0:
        print("Empty dataset!")
        exit()

    txt_flows,all_label = trans_to_txt(balanced_flows)
    new_balan_flow = txt_flows
    preprocessed_train, preprocessed_test = train_test_split(new_balan_flow, train_size=TRAIN_SIZE, shuffle=True)
    preprocessed_train, preprocessed_val = train_test_split(preprocessed_train, train_size=TRAIN_SIZE, shuffle=True)

    output_file = output_folder + '/' + str(int(time_window)) + 't-' + str(max_flow_len) + 'n-' + dataset_id
    with open(output_file + '-train.txt', 'w') as file:
        for f in preprocessed_train:
            file.write(f + '\n')

    with open(output_file + '-test.txt', 'w') as file:
        for f in preprocessed_test:
            file.write(f + '\n')

    with open(output_file + '-val.txt', 'w') as file:
        for f in preprocessed_val:
            file.write(f + '\n')

