# A Real-Time Channel-Level Intrusion Detection System Based on Multimodal Learning
### Overview
This paper proposes RCML-IDS, a Real-time Channel-level Intrusion Detection System based on Multimodal Learning. The core novelty of RCML-IDS lies in its real-time, online processing capability, enabled by a time-window-based traffic preprocessing mechanism. Additionally, it performs channel-level traffic aggregation and integrates multimodal features, namely raw bytes and packet lengths, to capture rich behavioral patterns from encrypted traffic. Architecturally,
two Transformers learn multi-level byte representations from local to global contexts, while an LSTM captures temporal patterns in packet length sequences. 

### Requirements
All experiments are implemented on the PyTorch platform with an NVIDIA RTX 3090 GPU. Python package information is summarized in `requirements.txt`.

### Quick start

Follow these steps to run RCML-IDS:

#### 1. Data Preprocessing
Extract channel features from raw PCAP files:
```
python data_process.py -n 30 -p 100 -w 1 -i IoT -s 10000
```

#### 2. Pre-training
Train Transformer encoders on raw byte features:
```
python pretrain.py --do_eval --do_train
```

#### 3. Formal Training
Train the multimodal classification model:
```
python main.py --batch 128 --pad_num 30 --pad_len 100 --pad_len_seq 30 --mode 'train' --imploss --feature 'raw+length' --embway pretrain
```
