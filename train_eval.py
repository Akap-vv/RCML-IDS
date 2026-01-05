# coding: UTF-8
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import time
import os
from utils import get_time_dif
from tensorboardX import SummaryWriter
import copy
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def train(config, model, train_iter, dev_iter, test_iter, start_train_time):
    if config.load:
        print("\n\nloading model from: {}".format(config.save_path))
        model.load_state_dict(torch.load(config.save_path))

    # L2
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0.001)
    # lr
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992)

    model.train()

    total_batch = 0  # current epoch
    dev_best_loss = float('inf')
    dev_best_acc = 0
    last_improve = 0
    quit = False
    flag = False  # whether it is long time without improvement

    config.loss_path += '_'+start_train_time+'.txt'
    if os.path.exists(config.loss_path):
        f_loss = open(config.loss_path, 'a')
    else:
        f_loss = open(config.loss_path,"w")
    f_loss.write("epoch,train_loss, dev_loss, train_acc, dev_acc\n")

    start_time = time.time()
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if lr>1e-5:
            scheduler.step()
        print("lr now is: ", lr)

        if quit:
            print('best acc =', dev_best_acc, ',exited training!')
            break
        for i, (trains, labels) in enumerate(train_iter):
            # trains: (x, seq_len), y
            outputs = model(trains)
            model.zero_grad()
            if config.imploss:
                # improved loss function
                if len(outputs)==3:
                    loss1 = F.cross_entropy(outputs[0], labels)
                    loss2 = F.cross_entropy(outputs[1], labels)
                    loss3 = F.cross_entropy(outputs[2], labels)
                    loss = loss1 + loss2 + loss3
            else:
                loss = F.cross_entropy(outputs[0], labels)

            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs[0].data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    dev_best_acc = dev_acc *100

                    best_model = copy.deepcopy(model)
                    temp_save_path = config.save_path + '_val_acc{:.2f}.ckpt'.format(dev_best_acc)

                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                if config.imploss:
                    if len(outputs) == 3:
                        msg = 'Iter: {0:>6},  Train Acc: {1:>6.2%},  Val Loss: {2:.4f},  Val Acc: {3:>6.2%},  Time: {4} {5} \n' \
                              'final_Loss: {6:.4f}, raw_loss:{7:.4f}, length_loss:{8:.4f}, sum_Loss: {9:.4f} \n'
                        print(msg.format(total_batch, train_acc, dev_loss, dev_acc, time_dif, improve,
                                         loss1.item(), loss2.item(), loss3.item(), loss.item()))
                else:
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                    print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                f_loss.write('{0:>3d},{1:>5.3f},{2:>5.3f},{3:>5.3f},{4:>5.3f},\n'.format(epoch+1,loss.item(),dev_loss,train_acc,dev_acc))

                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # early stop
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

    end_time = time.time()
    print('Total training time(s):',end_time-start_time)
    print('Training time per epoch(s):',(end_time-start_time)/(epoch+1))
    f_loss.write('\n\n')
    f_loss.close()
    torch.save(best_model.state_dict(), temp_save_path)
    return test(config, model, test_iter,temp_save_path,config.loss_path)


def test(config, model, test_iter,temp_save_path=None,loss_path=None):
    if temp_save_path:
        model.load_state_dict(torch.load(temp_save_path))
    else:
        model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc, test_loss, f1, test_confusion = evaluate(config, model, test_iter, result_path = loss_path,test=True)
    fpr, tpr, ftf = OtherMetrics(test_confusion)
    return test_acc, test_loss, f1, fpr, tpr, ftf

def evaluate(config, model, data_iter, result_path = None, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    diff = .0
    datasize = 0
    with torch.no_grad():
        for texts, labels in data_iter:
            stime = time.time()
            outputs = model(texts)
            etime = time.time()
            loss = F.cross_entropy(outputs[0], labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()

            predic = torch.max(outputs[0].data, 1)[1].cpu().numpy()

            diff += etime-stime
            datasize += predic.shape[0]
            classlist = config.class_list
            pre = predic.tolist()
            lab = labels.tolist()

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        print("\nTotal test time (s): " + str(diff))
        print("Per Sample Use (s): " + str(diff / datasize) + "\n")
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        print(report)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        print(confusion)
        f1 = metrics.f1_score(labels_all, predict_all, average='macro')
        if result_path is not None:
            if os.path.exists(result_path):
                f_loss = open(result_path, 'a')
            f_loss.write("\nTotal Time (s): " + str(diff))
            f_loss.write("\nPer Sample Use (s): " + str(diff / datasize) + "\n")
            f_loss.write(report)
            f_loss.write('\n\n')
            f_loss.close()

        return acc, loss_total / len(data_iter), f1, confusion
    return acc, loss_total / len(data_iter)

def OtherMetrics(cnf_matrix):
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    TPR = TP / (TP + FN)  # recall
    FPR = FP / (FP + TN)

    FTF = 0
    weight = cnf_matrix.sum(axis=1)
    w_sum = weight.sum(axis=0)

    for i in range(len(weight)):
        FTF += weight[i] * TPR[i] / (1+FPR[i])
    FTF /= w_sum

    return float(str(np.around(np.mean(FPR), decimals=4).tolist())), float(str(np.around(np.mean(TPR), decimals=4).tolist())), \
           float(str(np.around(FTF, decimals=4)))
