import os
import sys
import random
import torch
import argparse
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score
from corpus import Corpus
from data_process import MyDataset, form_dataset, create_embedding_matrix
from avg import AVG
from cnn import CNN
from lstm import LSTM
from lstm_mem import LSTMMEM
from lstm_bia import LSTMBiA
from lstm_merge import LSTMMerge
from lstm_att import LSTMAtt


def evaluate(model, test_data, need_user_history=False, threshold=0.5):  # evaluation metrics
    if model is not None:
        model.eval()

    labels_all = []
    pred_labels_all = []
    for step, one_data in enumerate(test_data):
        label = one_data[-1].data.numpy()
        if model is not None:
            if need_user_history:
                predictions = model(one_data[0], one_data[1])
            else:
                predictions = model(one_data[0])
            if torch.cuda.is_available():  # run in GPU
                pred_label = predictions.cpu().data.numpy()
            else:
                pred_label = predictions.data.numpy()
        else:
            pred_label = np.ones(len(label), dtype=float)
        labels_all = np.concatenate([labels_all, label])
        pred_labels_all = np.concatenate([pred_labels_all, pred_label])

    try:
        auc = roc_auc_score(labels_all, pred_labels_all)
    except ValueError:
        auc = 0.0
    pred_labels_all = (pred_labels_all >= threshold)
    acc = accuracy_score(labels_all, pred_labels_all)
    pre = precision_score(labels_all, pred_labels_all)
    rec = recall_score(labels_all, pred_labels_all)
    fc = (0 if pre == rec == 0 else 2 * pre * rec / (pre + rec))
    return acc, fc, pre, rec, auc


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(torch.clamp(output, min=1e-15, max=1))) + \
            weights[0] * ((1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1)))
    else:
        loss = target * torch.log(torch.clamp(output, min=1e-15, max=1)) + \
               (1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1))

    return torch.neg(torch.mean(loss))


def train_epoch(model, train_data, loss_weights, optimizer, epoch, need_user_history=False):
    model.train()
    print('Epoch: %d start!' % epoch)
    avg_loss = 0.0
    count = 0
    for step, one_data in enumerate(train_data):
        label = one_data[-1]
        if torch.cuda.is_available():  # run in GPU
            label = label.cuda()
        if need_user_history:
            predictions = model(one_data[0], one_data[1])
        else:
            predictions = model(one_data[0])
        # print predictions, label
        loss = weighted_binary_cross_entropy(predictions, label, loss_weights)
        avg_loss += loss.item()
        count += 1
        if count % 1000 == 0:
            print('Epoch: %d, iterations: %d, loss: %g' % (epoch, count, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    print('Epoch: %d done! Train avg_loss: %g' % (epoch, avg_loss))
    return avg_loss


def train(config):
    filename = config.filename
    modelname = config.modelname
    config.bi_direction = True
    config.pred_pc = 1
    corp = Corpus(filename, config.max_word_num, config.pred_pc)
    if filename == "test.data":
        embedding_matrix = None
    else:
        embedding_matrix = create_embedding_matrix(filename, corp, config.embedding_dim)
    if modelname == "LSTMBiA":
        config.need_history = True
        train_data, test_data, dev_data = form_dataset(corp, config.train_pc, config.batch_size, config.need_history, config.history_size, config.entrytime)
        model = LSTMBiA(config.embedding_dim, corp.wordNum, config.hidden_dim, config.batch_size, config.dropout,
                        config.num_layer, config.model_num_layer, config.bi_direction, embedding_matrix)
    elif modelname == "LSTMMEM":
        config.need_history = True
        train_data, test_data, dev_data = form_dataset(corp, config.train_pc, config.batch_size, config.need_history, config.history_size, config.entrytime)
        model = LSTMMEM(config.embedding_dim, corp.wordNum, config.hidden_dim, config.hop_num, config.batch_size,
                        config.dropout, config.num_layer, config.bi_direction, embedding_matrix)
    elif modelname == "LSTMMerge":
        config.need_history = True
        train_data, test_data, dev_data = form_dataset(corp, config.train_pc, config.batch_size, config.need_history, config.history_size, config.entrytime)
        model = LSTMMerge(config.embedding_dim, corp.wordNum, config.hidden_dim, config.batch_size, config.dropout,
                          config.num_layer, config.bi_direction, embedding_matrix)
    elif modelname == "LSTMAtt":
        config.need_history = True
        train_data, test_data, dev_data = form_dataset(corp, config.train_pc, config.batch_size, config.need_history, config.history_size, config.entrytime)
        model = LSTMAtt(config.embedding_dim, corp.wordNum, config.hidden_dim, config.batch_size, config.dropout,
                        config.num_layer, config.bi_direction, embedding_matrix)
    elif modelname == "LSTM":
        train_data, test_data, dev_data = form_dataset(corp, config.train_pc, config.batch_size, config.need_history, config.history_size, config.entrytime)
        model = LSTM(config.embedding_dim, corp.wordNum, config.hidden_dim, config.batch_size, config.dropout,
                     config.num_layer, config.bi_direction, embedding_matrix)
    elif modelname == "CNN":
        train_data, test_data, dev_data = form_dataset(corp, config.train_pc, config.batch_size, config.need_history, config.history_size, config.entrytime)
        model = CNN(config.embedding_dim, corp.wordNum, config.hidden_dim, config.kernal_num, config.batch_size,
                    config.dropout, config.num_layer, config.bi_direction, embedding_matrix)
    elif modelname == "AVG":
        train_data, test_data, dev_data = form_dataset(corp, config.train_pc, config.batch_size, config.need_history, config.history_size, config.entrytime)
        model = AVG(config.embedding_dim, corp.wordNum, config.hidden_dim, config.batch_size, config.dropout,
                    config.num_layer, config.bi_direction, embedding_matrix)
    else:
        print 'Model name not correct!'
        sys.exit()
    loss_weights = torch.Tensor([1, config.train_weight])
    if torch.cuda.is_available():              # run in GPU
        model = model.cuda()
        loss_weights = loss_weights.cuda()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / ((epoch+1) ** 0.5))

    res_path = "BestResults/" + modelname + "/" + filename.split('.')[0] + "/"
    mod_path = "BestModels/" + modelname + "/" + filename.split('.')[0] + "/"
    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    if not os.path.isdir(mod_path):
        os.makedirs(mod_path)
    first_loc = str(config.entrytime)
    if modelname == "LSTMMEM":
        bi = "1" if config.bi_direction else "0"
        mod_path += first_loc + "_" + str(config.train_pc) + "_" + str(config.batch_size) + "_" + \
            str(config.embedding_dim) + "_" + str(config.hidden_dim) + "_" + str(config.hop_num) + "_" + \
            str(config.lr) + "_" + str(config.dropout) + "_" + str(config.num_layer) + "_" + bi + "_" + \
            str(config.train_weight) + "-" + str(config.runtime) + '.model'
        res_path += first_loc + "_" + str(config.train_pc) + "_" + str(config.batch_size) + "_" + \
            str(config.embedding_dim) + "_" + str(config.hidden_dim) + "_" + str(config.hop_num) + "_" + \
            str(config.lr) + "_" + str(config.dropout) + "_" + str(config.num_layer) + "_" + bi + "_" + \
            str(config.train_weight) + "-" + str(config.runtime) + '.data'
    elif modelname == "LSTMBiA":
        bi = "1" if config.bi_direction else "0"
        mod_path += first_loc + "_" + str(config.train_pc) + "_" + str(config.batch_size) + "_" + \
            str(config.embedding_dim) + "_" + str(config.hidden_dim) + "_" + str(config.lr) + "_" + \
            str(config.dropout) + "_" + str(config.num_layer) + "_" + str(
            config.model_num_layer) + "_" + bi + "_" + str(config.train_weight) + "-" + str(config.runtime) + '.model'
        res_path += first_loc + "_" + str(config.train_pc) + "_" + str(config.batch_size) + "_" + \
            str(config.embedding_dim) + "_" + str(config.hidden_dim) + "_" + str(config.lr) + "_" + \
            str(config.dropout) + "_" + str(config.num_layer) + "_" + str(
            config.model_num_layer) + "_" + bi + "_" + str(config.train_weight) + "-" + str(config.runtime) + '.data'
    elif modelname == "LSTMMerge" or modelname == "LSTMAtt" or modelname == "LSTM" or modelname == "AVG":
        bi = "1" if config.bi_direction else "0"
        mod_path += first_loc + "_" + str(config.train_pc) + "_" + str(config.batch_size) + "_" + \
            str(config.embedding_dim) + "_" + str(config.hidden_dim) + "_" + str(config.lr) + "_" + \
            str(config.dropout) + "_" + str(config.num_layer) + "_" + bi + "_" + str(config.train_weight) + "-" + \
            str(config.runtime) + '.model'
        res_path += first_loc + "_" + str(config.train_pc) + "_" + str(config.batch_size) + "_" + \
            str(config.embedding_dim) + "_" + str(config.hidden_dim) + "_" + str(config.lr) + "_" + \
            str(config.dropout) + "_" + str(config.num_layer) + "_" + bi + "_" + str(config.train_weight) + "-" + \
            str(config.runtime) + '.data'
    elif modelname == "CNN":
        bi = "1" if config.bi_direction else "0"
        mod_path += first_loc + "_" + str(config.train_pc) + "_" + str(config.batch_size) + "_" + \
            str(config.embedding_dim) + "_" + str(config.hidden_dim) + "_" + str(config.kernal_num) + "_" + \
            str(config.lr) + "_" + str(config.dropout) + "_" + str(config.num_layer) + "_" + bi + "_" + \
            str(config.train_weight) + "-" + str(config.runtime) + '.model'
        res_path += first_loc + "_" + str(config.train_pc) + "_" + str(config.batch_size) + "_" + \
            str(config.embedding_dim) + "_" + str(config.hidden_dim) + "_" + str(config.kernal_num) + "_" + \
            str(config.lr) + "_" + str(config.dropout) + "_" + str(config.num_layer) + "_" + bi + "_" + \
            str(config.train_weight) + "-" + str(config.runtime) + '.data'

    best_dev_auc = -1.0
    best_dev_f1 = -1.0
    thr = 0.5
    best_epoch = -1
    if model is not None and modelname != "RANDOM" and modelname != "History":
        no_improve = 0
        for epoch in range(config.max_epoch):
            scheduler.step()
            train_epoch(model, train_data, loss_weights, optimizer, epoch, config.need_history)
            _, dev_f1, _, _, dev_auc = evaluate(model, dev_data, config.need_history)

            if dev_auc > best_dev_auc:
                no_improve = 0
                best_dev_auc = dev_auc
                best_dev_f1 = dev_f1
                os.system('rm ' + mod_path)
                best_epoch = epoch
                print('New Best Dev!!! AUC: %g, F1 Score: %g' % (best_dev_auc, best_dev_f1))
                torch.save(model.state_dict(), mod_path)
            else:
                no_improve += 1

            if no_improve == 8:
                break
        model.load_state_dict(torch.load(mod_path))
    res = evaluate(model, test_data, config.need_history, thr)
    print('Result in test set: Accuracy %g, F1 Score %g, Precision %g, Recall %g, AUC %g' % (res[0], res[1], res[2], res[3], res[4]))
    with open(res_path, 'w') as f:
        f.write('Accuracy: %g, F1 Score: %g, Precision: %g, Recall: %g, AUC: %g\n' % (res[0], res[1], res[2], res[3], res[4]))
        f.write('Dev AUC: %g, F1 Score: %g\n' % (best_dev_auc, best_dev_f1))
        f.write('Best epoch: %d' % best_epoch)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("modelname", type=str, choices=["AVG", "LSTM", "LSTMMEM", "LSTMBiA", "LSTMBiANa", "LSTMMerge", "LSTMAtt", "CNN"])
    parser.add_argument("--cuda_dev", type=str, default="0")
    parser.add_argument("--max_word_num", type=int, default=-1)
    parser.add_argument("--entrytime", type=int, default=1)
    parser.add_argument("--train_pc", type=float, default=0.8)
    parser.add_argument("--embedding_dim", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--kernal_num", type=int, default=50)
    parser.add_argument("--hop_num", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--history_size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--model_num_layer", type=int, default=2)
    parser.add_argument("--need_history", action="store_true")
    parser.add_argument("--bi_direction", action="store_true")
    parser.add_argument("--runtime", type=int, default=0)
    parser.add_argument("--train_weight", type=float, default=2)

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_dev
    train(config)


