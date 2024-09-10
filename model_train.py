import logging

import numpy as np
import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from sklearn.metrics import roc_auc_score, confusion_matrix

from STMHGN import STMHGNet
from dataset_process import ABIDEDataset, SZDataset, create_name_for_brain_dataset
from utils import train_val_test_split
from visualization import t_sne, plot_loss_curve, plot_roc_curve, grad_cam
import config

model_name = "STMHGN"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = config.parse()


def train(train_loader, val_loader, test_loader):
    # criterion = torch.nn.CrossEntropyLoss()

    model = STMHGNet(args.input_dim, args.hidden_dim1, args.hidden_dim2, args.num_class)
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params}")

    # Optimizer
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2, eta_min=args.lr / 10)

    # Initial
    val_auc = 0
    min_val_loss = 0
    train_loss_list = []
    val_loss_list = []
    best_epoch = 0
    patience = args.patience
    earlystop = 0

    # train
    for epoch in range(args.num_epochs):
        for param_group in opt.param_groups:
            print("LR", param_group['lr'])
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            _, pred = model(batch)
            label = batch.y.long()
            loss = model.loss(pred, label)
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()
            total_loss += loss.item() * batch.num_graphs
        scheduler.step()
        total_loss /= len(train_loader.dataset)
        train_loss_list.append(total_loss)

        if epoch % 5 == 0:
            train_acc, _, _, _, _, _ = test(train_loader, model)
            val_acc, yprobs, ytrue, sen, spe, val_loss = test(val_loader, model)
            tt, _, _, _, _, _ = test(test_loader, model)
            auc = roc_auc_score(ytrue, yprobs)
            val_loss_list.append(val_loss)
            print("Epoch {}. train_loss: {:.4f}. train_acc: {:.4f}. val_loss: {:.4f}. val_accuracy: {:.4f}. AUC: {:.4f}. sens: {:.4f}. spec: {:.4f}".format(
                    epoch, total_loss, train_acc, val_loss, val_acc, auc, sen, spe))
            print('test_accuracy: {:.4f}.'.format(tt))
            # if auc >= val_auc:
            #     val_auc = auc
            #     max_auc_best_epoch = epoch
            #     print("best model saving !!!!")
            #     torch.save(model, args.save_path + 'best_modelt.pth')

            if min_val_loss == 0:
                min_val_loss = val_loss
                best_epoch = epoch
            elif val_loss <= min_val_loss:
                min_val_loss = val_loss
                best_epoch = epoch
                earlystop = 0
                print("Best model saving.")
                torch.save(model, args.save_path + 'best_modelt.pth')
            else:
                earlystop += 1
                if earlystop == patience:
                    print("Early stopping!")
                    break

    best_model = torch.load(args.save_path + 'best_modelt.pth')
    return best_model, val_auc, train_loss_list, val_loss_list, best_epoch


def test(loader, model):
    # criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    y_pre = []
    y_true = []
    y_probs = []
    total_loss = 0
    for data in loader:
        with torch.no_grad():
            data.to(device)
            _, pred = model(data)
            label = data.y.long()
            loss = model.loss(pred, label)

            probs = torch.softmax(pred, dim=1)
            pred = probs.argmax(dim=1)

            total_loss += loss.item() * data.num_graphs

            y_true.extend(label.tolist())
            y_pre.extend(pred.tolist())
            y_probs.extend(probs[:, 1].tolist())

            correct += pred.eq(label).sum().item()
    total = len(loader.dataset)
    total_loss /= len(loader.dataset)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pre).ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    return correct / total, y_probs, y_true, sen, spe, total_loss


def main(dataset_name, Atlas_name, threshold, time_length, k=5, fold=0):
    node_num = args.num_node
    if dataset_name == 'ABIDE':
        name_dataset = create_name_for_brain_dataset(dataset_name=dataset_name, num_nodes=node_num,
                                                     time_length=time_length,
                                                     threshold=threshold, Atlas_name=Atlas_name)
        dataset = ABIDEDataset(root=name_dataset,
                               num_nodes=node_num,
                               time_length=time_length,
                               threshold=threshold,
                               Atlas_name=Atlas_name
                               )
    else:
        name_dataset = create_name_for_brain_dataset(dataset_name=dataset_name, num_nodes=node_num,
                                                     time_length=time_length,
                                                     threshold=threshold, Atlas_name=Atlas_name)
        dataset = SZDataset(root=name_dataset,
                            num_nodes=node_num,
                            time_length=time_length,
                            threshold=threshold,
                            Atlas_name=Atlas_name,
                            )

    train_dataset, val_dataset, test_dataset = train_val_test_split(dataset=dataset, n_splits=k, fold=fold)
    print('Positive ratio of train_dataset:', int(sum([data.y.item() for data in train_dataset])), '/',
          len(train_dataset))
    print('Positive ratio of val_dataset:', int(sum([data.y.item() for data in val_dataset])), '/',
          len(val_dataset))
    print('Positive ratio of test_dataset:', int(sum([data.y.item() for data in test_dataset])), '/',
          len(test_dataset))

    train_dataset_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    best_model, val_auc, train_loss_list, val_loss_list, best_epoch = train(train_dataset_loader, val_dataset_loader, test_dataset_loader)

    test_acc, score, label, sen, spe, _ = test(test_dataset_loader, best_model)
    logger.info(f"Best epoch : {best_epoch}")
    logger.info('Test metrics of best model: ACC--{:.4f}, AUC--{:.4f}, SENS--{:.4f}, SPEC--{:.4f}'.format(test_acc,
                                                                                                          roc_auc_score(
                                                                                                              label,
                                                                                                              score),
                                                                                                          sen, spe))


if __name__ == "__main__":
    import os
    import random
    from datetime import datetime

    torch.manual_seed(2024)
    np.random.seed(2024)
    random.seed(2024)
    torch.cuda.manual_seed_all(2024)

    k = args.k
    now = datetime.now()
    dataset_name = args.dataset_name
    # Get the current date and time for log file naming
    c_time = f'{now.year}-{now.month}-{now.day}-{now.hour}_{now.minute}'
    log_name = f'{dataset_name}_{model_name}_k-{k}_results_{c_time}'
    log_path = f'./logs/{log_name}.log'

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    Atlas_name = 'Power_WM'
    threshold = args.adj_threshold_ratio
    logger.info(
        f'Dataset == {dataset_name}, Atlas_name == {Atlas_name}, Pool == {args.pooling_strategy}, num_H == {args.num_H}, TCN == {args.use_tcn}')

    for fold in range(0, k):
        logger.info(f'k--{k}, fold--{fold}:')
        logger.info('_'.join([model_name, Atlas_name, 'k_fold-' + str(k) + '_' + str(fold), str(threshold),
                              'lr-' + str(args.learning_rate), 'layer-' + str(args.num_layer),
                              'drop-' + str(args.drop_out_ratio), str(args.weight_decay)]))
        main(dataset_name=dataset_name, Atlas_name=Atlas_name, threshold=threshold, time_length=args.time_length, k=k,
             fold=fold)
