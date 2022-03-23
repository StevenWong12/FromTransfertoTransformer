from model.model import FCN, DilatedConvolution, Classifier, NonLinearClassifier, RNNDecoder
from data.preprocessing import load_data, get_k_fold, split_raw_and_test, k_fold, normalize_per_series
from model.loss import cross_entropy, reconstruction_loss
import numpy as np
import torch
import torch.optim
import sklearn
import pandas as pd
import os


def set_seed(args):

    np.random.seed(args.random_seed)
    # torch.random.seed(args.random_seed)
    sklearn.random.seed(args.random_seed)
    

def build_model(args):
    if args.backbone == 'fcn':
        model = FCN(args.num_classes)
    elif args.backbone == 'dilated':
        model = DilatedConvolution(args.in_channels, args.embedding_channels,
        args.out_channels, args.depth, args.reduced_size, args.kernel_size, args.num_classes)
    
    if args.task == 'classification':
        if args.classifier == 'nonlinear':
            classifier = NonLinearClassifier(args.classifier_input, 128, args.num_classes)
        elif args.classifier =='linear':
            classifier = Classifier(args.classifier_input, args.num_classes)
    
    elif args.task == 'reconstruction':
        if args.decoder_backbone == 'rnn':
            classifier = RNNDecoder()

    return model, classifier

def build_dataset(args):
    sum_dataset, sum_target, num_classes = load_data(args.dataroot, args.dataset)
    
    # torch assert label >= 0 && label < num_classes, wine = {0, 1}
    if args.dataset == 'ArrowHead':
        pass
    elif num_classes > 2 or args.dataset == 'Wine':
        sum_target -= 1
    elif num_classes == 2:
        sum_target = np.maximum(0, sum_target)
    
    
    print(sum_target)
    return sum_dataset, sum_target, num_classes

def get_raw_test_set(dataset, target):
    raw_dataset, raw_target, test_dataset, test_target = split_raw_and_test(dataset, target)
    return raw_dataset, raw_target, test_dataset, test_target

def get_train_val_set(dataset, target):
    training_data, training_target, val_data, val_target = get_k_fold(dataset, target)

    return training_data, training_target, val_data, val_target


def build_loss(args):
    if args.loss == 'cross_entropy':
        return cross_entropy()
    elif args.loss == 'reconstruction':
        return reconstruction_loss()

def build_optimizer(args):
    if args.optimizer == 'adam':
        return torch.optim.Adam(lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return torch.optim.SGD(lr=args.lr, weight_decay=args.weight_decay)


def evaluate(val_loader, model, classifier, loss, device):
    val_loss = 0
    val_accu = 0

    sum_len = 0
    for data, target in val_loader:
        '''
        data, target = data.to(device), target.to(device)
        target = target.to(torch.int64)
        '''
        val_pred = model(data)
        val_pred = classifier(val_pred)
        val_loss += loss(val_pred, target).item()

        val_accu += torch.sum(torch.argmax(val_pred.data, axis=1) == target)
        sum_len += len(target)
    
    return val_loss/sum_len, val_accu/sum_len 

def save_finetune_result(args, accu, var):
    save_path = os.path.join(args.save_dir, args.source_dataset, 'finetune_result.csv')
    accu = accu.cpu().numpy()
    var = var.cpu().numpy()
    if os.path.exists(save_path):
        result_form = pd.read_csv(save_path)
    else:
        result_form = pd.DataFrame(columns=['target', 'accuracy', 'var'])
    
    result_form = result_form.append({'target':args.dataset, 'accuracy':'%.2f' % accu, 'var':'%.4f' % var}, ignore_index=True)
    result_form = result_form.iloc[:, -3:]
    result_form.to_csv(save_path)

def get_all_datasets(data, target):
    return k_fold(data, target)