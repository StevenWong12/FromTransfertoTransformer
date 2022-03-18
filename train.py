import numpy as np
import argparse

from utils import build_model, get_raw_test_set, get_train_val_set, set_seed, build_dataset, build_loss, evaluate, save_finetune_result
from data.dataloader import UCRDataset
from data.preprocessing import normalize
from torch.utils.data import DataLoader
import os
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Base setup
    parser.add_argument('--backbone', type=str, default='fcn', help='encoder backbone, fcn or dilated')
    parser.add_argument('--task', type=str, default='classification', help='classification or reconstruct')
    parser.add_argument('--random_seed', type=int, default=43, help='shuffle seed')

    # Dataset setup
    parser.add_argument('--dataset', type=str, default=None, help='dataset(in ucr)')
    parser.add_argument('--dataroot', type=str, default=None, help='path of UCR folder')
    parser.add_argument('--num_classes', type=int, default=0,  help='number of class')


    # Dilated Convolution setup
    parser.add_argument('--depth', type=int, default=3, help='depth of the dilated conv model')
    parser.add_argument('--in_channels', type=int, default=1, help='input data channel')
    parser.add_argument('--embedding_channels', type=int, default=40, help='mid layer channel')
    parser.add_argument('--reduced_size', type=int, default=160, help='number of channels after Global max Pool')
    parser.add_argument('--out_channels', type=int, default=320, help='number of channels after linear layer')
    parser.add_argument('--kernel_size', type=int, default=3, help='convolution kernel size')

    # training setup
    parser.add_argument('--loss', type=str, default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=128, help='(32, 128) larger batch size on the big dataset, ')
    parser.add_argument('--epoch', type=int, default=1000, help='training epoch')
    parser.add_argument('--mode', type=str, default='pretrain', help='train mode, default pretrain')
    parser.add_argument('--save_dir', type=str, default='./result')

    # fintune setup
    parser.add_argument('--source_dataset', type=str, default=None, help='source dataset of the pretrained model')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args)



    sum_dataset, sum_target, num_classes = build_dataset(args)
    args.num_classes = num_classes

    model, classifier = build_model(args)
    model, classifier = model.to(device), classifier.to(device)
    loss = build_loss(args).to(device)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params' : classifier.parameters()}], 
        lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)


    if args.mode == 'pretrain':
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)

        if not os.path.exists(os.path.join(args.save_dir, args.dataset)):
            os.mkdir(os.path.join(args.save_dir, args.dataset))
        print('{} started pretrain'.format(args.dataset))

        train_set = UCRDataset(sum_dataset, sum_target)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=10)
        
        min_loss = torch.inf
        min_epoch = 0
        model_to_save = None

        num_steps = train_set.__len__() // args.batch_size
        for epoch in range(args.epoch):
            model.train()
            epoch_loss = 0
            epoch_accu = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                y = y.to(torch.int64)
                optimizer.zero_grad()
                pred = model(x)

                pred = classifier(pred)
                
                step_loss = loss(pred, y)

                step_loss.backward()
                optimizer.step()

                epoch_loss += step_loss.item()
                epoch_accu += torch.sum(torch.argmax(pred.data, axis=1) == y) / len(y)
            
            epoch_loss /= num_steps
            if epoch_loss < min_loss:
                min_loss = epoch_loss
                min_epoch = epoch
                model_to_save = model.state_dict()

            epoch_accu /= num_steps
            print("epoch : {}, loss : {}, accuracy : {}".format(epoch, epoch_loss, epoch_accu))
        
        print('{} finished pretrain, with min loss {} at epoch {}'.format(args.dataset, min_loss, min_epoch))
        torch.save(model_to_save, os.path.join(args.save_dir, args.dataset, 'pretrain_weights.pt'))


    if args.mode == 'finetune':
        print('start finetune on {}'.format(args.dataset))

        raw_dataset, raw_target, test_dataset, test_target = get_raw_test_set(sum_dataset, sum_target)
        train_datasets, train_targets, val_datasets, val_targets = get_train_val_set(raw_dataset, raw_target)

        losses = []
        accuracies = []
        for i, train_dataset in enumerate(train_datasets):
            model.load_state_dict(torch.load(os.path.join(args.save_dir, args.source_dataset, 'pretrain_weights.pt')))
            print('{} fold start training and evaluate'.format(i))
            max_accuracy = 0

            train_target = train_targets[i]
            val_dataset = val_datasets[i]
            val_target = val_targets[i]

            # normalize 
            train_dataset = normalize(train_dataset)
            val_dataset = normalize(val_dataset)

            train_set = UCRDataset(train_dataset, train_target)
            val_set = UCRDataset(val_dataset, val_target)

            
            train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=10)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=10)

            train_loss = []
            train_accuracy = []
            num_steps = args.epoch // args.batch_size

            last_loss = 100
            stop_count = 0
            increase_count = 0

            num_steps = train_set.__len__() // args.batch_size
            for epoch in range(args.epoch):
                if stop_count == 10 or increase_count == 10:
                    print('model convergent at epoch {}, early stopping'.format(epoch))
                    break

                epoch_train_loss = 0
                epoch_train_acc = 0
                model.train()
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    y = y.to(torch.int64)
                    optimizer.zero_grad()
                    pred = model(x)
                    pred = classifier(pred)

                    step_loss = loss(pred, y)
                    step_loss.backward()
                    optimizer.step()

                    epoch_train_loss += step_loss.item()
                    epoch_train_acc += torch.sum(torch.argmax(pred.data, axis=1) == y) / len(y)
                
                epoch_train_loss /= num_steps
                epoch_train_acc /= num_steps

                
                model.eval()
                val_loss, val_accu = evaluate(val_loader, model, classifier, loss, device)
            
                print("epoch : {}, train loss: {} , train accuracy : {}, \nval loss : {}, val accuracy : {}".format(epoch, epoch_train_loss, epoch_train_acc, val_loss, val_accu))
                
                
                train_loss.append(val_loss)
                train_accuracy.append(val_accu)

                max_accuracy = max(max_accuracy, val_accu)

                if abs(last_loss-val_loss) / last_loss <= 1e-4:
                    stop_count += 1
                else:
                    stop_count = 0 

                if val_loss > last_loss:
                    increase_count += 1
                else:
                    increase_count = 0

                last_loss = val_loss
            losses.append(train_loss)
            accuracies.append(max_accuracy)

            print('{} fold finish training'.format(i))

        accuracies = torch.Tensor(accuracies)
        save_finetune_result(args, torch.mean(accuracies), torch.var(accuracies))
        print('Done!')

