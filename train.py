import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
from time import time

from utils import *
from model import PreActResNet18
from sam import SAM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--max-lr', default=0.1, type=float)
    parser.add_argument('--opt', default='SAM', choices=['SAM', 'SGD'])
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--rho', default=0.05, type=float) # for SAM
    
    parser.add_argument('--norm', default='linf', choices=['linf', 'l2'])
    parser.add_argument('--train-eps', default=8., type=float)
    parser.add_argument('--train-alpha', default=2., type=float)
    parser.add_argument('--train-step', default=5, type=int)
    
    parser.add_argument('--test-eps', default=1., type=float)
    parser.add_argument('--test-alpha', default=0.5, type=float)
    parser.add_argument('--test-step', default=5, type=int)
    return parser.parse_args()

args = get_args()

def lr_schedule(epoch):
    if epoch < args.epochs * 0.75:
        return args.max_lr
    elif epoch < args.epochs * 0.9:
        return args.max_lr * 0.1
    else:
        return args.max_lr * 0.01

if __name__ == '__main__':
    
    dataset = args.dataset
    device = f'cuda:{args.device}'
    model = PreActResNet18(10 if dataset == 'cifar10' else 100).to(device)
    train_loader, test_loader = load_dataset(dataset, args.batch_size)
    params = model.parameters()
    criterion = nn.CrossEntropyLoss()


    if args.opt == 'SGD': 
        opt = torch.optim.SGD(params, lr=args.max_lr, momentum=0.9, weight_decay=5e-4)
    elif args.opt == 'SAM':
        base_opt = torch.optim.SGD
        opt = SAM(params, base_opt,lr=args.max_lr, momentum=0.9, weight_decay=5e-4, rho=args.rho)
    normalize = normalize_cifar if dataset == 'cifar10' else normalize_cifar100

    all_log_data = []
    train_pgd = PGD(args.train_step, args.train_alpha / 255., args.train_eps / 255., args.norm, False, normalize)
    test_pgd = PGD(args.test_step, args.test_alpha / 255., args.test_eps / 255., args.norm, False, normalize)

    for epoch in range(args.epochs):
        start_time = time()
        log_data = [0,0,0,0,0,0] # train_loss, train_acc, test_loss, test_acc, test_robust_loss, test_robust
        # train
        model.train()
        lr = lr_schedule(epoch)
        opt.param_groups[0].update(lr=lr)
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if args.adv:  
                delta = train_pgd.perturb(model, x, y)
            else:
                delta = torch.zeros_like(x).to(x.device)
                
            output = model(normalize(x + delta))
            loss = criterion(output, y)
            
            if args.opt == 'SGD':
                opt.zero_grad()
                loss.backward()
                opt.step()
                
            elif args.opt == 'SAM':
                loss.backward()
                opt.first_step(zero_grad=True)

                output_2 = model(normalize(x + delta))
                criterion(output_2, y).backward()
                opt.second_step(zero_grad=True)
            
            log_data[0] += (loss * len(y)).item()
            log_data[1] += (output.max(1)[1] == y).float().sum().item()
            
        # test
        model.eval()
        for x, y in test_loader:
            
            x, y = x.to(device), y.to(device)
            # clean
            output = model(normalize(x)).detach()
            loss = criterion(output, y)
            
            log_data[2] += (loss * len(y)).item()
            log_data[3] += (output.max(1)[1] == y).float().sum().item()
            continue
            delta = test_pgd.perturb(model, x, y)
            output = model(normalize(x + delta)).detach()
            loss = criterion(output, y)
            
            log_data[4] += (loss * len(y)).item()
            log_data[5] += (output.max(1)[1] == y).float().sum().item()
        
        log_data = np.array(log_data)
        log_data[0] /= 60000
        log_data[1] /= 60000
        log_data[2] /= 10000
        log_data[3] /= 10000
        log_data[4] /= 10000
        log_data[5] /= 10000
        all_log_data.append(log_data)
        
        print(f'Epoch {epoch}:\t',log_data,f'\tTime {time()-start_time:.1f}s')
        torch.save(model.state_dict(), f'models/{args.fname}.pth' if args.dataset == 'cifar10' else f'cifar100_models/{args.fname}.pth')
        
    all_log_data = np.stack(all_log_data,axis=0)
    
    df = pd.DataFrame(all_log_data)
    df.to_csv(f'logs/{args.fname}.csv')
    
    
    plt.plot(all_log_data[:, [2,4]])
    plt.grid()
    # plt.title(f'{dataset} {args.opt}{" adv" if args.adv else ""} Loss', fontsize=16)
    plt.legend(['clean', 'robust'], fontsize=16)
    plt.savefig(f'figs/{args.fname}_loss.png', dpi=200)
    plt.clf()
    
    plt.plot(all_log_data[:, [3,5]])
    plt.grid()
    #plt.title(f'{dataset} {args.opt}{" adv" if args.adv else ""} Acc', fontsize=16)
    plt.legend(['clean', 'robust'], fontsize=16)
    plt.savefig(f'figs/{args.fname}_acc.png', dpi=200)
    plt.clf()
