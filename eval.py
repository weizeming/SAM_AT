import torch
import torch.nn as nn
import argparse
import torch.nn.functional as F
import os
from model import PreActResNet18
from utils import *


if __name__ == '__main__':
    file_list = os.listdir('models')
    model = PreActResNet18()
    
    PGD1 = PGD(10, 0.25/255., 1./255., 'linf')
    PGD2 = PGD(10, 0.5/255., 2./255., 'linf')    
    
    PGD16 = PGD(10, 2./255., 16./255., 'l2')
    PGD32 = PGD(10, 4./255., 32./255., 'l2')        
    
    _, loader = load_dataset('cifar10', 1024)
    
    for m in file_list:
        ckpt = torch.load('models/' + m, map_location='cpu')
        model.load_state_dict(ckpt)
        model.eval()
        model.cuda()
        accs = []
        for id, attack in enumerate([PGD1, PGD2, PGD16, PGD32]):
            acc = 0
            for x,y in loader:
                x, y = x.cuda(), y.cuda()
                delta = attack.perturb(model, x, y)
                pred = model((normalize_cifar(x+delta)))
                acc += (pred.max(1)[1] == y).float().sum().item()
            acc /= 100
            accs.append(acc)
        print(m)
        print(' & '.join([str(a) for a in accs]))