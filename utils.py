import torch
import torch.nn as nn
from torch.utils.data import dataset, dataloader
from torchvision import datasets, transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

mu = torch.tensor(cifar10_mean).view(3,1,1)
std = torch.tensor(cifar10_std).view(3,1,1)

def normalize_cifar(x):
    return (x - mu.to(x.device))/(std.to(x.device))

CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

mu_cifar100 = torch.tensor(CIFAR100_MEAN).view(3,1,1).cuda()
std_cifar100 = torch.tensor(CIFAR100_STD).view(3,1,1).cuda()

def normalize_cifar100(x):
    return (x - mu_cifar100.to(x.device))/(std_cifar100.to(x.device))

def load_dataset(dataset='cifar10', batch_size=128):
    if dataset == 'cifar10':
        transform_ = transforms.Compose([transforms.ToTensor()])
        train_transform_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/data/cifar_data', train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/data/cifar_data', train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    
    elif dataset == 'cifar100':
        transform_ = transforms.Compose([transforms.ToTensor()])
        train_transform_ = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/data/cifar_data', train=True, download=True, transform=train_transform_),
            batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('/data/cifar_data', train=False, download=True, transform=transform_),
            batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    
class Attack():
    def __init__(self, iters, alpha, eps, norm, criterion, rand_init, rand_perturb, targeted, normalize=normalize_cifar):
        self.iters = iters
        self.alpha = alpha
        self.eps = eps
        self.norm = norm
        assert norm in ['linf', 'l2']
        self.criterion = criterion       # loss function for perturb
        self.rand_init = rand_init       # random initialization before perturb
        self.rand_perturb = rand_perturb # add random noise in each step
        self.targetd = targeted          # targeted attack
        self.normalize = normalize       # normalize_cifar

    def perturb(self, model, x, y):
        delta = torch.zeros_like(x).to(x.device)
        if self.rand_init:
            
            if self.norm == "linf":
                delta.uniform_(-self.eps, self.eps)
            elif self.norm == "l2":
                delta.normal_()
                d_flat = delta.view(delta.size(0),-1)
                n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*self.eps
            else:
                raise ValueError

        delta = torch.clamp(delta, 0-x, 1-x)
        delta.requires_grad = True

        for _ in range(self.iters):
            output = model(self.normalize(x+delta))
            loss = self.criterion(output, y)
            if self.targetd:
                loss *= -1
            loss.backward()
            g = delta.grad.detach()
            if self.norm == "linf":
                d = torch.clamp(delta + self.alpha * torch.sign(g), min=-self.eps, max=self.eps).detach()
            elif self.norm == "l2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (delta + scaled_g*self.alpha).view(delta.size(0),-1).renorm(p=2,dim=0,maxnorm=self.eps).view_as(delta).detach()
            d = torch.clamp(d, 0 - x, 1 - x)
            delta.data = d
            delta.grad.zero_()

        return delta.detach()

class PGD(Attack):
    def __init__(self, iters, alpha, eps, norm, targeted=False, normalize=normalize_cifar):
        super().__init__(iters, alpha, eps, norm, nn.CrossEntropyLoss(), True, False, targeted, normalize=normalize)