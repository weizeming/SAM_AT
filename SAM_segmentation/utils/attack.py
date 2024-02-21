import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

voc_mu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
voc_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def normalize_voc(x):
    return (x - voc_mu.to(x.device))/(voc_std.to(x.device))

city_mu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
city_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
def normalize_city(x):
    return (x - city_mu.to(x.device))/(city_std.to(x.device))

iccv09_mu = torch.tensor([0.4813, 0.4901, 0.4747]).view(3, 1, 1)
iccv09_std = torch.tensor([0.2495, 0.2492, 0.2748]).view(3, 1, 1)
def normalize_iccv09(x):
    return (x - iccv09_mu.to(x.device))/(iccv09_std.to(x.device))

class Attack():
    def __init__(self, iters, alpha, eps, norm, criterion, rand_init, rand_perturb, targeted, normalize):
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
        assert x.min() >= 0 and x.max() <= 1
        delta = torch.zeros_like(x, device=x.device)
        if self.rand_init:
            if self.norm == "linf":
                delta.uniform_(-self.eps, self.eps)
            elif self.norm == "l2":
                delta.normal_()
                d_flat = delta.view(delta.size(0), -1)
                n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
                r = torch.zeros_like(n).uniform_(0, 1)
                delta *= r/n*self.eps
            else:
                raise NotImplementedError("Only linf and l2 norms are implemented.")
            
        delta = torch.clamp(delta, 0-x, 1-x)
        delta.requires_grad = True

        for i in range(self.iters):
            # output = model(self.normalize(x+delta))
            output = model(x+delta)
            loss = self.criterion(output, y)
            if self.targetd:
                loss = -loss
            loss.backward()
            grad = delta.grad.detach()
            if self.norm == "linf":
                d = torch.clamp(delta + self.alpha * torch.sign(grad), min=-self.eps, max=self.eps).detach()
            elif self.norm == "l2":
                grad_norm = torch.norm(grad.view(grad.size(0), -1), dim=1).view(-1, 1, 1, 1)
                scaled_grad = grad / (grad_norm + 1e-10)
                d = (delta + scaled_grad * self.alpha).view(delta.size(0), -1).renorm(p=2, dim=0, maxnorm=self.eps).view_as(delta).detach()

            d = torch.clamp(d, 0-x, 1-x)
            delta.data = d
            delta.grad.zero_()

        return delta.detach()

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    
    def forward(self, predict, target):
        target = self._convert_target(target)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]
    
    def _convert_target(self, target):
        device = target.device
        target = make_one_hot(target.unsqueeze(1), 9)
        target = target.to(device)
        return target

class PGD(Attack):
    def __init__(self, iters, alpha, eps, norm, rand_init, targeted=False, normalize=normalize_voc):
        # super().__init__(iters, alpha, eps, norm, DiceLoss(ignore_index=255), rand_init=rand_init, rand_perturb=False, targeted=targeted, normalize=normalize)
        super().__init__(iters, alpha, eps, norm, nn.CrossEntropyLoss(ignore_index=255, reduction='mean'), rand_init=rand_init, rand_perturb=False, targeted=targeted, normalize=normalize)