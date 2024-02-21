from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import time
import sys

import pandas as pd
from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Iccv2009Dataset
from utils import ext_transforms as et
from utils import PGD, normalize_voc, SAM, normalize_city, normalize_iccv09
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from utils import Logger
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

import wandb

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='voc',
                        choices=['voc', 'cityscapes', 'iccv09'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=30e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=1,)
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--gpu_id", type=str, default='2',
                        help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Experiment name setting
    parser.add_argument("--exp_name", type=str, default='exp',)
    
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'voc':
        train_transform = et.ExtCompose([
            # et.ExtResize(size=opts.crop_size),
            et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        train_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                    image_set='train', download=opts.download, transform=train_transform)
        val_dst = VOCSegmentation(root=opts.data_root, year=opts.year,
                                  image_set='val', download=False, transform=val_transform)

    if opts.dataset == 'cityscapes':
        train_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        val_transform = et.ExtCompose([
            # et.ExtResize( 512 ),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        train_dst = Cityscapes(root=opts.data_root,
                               split='train', transform=train_transform)
        val_dst = Cityscapes(root=opts.data_root,
                             split='val', transform=val_transform)
        
    if opts.dataset == 'iccv09':
        train_transform = et.ExtCompose([
            et.ExtResize(256),
            et.ExtRandomCrop(size=(240, 240)),
            et.ExtColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.4813, 0.4901, 0.4747],
                            std=[0.2495, 0.2492, 0.2748]),
        ])
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.4813, 0.4901, 0.4747],
                            std=[0.2495, 0.2492, 0.2748]),
        ])
        
        train_dst = Iccv2009Dataset(root='datasets/data/iccv09', split='train', transform=train_transform)
        val_dst = Iccv2009Dataset(root='datasets/data/iccv09', split='val', transform=val_transform)
    
    return train_dst, val_dst


def robustness_validate(opts, model, loader, device, metrics, attack, eps, norm, normalize, ret_samples_ids=None):
    """adversarial samples generation"""
    if attack is None:
        iters = -1
    elif attack == 'FGSM':
        iters = 1
    elif attack == 'PGD':
        iters = 10
    else:
        raise NotImplementedError("Only support FGSM and PGD attack.")
    alpha = eps / 4.
    train_rand_init = False if attack == 'FGSM' else True
    generator = PGD(
        iters=iters,
        alpha=alpha,
        eps=eps,
        norm=norm,                  # linf, l2
        rand_init=train_rand_init,
        targeted=False,
        normalize=normalize
    )
    
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    for i, (images, labels) in tqdm(enumerate(loader)):

        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)
        if iters > 0:
            delta = generator.perturb(model, images, labels)
            images = normalize(images + delta)
            # images += delta
        else:
            delta = torch.zeros_like(images, device=device)
            images = normalize(images+delta)
            # images += delta

        outputs = model(images)
        preds = outputs.detach().max(dim=1)[1].cpu().numpy()
        targets = labels.cpu().numpy()

        metrics.update(targets, preds)
        if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
            ret_samples.append(
                (images[0].detach().cpu().numpy(), targets[0], preds[0]))

    score = metrics.get_results()
    return score, ret_samples
    
    
if __name__ == "__main__":
    sys.stdout = Logger('voc_robustness_test_sam.txt')
    
    model_path_dict = {
    }
    
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'iccv09':
        opts.num_classes = 9
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1
    if opts.dataset == 'iccv09':
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=16)
    
    if opts.dataset == 'voc':
        normalize = normalize_voc
    elif opts.dataset == 'cityscapes':
        normalize = normalize_city
    elif opts.dataset == 'iccv09':
        normalize = normalize_iccv09
    
        
    results_dataframes = {}
    
    for name, state_path in model_path_dict.items():
        results = []
        print("\nModel: %s" % name)
        
        state = torch.load(state_path, map_location=torch.device('cpu'))
        model_state = state['model_state']
        new_state = {}
        for k, v in model_state.items():
            k = k.replace('module.', '')
            new_state[k] = v
        model.load_state_dict(new_state)
        # model = nn.DataParallel(model)
        model.to(device)
        model.eval()
        
        
        # Natural metrics
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack=None,
            eps=0,
            norm='l2',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("#####  Natural metrics  #####: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])
        
        """# FGSM robustness linf eps=1/255
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='FGSM',
            eps=1/255,
            norm='linf',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("FGSM robustness linf eps=1/255: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])

        # FGSM robustness linf eps=2/255
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='FGSM',
            eps=2/255,
            norm='linf',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("FGSM robustness linf eps=2/255: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])

        # FGSM robustness linf eps=4/255
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='FGSM',
            eps=4/255,
            norm='linf',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("FGSM robustness linf eps=4/255: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])
        
        # FGSM robustness l2 eps=0.5
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='FGSM',
            eps=0.5,
            norm='l2',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("FGSM robustness l2 eps=0.5: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])
        
        # FGSM robustness l2 eps=1
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='FGSM',
            eps=1,
            norm='l2',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("FGSM robustness l2 eps=1: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])

        # FGSM robustness l2 eps=2
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='FGSM',
            eps=2,
            norm='l2',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("FGSM robustness l2 eps=2: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])"""
        
        # PGD robustness linf eps=1/255
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='PGD',
            eps=1/255,
            norm='linf',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("PGD robustness linf eps=1/255: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])
        
        # PGD robustness linf eps=2/255
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='PGD',
            eps=2/255,
            norm='linf',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("PGD robustness linf eps=2/255: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])

        """# PGD robustness linf eps=4/255
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='PGD',
            eps=4/255,
            norm='linf',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("PGD robustness linf eps=4/255: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])"""
        
        
        # PGD robustness l2 eps=0.5
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='PGD',
            eps=0.5,
            norm='l2',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("PGD robustness l2 eps=0.5: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])
        
        # PGD robustness l2 eps=1
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='PGD',
            eps=1,
            norm='l2',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("PGD robustness l2 eps=1: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])

        """# PGD robustness l2 eps=2
        val_score, ret_samples = robustness_validate(
            opts=opts,
            model=model,
            loader=val_loader,
            device=device,
            metrics=StreamSegMetrics(opts.num_classes),
            attack='PGD',
            eps=2,
            norm='l2',
            normalize=normalize,
            ret_samples_ids=None
        )
        print("PGD robustness l2 eps=2: %.4f" % val_score['Mean IoU'])
        results.append(val_score['Mean IoU'])"""
    
        results_dataframes[name] = results
        
    #dataframe = pd.DataFrame(data=results_dataframes, index=["natural", "FGSM-linf-1/255", "FGSM-linf-2/255", "FGSM-linf-4/255", "FGSM-l2-2", "FGSM-l2-4", "FGSM-l2-8", "PGD-linf-1/255", "PGD-linf-2/255", "PGD-linf-4/255", "PGD-l2-2", "PGD-l2-4", "PGD-l2-8"])
    #dataframe = dataframe.T
    #dataframe.to_csv("iccv09-sam.csv", sep=',', header=True, index=True)
    