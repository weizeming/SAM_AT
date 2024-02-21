from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
import time
import sys

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, Iccv2009Dataset
from utils import ext_transforms as et
from utils import PGD, normalize_voc, SAM, normalize_city, normalize_iccv09
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

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
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
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
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam', 'SAM'],
                        help="optimizer (default: sgd)")

    # PASCAL VOC Options
    parser.add_argument("--year", type=str, default='2012',
                        choices=['2012_aug', '2012', '2011', '2009', '2008', '2007'], help='year of VOC')

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')

    # Experiment name setting
    parser.add_argument("--exp_name", type=str, default='exp',)

    # Adversarial training
    parser.add_argument("--adv", action='store_true', default=False,)
    parser.add_argument("--train_step", type=int, default=1,)
    parser.add_argument("--train_eps", type=float, default=64.)
    parser.add_argument("--train_alpha", type=float, default=64.)
    parser.add_argument("--train_rand_init", action='store_true', default=False,)
    parser.add_argument("--norm", type=str, default='linf', choices=['linf', 'l2'])

    # SAM
    parser.add_argument("--rho", type=float, default=0.1,)
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


def validate(opts, model, loader, device, metrics, normalize, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            images = normalize(images)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    if os.path.exists(os.path.join('checkpoints', opts.exp_name)):
        # ask if overwrite
        print("Experiment name already exists. Do you want to overwrite? (y/n)")
        while True:
            ans = input()
            if ans == 'y':
                break
            elif ans == 'n':
                print("Please run again with a different experiment name.")
                exit()
            else:
                print("Please enter y or n.")
    utils.mkdir(os.path.join('checkpoints', opts.exp_name))
    # let all the print goes to log file while still print on screen
    sys.stdout = utils.Logger(os.path.join('checkpoints', opts.exp_name, 'log.txt'))
    print('Experiment starts at %s' % time.asctime(time.localtime(time.time())))
    print(opts.exp_name, opts.model, opts.dataset)


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
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))


    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    if opts.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        backbone_optimizer = None
    # Adam
    elif opts.optimizer == 'adam':
        print('Using Adam optimizer')
        optimizer = torch.optim.Adam(params=[
            {'params': model.backbone.parameters(), 'lr':  0.05 * opts.lr},
            {'params': model.classifier.parameters(), 'lr': 0.05 * opts.lr},
        ], lr=opts.lr)
        backbone_optimizer = None
    elif opts.optimizer == 'SAM':
        print('Using SAM optimizer, rho={}'.format(opts.rho))
        optimizer = SAM(params=[
            {'params': model.backbone.parameters(), 'lr':  opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], base_optimizer=torch.optim.SGD, lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay, rho=opts.rho)
        backbone_optimizer = torch.optim.SGD(params=[
            {'params': model.backbone.parameters(), 'lr': opts.lr},
            {'params': model.classifier.parameters(), 'lr': opts.lr},
        ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
        
    """if opts.dataset == 'voc' and opts.optimizer == 'SAM':
        state_path = 'checkpoints/voc_sgd_2/best_deeplabv3_mobilenet_voc_os16.pth'
        checkpoint = torch.load(state_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        backbone_optimizer.load_state_dict(checkpoint["optimizer_state"])
        print("Model restored from %s" % state_path)
        del checkpoint  # free memory"""

    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)


    # Adversarial training
    if opts.dataset == 'voc':
        normalize = normalize_voc 
    elif opts.dataset == 'cityscapes':
        normalize = normalize_city
    elif opts.dataset == 'iccv09':
        normalize = normalize_iccv09
        
    train_pgd = PGD(opts.train_step, opts.train_eps / 255., opts.train_eps / 255., opts.norm, opts.train_rand_init, False, normalize)
    if opts.adv:
        print('Using adversarial training, alpha={}, eps={}, step={}, norm={}'.format(opts.train_alpha, opts.train_eps, opts.train_step, opts.norm))

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    
    # save opts
    with open(os.path.join('checkpoints', opts.exp_name, 'opts.txt'), 'w') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(vars(opts).items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)


    wandb.init(project="SAM-AT-seg", name=opts.exp_name, 
               config={
                   'model': opts.model,
                   'dataset': opts.dataset,
                   'optimizer': opts.optimizer,
                   'lr': opts.lr,
                   'batch_size': opts.batch_size,
                   'crop_size': opts.crop_size,
                   'total_itrs': opts.total_itrs,
                   'val_interval': opts.val_interval,
                   'adv': opts.adv,
                   'train_step': opts.train_step,
                   'train_eps': opts.train_eps,
                   'train_alpha': opts.train_alpha,
                   'train_rand_init': opts.train_rand_init,
                   'norm': opts.norm,
                   'rho': opts.rho,
                   'optimizer': opts.optimizer,
               })
    
    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id, normalize=normalize)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    
    # test if image in in 0-1 range
    images, labels = next(iter(train_loader))
    assert images.max() <= 1.0 and images.min() >= 0.0, "Images should be in 0-1 range but max={}, min={}".format(images.max(), images.min())
    
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for (images, labels) in train_loader:
            
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            if opts.adv:
                delta = train_pgd.perturb(model, images, labels)
            else:
                delta = torch.zeros_like(images, device=images.device)

            images = normalize(images + delta)
            outputs = model(images)
            loss = criterion(outputs, labels)


            if opts.optimizer == "sgd" or opts.optimizer == "adam":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            elif opts.optimizer == "SAM":
                if cur_itrs <= 10e3:
                    backbone_optimizer.zero_grad()
                    loss.backward()
                    backbone_optimizer.step()
                else:
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    criterion(model(images), labels).backward()
                    optimizer.second_step(zero_grad=True)

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0
                wandb.log({'train/loss': np_loss}, step=cur_itrs)

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/%s/latest_%s_%s_os%d.pth' % (opts.exp_name, opts.model, opts.dataset, opts.output_stride))
                # save_ckpt('checkpoints/latest_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, normalize=normalize,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/%s/best_%s_%s_os%d.pth' % (opts.exp_name, opts.model, opts.dataset, opts.output_stride))
                    # save_ckpt('checkpoints/best_%s_%s_os%d.pth' % (opts.model, opts.dataset, opts.output_stride))

                wandb.log({'val/mean_iou': val_score['Mean IoU']}, step=cur_itrs)
                model.train()
            if opts.optimizer == "adam":
                pass
            elif cur_itrs > 1:
                scheduler.step()

            if cur_itrs >= opts.total_itrs:
                wandb.finish()
                return
    

if __name__ == '__main__':
    main()
