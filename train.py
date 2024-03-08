import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import AverageMeter, accuracy, ProgressMeter, get_default_ImageNet_val_loader, \
    get_default_ImageNet_train_sampler_loader, log_msg, MetricLogger, is_main_process, setup_for_distributed
import utils
from pathlib import Path
from datasets import build_dataset
from repvgg import get_RepVGG_func_by_name
from timm.models import create_model
from timm.optim import create_optimizer


IMAGENET_TRAINSET_SIZE = 1281167


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                    type=str, help='Image Net dataset path')
parser.add_argument('--inat-category', default='name',
                    choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                    type=str, help='semantic granularity')

parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=100, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--custwd', dest='custwd', action='store_true',
                    help='Use custom weight decay. It improves the accuracy and makes quantization easier.')
parser.add_argument('--tag', default='testtest', type=str,
                    help='the tag for identifying the log and model files. Just a string.')
parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--deploy', action='store_true')
parser.add_argument('--no-deploy', action='store_false', dest='deploy')
parser.set_defaults(deploy=False)

best_acc1 = 0


def sgd_optimizer(model, lr, momentum, weight_decay, use_custwd):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_weight_decay = weight_decay
        apply_lr = lr
        if (use_custwd and ('rbr_dense' in key or 'rbr_1x1' in key )) or 'bias' in key or 'bn' in key or 'scale' in key:
            apply_weight_decay = 0
            print('set weight decay=0 for {}'.format(key))
        if 'bias' in key:
            apply_lr = 2 * lr       #   Just a Caffe-style common practice. Made no difference.
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_weight_decay}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    log_file = 'train_{}_{}_exp.txt'.format(args.arch, args.tag)
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)

    if 'Rep' in args.arch:
        repvgg_build_func = get_RepVGG_func_by_name(args.arch)
        model = repvgg_build_func(deploy=args.deploy)
    else:
        model = create_model(
            args.arch
        )

    if not args.output_dir:
        args.output_dir = args.arch+args.tag
        if utils.is_main_process():
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)

    output_dir = Path(args.output_dir)

    is_main = not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0)

    # if is_main:
    #     for n, p in model.named_parameters():
    #         print(n, p.size())
    #     for n, p in model.named_buffers():
    #         print(n, p.size())
    #     log_msg('epochs {}, lr {}, weight_decay {}'.format(args.epochs, args.lr, args.weight_decay), log_file)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    device = torch.device(args.gpu)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if 'REP' in args.arch:
        optimizer = sgd_optimizer(model, args.lr, args.momentum, args.weight_decay, args.custwd)
    else:
        optimizer = create_optimizer(args, model)

    lr_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.epochs * IMAGENET_TRAINSET_SIZE // args.batch_size // ngpus_per_node)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    args.data_path = args.data

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = args.world_size #utils.get_world_size()
        global_rank = args.rank #utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    train_sampler = sampler_train
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False
    )


    # train_sampler, train_loader = get_default_ImageNet_train_sampler_loader(args)
    train_loader = data_loader_train #data_prefetcher(train_loader)
    # val_loader = get_default_ImageNet_val_loader(args)
    val_loader = data_loader_val  #data_prefetcher(val_loader)
    from torch.cuda.amp import GradScaler
    loss_scaler = GradScaler()

    max_accuracy = 0.0

    if args.evaluate:
        test_stats = evaluate(val_loader, model, device)
        return
    from utils import get_scale, set_scale

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        if 'scale' in args.tag:
            print('epoch {} scale {}'.format(epoch, get_scale()))
            scale = 1 - epoch / (args.epochs-1)
            set_scale(scale)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, lr_scheduler, is_main=is_main, loss_scaler=loss_scaler)
        test_stats = evaluate(val_loader, model, device)
        print(f"Accuracy of the network on the {len(val_loader)} test images: {test_stats['acc1']:.1f}%")
        if max_accuracy < test_stats['acc1']:
            if is_main_process():
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                }, True, filename=args.output_dir + os.sep + '{}_{}.pth.tar'.format(args.arch, args.tag),
                    best_filename=args.output_dir + os.sep + '{}_{}_best.pth.tar'.format(args.arch, args.tag))

        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets)

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class data_prefetcher(object):
    def __init__(self, loader):
        self._loader = loader
        self.loader = iter(self._loader)
        self.stream = torch.cuda.Stream()
        self.preload()
        self.restart = False

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        self.next_batch = list(self.next_batch)
        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_batch)):
                self.next_batch[i] = self.next_batch[i].cuda(non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            if self.restart:
                self.loader = iter(self._loader)
                self.preload()
                self.restart = False
            else:
                self.restart = True
                raise  StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        next_batch = self.next_batch
        # if next_batch is not None:
        #     next_batch.record_stream(torch.cuda.current_stream())
        self.preload()
        return next_batch

    def __len__(self):
        return len(self.loader)


def train(train_loader, model, criterion, optimizer, epoch, args, lr_scheduler, is_main, loss_scaler=None):
    import utils
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # switch to train mode
    model.train()
    device = torch.device(args.device)

    for images, targets in metric_logger.log_every(train_loader, print_freq, header):
        # images = images.cuda(args.gpu, non_blocking=True)
        # targets = targets.cuda(args.gpu, non_blocking=True)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, targets)
            if args.custwd:
                for module in model.modules():
                    if hasattr(module, 'get_custom_L2'):
                        loss += args.weight_decay * 0.5 * module.get_custom_L2()

        loss_value = loss.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss_scaler.scale(loss).backward()
        loss_scaler.step(optimizer)
        loss_scaler.update()

        lr_scheduler.step()
        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_checkpoint(state, is_best, filename, best_filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_filename)


if __name__ == '__main__':
    main()
