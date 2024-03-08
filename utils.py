import torch
import math
import torchvision.datasets as datasets
import os
import torchvision.transforms as transforms
import PIL
import time
from collections import defaultdict, deque
import datetime
import torch.distributed as dist
import io
import math
import hashlib
import subprocess


class MyScaler(object):
    def __init__(self, max_iter=1, cosine=False):
        self.max_iter = max_iter
        self.iter = 0
        self.cosine = cosine

    def get_scale(self):
        if self.cosine:
            if self.iter <= self.max_iter:
                return (1 + math.cos(math.pi * self.iter / self.max_iter)) / 2
            else:
                return 0.0
        else:
            return max(0.0, 1.0 - self.iter/self.max_iter)

    def step(self):
        self.iter += 1

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def set_iter(self, iter):
        self.iter = iter


my_scaler = MyScaler()


def special_arch(args):
    if 'off' in args.tag:
        if 'Rep' in args.model or 'gre' in args.model or 'resnet' in args.model:
            return True
        else:
            return False
    else:
        return False



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {}
    for k, v in checkpoint.items():
        if k.startswith('module.'):
            ckpt[k[7:]] = v
        else:
            ckpt[k] = v
    model.load_state_dict(ckpt)

def read_hdf5(file_path):
    import h5py
    import numpy as np
    result = {}
    with h5py.File(file_path, 'r') as f:
        for k in f.keys():
            value = np.asarray(f[k])
            result[str(k).replace('+', '/')] = value
    print('read {} arrays from {}'.format(len(result), file_path))
    f.close()
    return result

def model_load_hdf5(model:torch.nn.Module, hdf5_path, ignore_keys='stage0.'):
    weights_dict = read_hdf5(hdf5_path)
    for name, param in model.named_parameters():
        print('load param: ', name, param.size())
        if name in weights_dict:
            np_value = weights_dict[name]
        else:
            np_value = weights_dict[name.replace(ignore_keys, '')]
        value = torch.from_numpy(np_value).float()
        assert tuple(value.size()) == tuple(param.size())
        param.data = value
    for name, param in model.named_buffers():
        print('load buffer: ', name, param.size())
        if name in weights_dict:
            np_value = weights_dict[name]
        else:
            np_value = weights_dict[name.replace(ignore_keys, '')]
        value = torch.from_numpy(np_value).float()
        assert tuple(value.size()) == tuple(param.size())
        param.data = value



class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, T_cosine_max, eta_min=0, last_epoch=-1, warmup=0):
        self.eta_min = eta_min
        self.T_cosine_max = T_cosine_max
        self.warmup = warmup
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            return [self.last_epoch / self.warmup * base_lr for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch - self.warmup) / (self.T_cosine_max - self.warmup))) / 2
                    for base_lr in self.base_lrs]


def log_msg(message, log_file):
    print(message)
    with open(log_file, 'a') as f:
        print(message, file=f)


def get_ImageNet_train_dataset(args, trans):
    if os.path.exists('/home/dingxiaohan/ndp/imagenet.train.nori.list'):
        #   This is the data source on our machine. You won't need it.
        from noris_dataset import ImageNetNoriDataset
        train_dataset = ImageNetNoriDataset('/home/dingxiaohan/ndp/imagenet.train.nori.list', trans)
    else:
        #   Your ImageNet directory
        traindir = os.path.join(args.data, 'train')
        train_dataset = datasets.ImageFolder(traindir, trans)
    return train_dataset


def get_ImageNet_val_dataset(args, trans):
    if os.path.exists('/home/dingxiaohan/ndp/imagenet.val.nori.list'):
        #   This is the data source on our machine. You won't need it.
        from noris_dataset import ImageNetNoriDataset
        val_dataset = ImageNetNoriDataset('/home/dingxiaohan/ndp/imagenet.val.nori.list', trans)
    else:
        #   Your ImageNet directory
        traindir = os.path.join(args.data, 'val')
        val_dataset = datasets.ImageFolder(traindir, trans)
    return val_dataset


def get_default_train_trans(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    else:
        raise ValueError('Not yet implemented.')
    return trans


def get_default_val_trans(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if (not hasattr(args, 'resolution')) or args.resolution == 224:
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
    else:
        trans = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=PIL.Image.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize,
        ])
    return trans


def get_default_ImageNet_train_sampler_loader(args):
    train_trans = get_default_train_trans(args)
    train_dataset = get_ImageNet_train_dataset(args, train_trans)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    return train_sampler, train_loader


def get_default_ImageNet_val_loader(args):
    val_trans = get_default_val_trans(args)
    val_dataset = get_ImageNet_val_dataset(args, val_trans)
    if hasattr(args, 'val_batch_size'):
        bs = args.val_batch_size
    else:
        bs = args.batch_size

    num_tasks = args.world_size
    global_rank = get_rank()

    sampler_val = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, sampler=sampler_val,
        batch_size=int(1.5*bs), shuffle=False,
        num_workers=args.workers, pin_memory=True)
    return val_loader


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def get_git_hash():
    return subprocess.check_output(['git', 'log', '-n', '1', '--pretty=tformat:%H']).strip()