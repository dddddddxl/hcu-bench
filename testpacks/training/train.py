"""HCU Bench's own single-device ImageFolder classification training entry point.

Uses installed Torch/TorchVision public APIs; no external script, Apex or weight download.
The external modelzoo program is not copied into this implementation.
"""
import argparse
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recipes import RECIPES, model


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data')
    p.add_argument('--arch', choices=sorted({r['architecture'] for r in RECIPES.values()}), default='resnet50')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--print-freq', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--save-path', required=True)
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--resume')
    return p


def epoch(loader, network, criterion, device, index, optimizer=None, print_freq=1):
    import torch
    training = optimizer is not None
    network.train(training)
    loss_sum = correct1 = correct5 = samples = 0
    with torch.set_grad_enabled(training):
        for step, (images, labels) in enumerate(loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = network(images)
            loss = criterion(logits, labels)
            value = loss.item()
            if not math.isfinite(value):
                raise ValueError('Nonfinite training/validation loss')
            if training:
                loss.backward()
                optimizer.step()
            count = labels.size(0)
            prediction = logits.detach().topk(5, dim=1).indices
            matches = prediction.eq(labels[:, None])
            correct1 += matches[:, :1].sum().item()
            correct5 += matches.sum().item()
            samples += count
            loss_sum += value * count
            if step % print_freq == 0 or step + 1 == len(loader):
                prefix = 'Epoch: [{}][{}/{}]'.format(index, step, len(loader)) if training else 'Test: [{}/{}]'.format(step, len(loader))
                print('{} Loss {:.4e} ({:.4e})'.format(prefix, value, loss_sum / samples), flush=True)
    if not samples:
        raise ValueError('Empty ImageFolder loader')
    result = dict(top1=100 * correct1 / samples, top5=100 * correct5 / samples, loss=loss_sum / samples)
    if not training:
        print(' * Acc@1 {:.3f} Acc@5 {:.3f}'.format(result['top1'], result['top5']), flush=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.gpu != 0 or args.world_size != 1 or args.rank != 0 or args.epochs < 1
            or args.batch_size < 2 or args.workers < 0 or args.print_freq < 1):
        raise ValueError('Invalid single-device training arguments')
    import torch
    from torchvision import datasets, transforms
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one assigned HCU must be visible')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    network = model(args.arch).to(device)
    optimizer = torch.optim.SGD(network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    augment = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    validation = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    train_data = datasets.ImageFolder(str(Path(args.data) / 'train'), augment)
    val_data = datasets.ImageFolder(str(Path(args.data) / 'val'), validation)
    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError('Training and validation class indices differ')
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True)
        if checkpoint['arch'] != args.arch:
            raise ValueError('Checkpoint architecture mismatch')
        network.load_state_dict(checkpoint['state_dict'], strict=True)
    if args.evaluate:
        if not args.resume:
            raise ValueError('Evaluation requires this run checkpoint')
        epoch(val_loader, network, criterion, device, 0, print_freq=args.print_freq)
        return
    if args.resume:
        raise ValueError('Training resume is not implemented; evaluation reload only')
    target = Path(args.save_path) / '0'
    target.mkdir(parents=True, exist_ok=True)
    print('HCU Bench builtin training: {} / FP32 / no pretrained weights'.format(args.arch), flush=True)
    for index in range(args.epochs):
        for group in optimizer.param_groups:
            group['lr'] = args.lr * (0.1 ** (index // 30))
        epoch(train_loader, network, criterion, device, index, optimizer, args.print_freq)
        result = epoch(val_loader, network, criterion, device, index, print_freq=args.print_freq)
        torch.save(dict(epoch=index + 1, arch=args.arch, state_dict=network.state_dict(),
                        optimizer=optimizer.state_dict(), best_acc1=result['top1'], backend='hcu_bench_builtin'),
                   target / ('checkpoint_{}.pth.tar'.format(index)))


if __name__ == '__main__':
    main()
