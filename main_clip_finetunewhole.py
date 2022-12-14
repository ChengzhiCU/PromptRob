from __future__ import print_function

import argparse
import os
from tqdm import tqdm
import time
import random
# import wandb

import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, Caltech101, STL10, StanfordCars, Food101, SUN397
from torchvision.datasets import CIFAR100, CIFAR10, OxfordIIITPet, Flowers102, Country211, Caltech256

import torchvision.transforms as transforms
import torchvision

import clip
from models import prompters
from models.prompters import TokenPrompter, NullPrompter
from utils import CLASS_SPLIT_CIFAR100, accuracy, AverageMeter, ProgressMeter, save_checkpoint
from utils import cosine_lr, convert_models_to_fp32, refine_classname

import torch.nn.functional as F
import numpy as np
import torch.nn as nn


def parse_option():
    parser = argparse.ArgumentParser('Visual Prompting for CLIP')

    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--validate_freq', type=int, default=1,
                        help='validate frequency')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epoch5s')

    # optimization
    parser.add_argument('--optim', type=str, default='sgd',
                        help='optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.1,  ## Why so large
                        help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--warmup", type=int, default=1000,
                        help="number of steps to warmup for")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--train_eps', type=float, default=2,
                        help='momentum')
    parser.add_argument('--train_numsteps', type=int, default=5)
    parser.add_argument('--train_stepsize', type=int, default=1)
    parser.add_argument('--test_eps', type=float, default=2,
                        help='momentum')
    parser.add_argument('--test_numsteps', type=int, default=5)
    parser.add_argument('--test_stepsize', type=int, default=1)
    parser.add_argument('--patience', type=int, default=1000)

    # model
    parser.add_argument('--model', type=str, default='clip')
    parser.add_argument('--arch', type=str, default='vit_b32')
    parser.add_argument('--method', type=str, default='padding',
                        choices=['padding', 'random_patch', 'fixed_patch'],
                        help='choose visual prompting method')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--prompt_size', type=int, default=30,
                        help='size for visual prompts')
    parser.add_argument('--add_prompt_size', type=int, default=10,
                        help='size for additional visual prompts')

    # dataset
    parser.add_argument('--root', type=str, default='./data',
                        help='dataset')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='dataset')
    parser.add_argument('--image_size', type=int, default=224,
                        help='image size')

    # other
    parser.add_argument('--seed', type=int, default=0,
                        help='seed for initializing training')
    parser.add_argument('--model_dir', type=str, default='./save/models',
                        help='path to save models')
    parser.add_argument('--image_dir', type=str, default='./save/images',
                        help='path to save images')
    parser.add_argument('--filename', type=str, default=None,
                        help='filename to save')
    parser.add_argument('--trial', type=int, default=1,
                        help='number of trials')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to resume from checkpoint')
    parser.add_argument('--evaluate', default=False,
                        action="store_true",
                        help='evaluate model test set')
    parser.add_argument('--gpu', type=int, default=None,
                        help='gpu to use')
    parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--use_wandb', default=False,
    #                     action="store_true",
    #                     help='whether to use wandb')

    args = parser.parse_args()

    args.filename = '{}_{}_{}_{}_{}_{}_{}_lr_{}_decay_{}_bsz_{}_warmup_{}_trial_{}_addp_{}'. \
        format(args.name, args.method, args.prompt_size, args.dataset, args.model, args.arch,
               args.optim, args.learning_rate, args.weight_decay, args.batch_size, args.warmup, args.trial,
               args.add_prompt_size)

    return args


best_acc1 = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR100_MEAN = (0.48145466, 0.4578275, 0.40821073)
CIFAR100_STD = (0.26862954, 0.26130258, 0.27577711)

mu = torch.tensor(CIFAR100_MEAN).view(3, 1, 1).cuda()
std = torch.tensor(CIFAR100_STD).view(3, 1, 1).cuda()


def normalize(X):
    return (X - mu) / std


def clip_img_preprocessing(X):
    img_size = 224
    X = torch.nn.functional.upsample(X, size=(img_size, img_size), mode='bicubic')
    X = normalize(X)

    return X


# for multiGPU clip
def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t()
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image, prompt_token=None):
        return self.model.encode_image(image, prompt_token)


# alpha_test = 1. / 255
# attack_iters_test = 5

epsilon = 2. / 255
upper_limit, lower_limit = 1, 0


def main():
    global best_acc1, device

    args = parse_option()
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    imagenet_root = '/local/vondrick/datasets/ImageNet-clean'

    # create model
    add_prompt_len = args.add_prompt_size

    model, preprocess = clip.load('ViT-B/32', device, jit=False, prompt_len=add_prompt_len)
    convert_models_to_fp32(model)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text)
    model_image = torch.nn.DataParallel(model_image)

    convert_models_to_fp32(model_text)
    convert_models_to_fp32(model_image)
    # model_text, model_image = None, None

    # model = torch.nn.parallel.DistributedDataParallel(model).cuda()

    # model = torch.nn.DataParallel(model) #.to(device)
    model.eval()
    model_text.train()
    model_image.eval()

    # prompter = prompters.__dict__[args.method](args)  # .to(device)
    # add_prompter = TokenPrompter(add_prompt_len)  # .to(device)
    #
    # prompter = torch.nn.DataParallel(prompter).cuda()
    # add_prompter = torch.nn.DataParallel(add_prompter).cuda()



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
            # prompter.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # create data
    template = 'This is a photo of a {}'
    print(f'template: {template}')

    # print(preprocess, 'preprocess')
    # exit()

    # TODO: we can train on cifar10 and test on cifar10, 100 in zero shot way, to see if generalize.
    preprocess = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])
    preprocess224 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), # TODO: may use later
        transforms.ToTensor()
    ])

    if args.dataset == 'cifar100':
        train_dataset = CIFAR100(args.root, transform=preprocess,
                                 download=True, train=True)

        val_dataset = CIFAR100(args.root, transform=preprocess,
                               download=True, train=False)
    elif args.dataset == 'cifar10':
        train_dataset = CIFAR10(args.root, transform=preprocess,
                                download=True, train=True)

        val_dataset = CIFAR10(args.root, transform=preprocess,
                              download=True, train=False)

    elif args.dataset == 'ImageNet':
        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(imagenet_root, 'train'),
            transform=preprocess224
        )

    elif args.dataset == 'zeroshot_cifar100':
        train_class_count = 50
        train_dataset = CLASS_SPLIT_CIFAR100(args.root, transform=preprocess,
                                             download=True, train=True, train_class_count=train_class_count)

        val_dataset = CLASS_SPLIT_CIFAR100(args.root, transform=preprocess,
                                           download=True, train=False, train_class_count=train_class_count)

        # train_dataset = torchvision.datasets.ImageFolder(
        #     root=imagenet_root,
        #     split='train',
        #     transform=transforms.Compose([
        #         transforms.RandomResizedCrop(224),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #     ]))

    val_dataset_list = []
    # val_dataset_name=['cifar10', 'cifar100', 'LSUN', 'STL-10', 'StanfordCars', 'Food101', 'SUN397', ]
    # val_dataset_name = ['cifar10', 'cifar100', 'STL10', 'StanfordCars', 'Food101', 'SUN397']
    val_dataset_name = ['cifar10', 'cifar100', 'STL-10', 'SUN397']
    val_dataset_name = ['SUN397', 'StanfordCars', 'Food101']
    val_dataset_name = ['ImageNet', 'cifar10', 'cifar100', 'STL10', 'StanfordCars', 'SUN397', 'Food101', 'oxfordpet',
                        'FER2013', 'flowers102', 'Country211', 'Caltech256']
    val_dataset_name = ['ImageNet', 'cifar10', 'cifar100', 'STL10', 'StanfordCars', 'SUN397', 'Food101', 'oxfordpet']
    # val_dataset_name = ['zeroshot_cifar100', 'cifar10']

    val_dataset_name = ['cifar10', 'cifar100', 'StanfordCars', 'SUN397', 'Food101', 'oxfordpet']
    val_dataset_name = ['cifar10', 'cifar100']

    for each in val_dataset_name:
        if each == 'cifar10':
            val_dataset_list.append(CIFAR10(args.root, transform=preprocess,
                                            download=True, train=False))
        elif each == 'cifar100':
            val_dataset_list.append(CIFAR100(args.root, transform=preprocess,
                                             download=True, train=False))
        elif each == 'zeroshot_cifar100':
            assert args.dataset == 'zeroshot_cifar100'
            val_dataset_list.append(CLASS_SPLIT_CIFAR100(args.root, transform=preprocess,
                                                         download=True, train=False,
                                                         train_class_count=train_class_count))
        elif each == 'Caltech256':
            val_dataset_list.append(Caltech256(args.root, transform=preprocess224,
                                               download=True))
        elif each == 'STL10':
            val_dataset_list.append(STL10(args.root, split='test',
                                          transform=preprocess, download=True))
        elif each == 'SUN397':
            val_dataset_list.append(SUN397(args.root,
                                           transform=preprocess224, download=True))
        elif each == 'StanfordCars':
            val_dataset_list.append(StanfordCars(args.root, split='test',
                                                 transform=preprocess224, download=True))
        elif each == 'Food101':
            val_dataset_list.append(Food101(args.root, split='test',
                                            transform=preprocess224, download=True))
        elif each == 'oxfordpet':
            val_dataset_list.append(OxfordIIITPet(args.root, split='test',
                                                  transform=preprocess224, download=True))
        # elif each == 'FER2013':
        #     val_dataset_list.append(OxfordIIITPet(args.root, split='test',
        #                                    transform=preprocess224, download=True))
        # elif each == 'flowers102':
        #     val_dataset_list.append(Flowers102(args.root, split='test',
        #                                    transform=preprocess224, download=True))
        # elif each == 'Country211':
        #     val_dataset_list.append(Country211(args.root, split='test',
        #                                    transform=preprocess224, download=True))
        elif each == 'ImageNet':
            val_dataset_list.append(torchvision.datasets.ImageFolder(
                os.path.join(imagenet_root, 'val'),
                transform=preprocess224))

            # val_dataset_list.append(torchvision.datasets.ImageNet(
            # root=imagenet_root,
            # split='val',
            # transform=preprocess224))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    #     val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    # else:
    train_sampler = None
    val_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size, pin_memory=True,
                              num_workers=args.num_workers, shuffle=True, sampler=train_sampler)

    val_loader_list = [DataLoader(each,
                                  batch_size=args.batch_size, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False, sampler=val_sampler) for each in
                       val_dataset_list]

    class_names = train_dataset.classes

    if args.dataset == 'ImageNet':
        from utils import load_imagenet_folder2name
        folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
        new_class_names = []
        for each in class_names:
            new_class_names.append(folder2name[each])

        class_names = new_class_names

    class_names = refine_classname(class_names)
    texts_train = [template.format(label) for label in class_names]

    prompter = NullPrompter()
    add_prompter = TokenPrompter(add_prompt_len)  # .to(device)

    prompter = torch.nn.DataParallel(prompter).cuda()
    add_prompter = torch.nn.DataParallel(add_prompter).cuda()

    texts_list = []
    for cnt, each in enumerate(val_dataset_list):
        class_names = each.classes
        if val_dataset_name[cnt] == 'ImageNet':
            from utils import load_imagenet_folder2name
            folder2name = load_imagenet_folder2name('imagenet_classes_names.txt')
            new_class_names = []
            for each in class_names:
                new_class_names.append(folder2name[each])
            class_names = new_class_names

        class_names = refine_classname(class_names)
        texts_tmp = [template.format(label) for label in class_names]
        texts_list.append(texts_tmp)

    # define criterion and optimizer
    optimizer = torch.optim.SGD(model_image.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup, total_steps)

    cudnn.benchmark = True

    # make dir
    refined_template = template.lower().replace(' ', '_')
    args.filename = f'{args.filename}_template_{refined_template}'

    args.model_folder = os.path.join(args.model_dir, args.filename)
    if not os.path.isdir(args.model_folder):
        os.makedirs(args.model_folder)

    # wandb
    # if args.use_wandb:
    #     wandb.init(project='Visual Prompting')
    #     wandb.config.update(args)
    #     wandb.run.name = args.filename
    #     wandb.watch(prompter, criterion, log='all', log_freq=10)

    if args.evaluate:
        acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model, prompter, add_prompter, criterion,
                             args)
        return

    epochs_since_improvement = 0

    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, texts_train, model, model_text, model_image, prompter, add_prompter, optimizer, scheduler,
              criterion, scaler, epoch, args)

        # evaluate on validation set
        if epoch % args.validate_freq == 0:
            acc1_mean = validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
                                 prompter, add_prompter, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1_mean > best_acc1
        best_acc1 = max(acc1_mean, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': prompter.state_dict(),
            'add_prompter': add_prompter.state_dict(),
            'clip': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best=is_best)

        if is_best:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            print(f"There's no improvement for {epochs_since_improvement} epochs.")

            if epochs_since_improvement >= args.patience:
                print("The training halted by early stopping criterion.")
                break

    # wandb.run.finish()


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, X, target, text_tokens, alpha,
               attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        prompted_images = prompter(clip_img_preprocessing(X + delta))
        prompt_token = add_prompter()
        # output, _ = model(prompted_images, text_tokens, prompt_token)
        # print(output.size(), target.size(), 'multi')

        # image_embedding = model_image(prompted_images, prompt_token)
        # text_embedding = model_text(text_tokens)
        #
        # logit_scale = model.logit_scale.exp()
        # output, _ = create_logits(image_embedding, text_embedding, logit_scale)
        # print('prompted_images', prompted_images.size(), target.size())

        output, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def attack_pgd_noprompt(prompter, model, model_text, model_image, criterion, X, target, text_tokens, alpha,
                        attack_iters, norm, restarts=1, early_stop=True, epsilon=0):
    delta = torch.zeros_like(X).cuda()
    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0), -1)
        n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
        r = torch.zeros_like(n).uniform_(0, 1)
        delta *= r / n * epsilon
    else:
        raise ValueError
    delta = clamp(delta, lower_limit - X, upper_limit - X)
    delta.requires_grad = True
    for _ in range(attack_iters):
        # output = model(normalize(X ))

        _images = clip_img_preprocessing(X + delta)
        # output, _ = model(_images, text_tokens)

        output, _ = multiGPU_CLIP(model_image, model_text, model, _images, text_tokens, None)

        loss = criterion(output, target)

        loss.backward()
        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]
        x = X[:, :, :, :]
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    return delta


def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
    image_features = model_image(images, prompt_token=prompt_token)
    text_features = model_text(text_tokens).detach()   # When we finetuning CLIP, we do not change the text encoder.

    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    logit_scale = model.logit_scale.exp()
    scaled_text_features = logit_scale * text_features

    logits_per_image = image_features @ scaled_text_features.t()
    logits_per_text = scaled_text_features @ image_features.t()
    return logits_per_image, logits_per_text


# def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
#     img_embed, scale_text_embed = model(images, text_tokens, prompt_token)
#     # print('test len', len(scale_text_embed), scale_text_embed.size())
#     # print('img', img_embed.size())
#     # exit()

#     # output, _ = create_logits(image_embedding, text_embedding, logit_scale)
#     logits_per_image = img_embed @ scale_text_embed.t()
#     logits_per_text = scale_text_embed @ img_embed.t()
#     return logits_per_image, logits_per_text

# def multiGPU_CLIP(model_image, model_text, model, images, text_tokens, prompt_token=None):
#     image_embedding = model_image(images, prompt_token)
#     text_embedding = model_text(text_tokens)
#
#     print('image_embedding',images.size(),  len(image_embedding))
#
#
#     logit_scale = model.logit_scale.exp()
#     output, _ = create_logits(image_embedding, text_embedding, logit_scale)
#     return output, _


def train(train_loader, texts, model, model_text, model_image, prompter, add_prompter, optimizer, scheduler, criterion,
          scaler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    prompter.train()

    num_batches_per_epoch = len(train_loader)

    alpha = 1 / 255
    attack_iters = 5

    print('text token', texts)

    end = time.time()
    for i, (images, target) in enumerate(tqdm(train_loader)):

        # measure data loading time
        data_time.update(time.time() - end)

        BATCH_SIZE = images.size(0)
        # print('bs', BATCH_SIZE)

        # adjust learning rate
        step = num_batches_per_epoch * epoch + i
        scheduler(step)

        optimizer.zero_grad()

        images = images.to(device)
        target = target.to(device)
        text_tokens = clip.tokenize(texts).to(device)

        # print(images.min(), images.max())

        # with automatic mixed precision
        with autocast():

            delta = attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, images,
                               target, text_tokens, alpha, attack_iters, 'l_inf', epsilon=args.train_eps)
            # print('delta', delta.min(), delta.max())

            tmp = clip_img_preprocessing(images + delta)
            prompted_images = prompter(tmp)
            prompt_token = add_prompter()
            # output, _ = model(prompted_images, text_tokens, prompt_token) ## only for single GPU

            # for multiple GPU
            output, _ = multiGPU_CLIP(model_image, model_text, model, prompted_images, text_tokens, prompt_token)

            loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
        scaler.update()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)

        # measure accuracy
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if args.debug:
                break
            # break

            # if args.use_wandb:
            #     wandb.log({
            #         'training_loss': losses.avg,
            #         'training_acc': top1.avg
            #          })

        if i % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': prompter.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, args)

    return losses.avg, top1.avg


# def validate(val_loader, texts, model, prompter, add_prompter, criterion, args):
def validate(val_loader_list, val_dataset_name, texts_list, model, model_text, model_image,
             prompter, add_prompter, criterion, args):
    dataset_num = len(val_loader_list)
    acc_all = []

    for cnt in range(dataset_num):

        val_loader = val_loader_list[cnt]
        texts = texts_list[cnt]
        dataset_name = val_dataset_name[cnt]

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1_org = AverageMeter('Original Acc@1', ':6.2f')
        top1_prompt = AverageMeter('Prompt Acc@1', ':6.2f')
        top1_adv_org = AverageMeter('Adv Original Acc@1', ':6.2f')
        top1_adv_prompt = AverageMeter('Adv Prompt Acc@1', ':6.2f')

        progress = ProgressMeter(
            len(val_loader),
            [batch_time, losses, top1_org, top1_prompt, top1_adv_org, top1_adv_prompt],
            prefix=dataset_name + '_Validate: ')

        # switch to evaluation mode
        prompter.eval()
        add_prompter.eval()

        # print(val_dataset_name, 'text token', texts_list)

        #
        end = time.time()
        for i, (images, target) in enumerate(tqdm(val_loader)):

            if 'cifar' not in val_dataset_name:
                if i % 20 != 0:
                    continue

            images = images.to(device)
            target = target.to(device)
            text_tokens = clip.tokenize(texts).to(device)

            # print(images.size())

            with autocast():

                # clean images, with prompt and without prompt
                # compute output
                with torch.no_grad():
                    prompt_token = add_prompter()
                    # output_prompt, _ = model(prompter(clip_img_preprocessing(images)), text_tokens, prompt_token)
                    output_prompt, _ = multiGPU_CLIP(model_image, model_text, model,
                                                     prompter(clip_img_preprocessing(images)), text_tokens,
                                                     prompt_token)

                    # output_org, _ = model(clip_img_preprocessing(images), text_tokens)
                    output_org, _ = multiGPU_CLIP(model_image, model_text, model, clip_img_preprocessing(images),
                                                  text_tokens, None)

                    loss = criterion(output_prompt, target)

                    # measure accuracy and record loss
                    acc1 = accuracy(output_prompt, target, topk=(1,))
                    losses.update(loss.item(), images.size(0))
                    top1_prompt.update(acc1[0].item(), images.size(0))

                    acc1 = accuracy(output_org, target, topk=(1,))
                    top1_org.update(acc1[0].item(), images.size(0))

                torch.cuda.empty_cache()

                # generate adv example
                delta_prompt = attack_pgd(prompter, model, model_text, model_image, add_prompter, criterion, images,
                                          target, text_tokens,
                                          args.test_stepsize, args.test_numsteps, 'l_inf', epsilon=args.test_eps)

                # compute output
                torch.cuda.empty_cache()
                with torch.no_grad():
                    prompt_token = add_prompter()
                    # output_prompt_adv, _ = model(prompter(clip_img_preprocessing(images + delta_prompt)), text_tokens, prompt_token)
                    output_prompt_adv, _ = multiGPU_CLIP(model_image, model_text, model,
                                                         prompter(clip_img_preprocessing(images + delta_prompt)),
                                                         text_tokens, prompt_token)

                    loss = criterion(output_prompt_adv, target)

                # bl attack
                torch.cuda.empty_cache()

                delta_noprompt = attack_pgd_noprompt(prompter, model, model_text, model_image, criterion, images,
                                                     target, text_tokens, args.test_stepsize, args.test_numsteps,
                                                     'l_inf', epsilon=args.test_eps)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    # output_org_adv, _ = model(clip_img_preprocessing(images+delta_noprompt), text_tokens)
                    output_org_adv, _ = multiGPU_CLIP(model_image, model_text, model,
                                                      clip_img_preprocessing(images + delta_noprompt),
                                                      text_tokens, None)

                torch.cuda.empty_cache()
                # measure accuracy and record loss
                acc1 = accuracy(output_prompt_adv, target, topk=(1,))
                losses.update(loss.item(), images.size(0))
                top1_adv_prompt.update(acc1[0].item(), images.size(0))

                acc1 = accuracy(output_org_adv, target, topk=(1,))
                top1_adv_org.update(acc1[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if args.debug:
                    break

        torch.cuda.empty_cache()

        print(dataset_name + ' * Adv Prompt Acc@1 {top1_adv_prompt.avg:.3f} Adv Original Acc@1 {top1_adv_org.avg:.3f} '
                             '*  Prompt Acc@1 {top1_prompt.avg:.3f} Original Acc@1 {top1_org.avg:.3f}'
              .format(top1_adv_prompt=top1_adv_prompt, top1_adv_org=top1_adv_org,
                      top1_prompt=top1_prompt, top1_org=top1_org))
        acc_all.append(top1_adv_prompt.avg)

    # if args.use_wandb:
    #     wandb.log({
    #         'val_loss': losses.avg,
    #         'val_acc_prompt': top1_prompt.avg,
    #         'val_acc_org': top1_org.avg,
    #     })

    return np.mean(acc_all)


if __name__ == '__main__':
    main()
