import argparse
import math
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import mlflow
to_mlflow=['dataname', 'img_size', 'num_class', 'orid_norm', \
    'backbone', 'pretrained', 'transfer', 'transfer_omit', \
        'optim', 'eps', 'dtgfl', 'gamma_pos', 'gamma_neg', 'loss_clip', 'lr', 'weight_decay', 'dropout', \
            'dim_feedforward', 'hidden_dim', 'enc_layers', 'dec_layers', \
                'dataset_dir', 'nheads', 'pre_norm', 'amp']
import torchmetrics
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

from torch.utils.tensorboard import SummaryWriter

import _init_paths
from dataset.get_dataset import get_datasets

from utils.logger import setup_logger
import models
import models.aslloss
from models.query2label import build_q2l
from utils.metric import voc_mAP
from utils.misc import clean_state_dict
from utils.slconfig import get_raw_dict


def parser_args():
    parser = argparse.ArgumentParser(description='Query2Label MSCOCO Training')
    parser.add_argument('--dataname', help='dataname', default='custom', choices=['coco14', 'custom'])
    parser.add_argument('--dataset_dir', help='dir of dataset', default='/comp_robot/liushilong/data/COCO14/')
    parser.add_argument('--run_name', default=None, type=str)
    parser.add_argument('--target', default=None, type=str)
    parser.add_argument('--img_size', default=448, type=int,
                        help='size of input images')

    parser.add_argument('--output', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--num_class', default=80, type=int,
                        help="Number of query slots")
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')
    parser.add_argument('--optim', default='AdamW', type=str, choices=['AdamW', 'Adam_twd'],
                        help='which optim to use')

    # loss
    parser.add_argument('--eps', default=1e-5, type=float,
                        help='eps for focal loss (default: 1e-5)')
    parser.add_argument('--dtgfl', action='store_true', default=False, 
                        help='disable_torch_grad_focal_loss in asl')              
    parser.add_argument('--gamma_pos', default=0, type=float,
                        metavar='gamma_pos', help='gamma pos for simplified asl loss')
    parser.add_argument('--gamma_neg', default=2, type=float,
                        metavar='gamma_neg', help='gamma neg for simplified asl loss')
    parser.add_argument('--loss_dev', default=-1, type=float,
                                            help='scale factor for loss')
    parser.add_argument('--loss_clip', default=0.0, type=float,
                                            help='scale factor for clip')  

    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--val_interval', default=1, type=int, metavar='N',
                        help='interval of validation')

    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay (default: 1e-2)',
                        dest='weight_decay')

    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_omit', default=[], type=str, nargs='*')
    parser.add_argument('--transfer', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--transfer_omit', default=[], type=str, nargs='*')    

    parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')


    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')


    # data aug
    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using mean [0,0,0] and std [1,1,1] to normalize input images')


    # * Transformer
    parser.add_argument('--enc_layers', default=1, type=int, 
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=8192, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=2048, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--backbone', default='resnet101', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--keep_other_self_attn_dec', action='store_true', 
                        help='keep the other self attention modules in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_first_self_attn_dec', action='store_true',
                        help='keep the first self attention module in transformer decoders, which will be removed default.')
    parser.add_argument('--keep_input_proj', action='store_true', 
                        help="keep the input projection layer. Needed when the channel of image features is different from hidden_dim of Transformer layers.")

    # * raining
    parser.add_argument('--amp', action='store_true', default=False,
                        help='apply amp')
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--kill-stop', action='store_true', default=False,
                        help='apply early stop')
    args = parser.parse_args()
    return args

def get_args():
    args = parser_args()
    return args


best_mAP = 0
best_Acc = 0

def main():
    args = get_args()
    
    mlflow.set_tracking_uri("http://192.168.0.56:5000")
    remote_server_uri = "http://192.168.0.56:5000" # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.56:9090"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    mlflow.set_experiment("Transfer-CvT_OneLabel")
    run_name=None
    if args.run_name:
        run_name=args.run_name
    mlflow.start_run(run_name=run_name)
    for k, v in vars(args).items():
        if k in to_mlflow:
            mlflow.log_param(k,v)
    
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        #   python main.py ...
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(args.local_rank))

    return main_worker(args, logger)

def main_worker(args, logger):
    global best_mAP
    global best_Acc

    # build model
    model = build_q2l(args)
    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay) # 0.9997
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # criterion
    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        clip=args.loss_clip,
        disable_torch_grad_focal_loss=args.dtgfl,
        eps=args.eps,
    )

    # optimizer
    args.lr_mult = args.batch_size / 256
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )          
    else:
        raise NotImplementedError


    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            # import ipdb; ipdb.set_trace()
            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=False)
            # model.module.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
        
    elif args.transfer:
        if os.path.isfile(args.transfer):
            logger.info("=> loading checkpoint for Transfer'{}'".format(args.transfer))
            checkpoint = torch.load(args.transfer, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.transfer_omit))
            # import ipdb; ipdb.set_trace()
            for omit_name in args.transfer_omit:
                del state_dict[omit_name]
            
            model.module.load_state_dict(state_dict, strict=False)            
            
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.transfer))

    # Data loading code
    train_dataset, val_dataset = get_datasets(args)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    
    acc = AverageMeter('Acc', ':5.3f')
    f1score = AverageMeter('f1score', ':5.3f')
    recall = AverageMeter('recall', ':5.3f')
    precision = AverageMeter('precision', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [losses, mAPs, acc, f1score, precision, recall],
        prefix='=> Test Epoch: ')
    
    acc_ema = AverageMeter('Acc_ema', ':5.3f')
    f1score_ema = AverageMeter('f1score_ema', ':5.3f')
    recall_ema = AverageMeter('recall_ema', ':5.3f')
    precision_ema = AverageMeter('precision_ema', ':5.3f')
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress_ema = ProgressMeter(
            args.epochs,
            [losses_ema, mAPs_ema, acc_ema, f1score_ema, recall_ema, precision_ema],
            prefix='=> EMA Test Epoch: ')


    # one cycle learning rate
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.2)


    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model.module, args.ema_decay)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # train for one epoch
        loss, acc, f1score, precision, recall = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger)

        mlflow.log_metric('metric/learning_rate', optimizer.param_groups[0]['lr'], step=epoch)
        mlflow.log_metric('metric/train_loss', loss, step=epoch)
        mlflow.log_metric('metric/train_acc', acc, step=epoch)
        mlflow.log_metric('metric/train_f1score', f1score, step=epoch)
        mlflow.log_metric('metric/train_precision', precision, step=epoch)
        mlflow.log_metric('metric/train_recall', recall, step=epoch)

        if epoch % args.val_interval == 0:

            # evaluate on validation set
            loss, mAP, aps, acc, f1score, precision, recall  = validate(val_loader, model, criterion, args, logger)

            mlflow.log_metric('metric/valid_loss', loss, step=epoch)
            mlflow.log_metric('metric/valid_mAP', mAP, step=epoch)
            mlflow.log_metric('metric/valid_acc', acc, step=epoch)
            mlflow.log_metric('metric/valid_f1score', f1score, step=epoch)
            mlflow.log_metric('metric/valid_precision', precision, step=epoch)
            mlflow.log_metric('metric/valid_recall', recall, step=epoch)
            
            losses.update(loss)
            mAPs.update(mAP)
              
            loss_ema, mAP_ema, aps_ema, acc_ema, f1score_ema, precision_ema, recall_ema = validate(val_loader, ema_m.module, criterion, args, logger)
            
            mlflow.log_metric('metric/valid_loss_ema', loss_ema, step=epoch)
            mlflow.log_metric('metric/valid_mAP_ema', mAP_ema, step=epoch)
            mlflow.log_metric('metric/valid_acc_ema', acc_ema, step=epoch)
            mlflow.log_metric('metric/valid_f1score_ema', f1score_ema, step=epoch)
            mlflow.log_metric('metric/valid_precision_ema', precision_ema, step=epoch)
            mlflow.log_metric('metric/valid_recall_ema', recall_ema, step=epoch)
            
            losses_ema.update(loss_ema)
            mAPs_ema.update(mAP_ema)

            regular_mAP_list.append(mAP)
            ema_mAP_list.append(mAP_ema)

            progress.display(epoch, logger)
            progress_ema.display(epoch, logger)

            # remember best (regular) mAP and corresponding epochs
            if mAP > best_regular_mAP:
                best_regular_mAP = max(best_regular_mAP, mAP)
                best_regular_epoch = epoch
            if mAP_ema > best_ema_mAP:
                best_ema_mAP = max(mAP_ema, best_ema_mAP)
            
            if mAP_ema > mAP:
                mAP = mAP_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = mAP > best_mAP
            if is_best:
                best_epoch = epoch
            best_mAP = max(mAP, best_mAP)
            
            is_best_acc = acc > best_Acc
            best_Acc = max(acc, best_Acc)
            # not implemented best acc_ema
            
            logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
            logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))
            logger.info("   | best Acc {}".format(best_Acc))

            if dist.get_rank() == 0 or is_best_acc:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename='model_best', output = args.output)
                mlflow.log_artifacts(args.output)

            if math.isnan(loss) or math.isnan(loss_ema):
                # save_checkpoint({
                #     'epoch': epoch + 1,
                #     # 'arch': args.arch,
                #     'state_dict': model.state_dict(),
                #     'best_mAP': best_mAP,
                #     'optimizer' : optimizer.state_dict(),
                # }, is_best=is_best, filename='checkpoint_nan'))
                logger.info('Loss is NaN, break')
                mlflow.log_artifacts(args.output)
                mlflow.end_run()

                sys.exit(1)


            # early stop
            if args.early_stop:
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 5:
                    mlflow.log_artifacts(args.output)
                    if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        if dist.get_rank() == 0 and args.kill_stop:
                            filename = sys.argv[0].split(' ')[0].strip()
                            killedlist = kill_process(filename, os.getpid())
                            logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist)) 
                        break

    print("Best mAP:", best_mAP)
    
    return 0



def train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    metric_acc = BinaryAccuracy(num_labels=args.num_class).cuda()
    metric_f1score = BinaryF1Score(num_labels=args.num_class).cuda()
    metric_recall = BinaryRecall(num_labels=args.num_class).cuda()
    metric_precision = BinaryPrecision(num_labels=args.num_class).cuda()
    
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    
    acc = AverageMeter('Acc', ':5.3f')
    f1score = AverageMeter('f1score', ':5.3f')
    precision = AverageMeter('precision', ':5.3f')
    recall = AverageMeter('recall', ':5.3f')
    
    progress = ProgressMeter(
        len(train_loader),
        [lr, losses, acc, f1score, precision, recall],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            output = model(images)
            loss = criterion(output, target)
            if args.loss_dev > 0:
                loss *= args.loss_dev
            output_sm = torch.sigmoid(output)
            
        # update metrcis
        metric_acc.update(output_sm, target)
        metric_f1score.update(output_sm, target)
        metric_precision.update(output_sm, target)
        metric_recall.update(output_sm, target)

        # update average
        acc.update(metric_acc.compute())
        f1score.update(metric_f1score.compute())
        precision.update(metric_precision.compute())
        recall.update(metric_recall.compute())

        # record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        if epoch >= args.ema_epoch:
            ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return loss, acc.avg, f1score.avg, precision.avg, recall.avg



@torch.no_grad()
def validate(val_loader, model, criterion, args, logger):
    
    metric_acc = BinaryAccuracy(num_labels=args.num_class).cuda()
    metric_f1score = BinaryF1Score(num_labels=args.num_class).cuda()
    metric_recall = BinaryRecall(num_labels=args.num_class).cuda()
    metric_precision = BinaryPrecision(num_labels=args.num_class).cuda()
    
    losses = AverageMeter('Loss', ':5.3f')
    
    acc = AverageMeter('Acc', ':5.3f')
    f1score = AverageMeter('f1score', ':5.3f')
    precision = AverageMeter('precision', ':5.3f')
    recall = AverageMeter('recall', ':5.3f')

    progress = ProgressMeter(
        len(val_loader),
        [losses, acc, f1score, precision, recall],
        prefix='Test: ')

    # switch to evaluate mode
    saveflag = False
    model.eval()
    saved_data = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast(enabled=args.amp):
                output = model(images)
                loss = criterion(output, target)
                if args.loss_dev > 0:
                    loss *= args.loss_dev
                output_sm = torch.sigmoid(output)
                if torch.isnan(loss):
                    saveflag = True

            # update metrcis
            metric_acc.update(output_sm, target)
            metric_f1score.update(output_sm, target)
            metric_precision.update(output_sm, target)
            metric_recall.update(output_sm, target)

            # update average
            acc.update(metric_acc.compute())
            f1score.update(metric_f1score.compute())
            precision.update(metric_precision.compute())
            recall.update(metric_recall.compute())

            # record loss
            losses.update(loss.item(), images.size(0))

            # save some data
            _item = torch.cat((output_sm.detach().cpu(), target.detach().cpu()), 1)
            del output_sm
            del target
            saved_data.append(_item)

            if i % args.print_freq == 0 and dist.get_rank() == 0:
                progress.display(i, logger)

        logger.info('=> synchronize...')
        if dist.get_world_size() > 1:
            dist.barrier()
        loss_avg, = map(
            _meter_reduce if dist.get_world_size() > 1 else lambda x: x.avg,
            [losses]
        )
        
        # import ipdb; ipdb.set_trace()
        # calculate mAP
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(dist.get_rank())
        np.savetxt(os.path.join(args.output, saved_name), saved_data)
        if dist.get_world_size() > 1:
            dist.barrier()

        if dist.get_rank() == 0:
            print("Calculating mAP:")
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(dist.get_world_size())]
            metric_func = voc_mAP                
            mAP, aps = metric_func([os.path.join(args.output, _filename) for _filename in filenamelist], args.num_class, return_each=True)
            
            logger.info("  mAP: {} Acc: {} f1score: {} precision: {} recall: {}".format(mAP, acc.avg, f1score.avg, precision.avg, recall.avg))
            logger.info("   aps: {}".format(np.array2string(aps, precision=5)))
        else:
            mAP = 0

        if dist.get_world_size() > 1:
            dist.barrier()

    return loss_avg, mAP, aps, acc.avg, f1score.avg, precision.avg, recall.avg


##################################################################################
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', output='path/to/output'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(output, filename+'.pth.tar'))
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', val_only=False):
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
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
        if self.val_only:
            fmtstr = '{name} {val' + self.fmt + '}'
        else:
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class AverageMeterHMS(AverageMeter):
    """Meter for timer in HH:MM:SS format"""
    def __str__(self):
        if self.val_only:
            fmtstr = '{name} {val}'
        else:
            fmtstr = '{name} {val} ({sum})'
        return fmtstr.format(name=self.name, 
                             val=str(datetime.timedelta(seconds=int(self.val))), 
                             sum=str(datetime.timedelta(seconds=int(self.sum))))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('  '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def kill_process(filename:str, holdpid:int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

if __name__ == '__main__':
    main()
