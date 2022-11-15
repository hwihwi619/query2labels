import argparse
import os, sys
import random
import datetime
import time
from typing import List
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import mlflow
import torchmetrics
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall

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
    available_models = ['Q2L-R101-448', 'Q2L-R101-576', 'Q2L-TResL-448', 'Q2L-TResL_22k-448', 'Q2L-SwinL-384', 'Q2L-CvT_w24-384']

    parser = argparse.ArgumentParser(description='Query2Label for multilabel classification')
    parser.add_argument('--dataname', help='dataname', default='custom', choices=['custom, chess'])
    parser.add_argument('--dataset_dir', help='dir of dataset')
    
    parser.add_argument('--run_name', default=None, type=str)
    parser.add_argument('--target', default=None, type=str)
    
    parser.add_argument('--img_size', default=448, type=int,
                        help='image size. default(448)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='Q2L-R101-448',
                        choices=available_models,
                        help='model architecture: ' +
                            ' | '.join(available_models) +
                            ' (default: Q2L-R101-448)')
    parser.add_argument('--config', type=str, help='config file')

    parser.add_argument('--output', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--loss', metavar='LOSS', default='asl', 
                        choices=['asl'],
                        help='loss functin')
    parser.add_argument('--num_class', default=9, type=int,
                        help="Number of classes.")
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 16), this is the total '
                            'batch size of all GPUs')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--infer', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--transfer', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--transfer_omit', default=[], type=str, nargs='*')
    

    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model. default is False. ')

    parser.add_argument('--eps', default=1e-5, type=float,
                    help='eps for focal loss (default: 1e-5)')

    # distribution training
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3451', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help='use mixture precision.')
    # data aug
    parser.add_argument('--orid_norm', action='store_true', default=False,
                        help='using oridinary norm of [0,0,0] and [1,1,1] for mean and std.')


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
    args = parser.parse_args()

    # update parameters with pre-defined config file
    if args.config:
        with open(args.config, 'r', encoding='UTF8') as f:
            cfg_dict = json.load(f)
        for k,v in cfg_dict.items():
            setattr(args, k, v)

    return args

def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()
    
    mlflow.set_tracking_uri("http://192.168.0.56:5000")
    remote_server_uri = "http://192.168.0.56:5000" # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.56:9090"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    mlflow.set_experiment("CvT_Inference")
    run_name=None
    if args.run_name:
        run_name=args.run_name
    mlflow.start_run(run_name=run_name)
    
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
    
    # set output dir and logger
    if not args.output:
        args.output = (f"logs/{args.arch}-{datetime.datetime.now()}").replace(' ', '-')
    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: "+' '.join(sys.argv))


    # save config to outputdir
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

    # build model
    model = build_q2l(args)
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)
    criterion = models.aslloss.AsymmetricLossOptimized(
        gamma_neg=args.gamma_neg, gamma_pos=args.gamma_pos,
        disable_torch_grad_focal_loss=True,
        eps=args.eps,
    )


    # optionally resume from a checkpoint
    if args.infer:
        if os.path.isfile(args.infer):
            logger.info("=> loading checkpoint '{}'".format(args.infer))
            checkpoint = torch.load(args.infer, map_location=torch.device(dist.get_rank()))
            state_dict = clean_state_dict(checkpoint['state_dict'])
            # for omit_name in args.transfer_omit:
            #     del state_dict[omit_name]
            model.module.load_state_dict(state_dict, strict=True)
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.infer))

    # Data loading code
    _, test_dataset = get_datasets(args)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=test_sampler)

    names = ['정상', args.target]
    
    _, mAP, aps, acc, f1score, precision, recall, scores = inference(test_loader, model, criterion, args, logger, names)
    logger.info(' * mAP {mAP:.1f} acc {acc:.1f} f1score {f1score:.1f} precision {precision:.1f} recall {recall:.1f}'
            .format(mAP=mAP, acc=acc, f1score=f1score, precision=precision, recall=recall))
    
    class_score_thresh_dict = {}
    class_score_thresh_dict["홍계"]     =  0.5
    class_score_thresh_dict["배꼽"]     =  0.5
    class_score_thresh_dict["피부손상F"] = 0.5
    class_score_thresh_dict["피부손상C"] = 0.5
    class_score_thresh_dict["피부손상S"] = 0.5
    class_score_thresh_dict["골절C"]    =  0.5
    class_score_thresh_dict["가슴멍"]    = 0.5
    class_score_thresh_dict["날개멍"]    = 0.5
    class_score_thresh_dict["다리멍"]    = 0.5
    print(args.target, 'threshold :', class_score_thresh_dict[args.target])
    
    make_csv(args.output, args.target, scores, class_score_thresh_dict, names)

    mlflow.log_artifacts(args.output)
    mlflow.end_run()


    


@torch.no_grad()
def inference(test_loader, model, criterion, args, logger, names):
    scores = {name: {} for name in names}  # {클래스: {파일명: 스코어}}
    
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
        len(test_loader),
        [losses, acc, f1score, precision, recall],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    saved_data = []
    with torch.no_grad():
        for i, (images, target, img_path) in enumerate(test_loader):
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
                scores[args.target][img_path[0]] = output_sm.item()

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

    return loss_avg, mAP, aps, acc.avg, f1score.avg, precision.avg, recall.avg, scores


##################################################################################

def _meter_reduce(meter):
    meter_sum = torch.FloatTensor([meter.sum]).cuda()
    meter_count = torch.FloatTensor([meter.count]).cuda()
    torch.distributed.reduce(meter_sum, 0)
    torch.distributed.reduce(meter_count, 0)
    meter_avg = meter_sum / meter_count

    return meter_avg.item()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
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
    # used for training only.
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist

# def get_predict(img_path, scores, class_score_thresh_dict, names):
#     # norm_index = names.index('정상')
    norm_index = 1
    new_label = norm_index
    for name in names:
        if name =='정상':
            continue
        # if new_label == 9 and score_dict[img_path][name] > class_score_thresh_dict[name]:
        if new_label == norm_index and scores[name][img_path] > class_score_thresh_dict[name]:
            new_label = names.index(name)
#     return new_label

def make_csv(output, target, scores, class_score_thresh_dict, names):
    cum_result = {"정상":0, "홍계":0, "배꼽":0, "피부손상F":0, "피부손상C":0, "피부손상S":0, "골절C":0, "가슴멍":0, "날개멍":0, "다리멍":0}
    score_values=[]    
    
    score_path = os.path.join(output, f'scores.csv')
    img_paths = list(scores[target].keys())
    
    for img_path in img_paths:
        if scores[target][img_path] > class_score_thresh_dict[target]:
            conclusion = target
        else:
            conclusion = '정상'
        score_values.append([img_path] + [scores[target][img_path]] + [conclusion])
        cum_result[conclusion] += 1

    # write scores.csv
    with open(score_path, 'w', encoding='utf-8-sig') as f:
        columns = ["image_path"] + [target] + ["predict"]
        f.write(','.join(columns)+'\n')
        for value in score_values:
            f.write(','.join([value[0]] + list(map(lambda v: f"{v:.6f}", value[1:-1]))+[value[-1]])+'\n')

    print()
    print('▶▶ 최종결과 :', cum_result)

if __name__ == '__main__':
    main()
    