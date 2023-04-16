#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from mmcv import Config
import numpy as np
from log.logger import Logger
from dataset.dataset import ENSODataset, inverse_normalization, make_rolling_Data_monthly
from utils.get_net import get_network
from utils.visualization import plot_acc_loss

os.environ['CUDA_VISION_DEVICES'] = '0'

assert torch.cuda.is_available(), 'Error: CUDA is not find!'


def parser():
    parse = argparse.ArgumentParser(description='Pytorch Cifar10 Training')
    # parse.add_argument('--local_rank',default=0,type=int,help='node rank for distributedDataParallel')
    parse.add_argument('--config', '-c', default='./config/config.py', help='config file path')
    # parse.add_argument('--net','-n',type=str,required=True,help='input which model to use')
    parse.add_argument('--net', '-n', default='CNN')
    parse.add_argument('--time_interval', default='1', type=int, help='time interval of prediction')
    parse.add_argument('--pretrain', '-p', action='store_true', help='Location pretrain data')
    parse.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    # parse.add_argument('--NumClasses','-nc',type=int,default=)
    # training/test
    parse.add_argument('--is_training', type=int, default=1)
    parse.add_argument('--device', type=str, default='cpu:0')
    args = parse.parse_args()
    # print(argparse.local_rank)
    return args


def get_model_params(net, args, cfg):
    total_params = sum(p.numel() for p in net.parameters())
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    with open(cfg.PARA.utils_paths.params_path + args.net + '_params.txt', 'a') as f:
        f.write('total_params:%d\n' % total_params)
        f.write('total_trainable_params: %d\n' % total_trainable_params)


def DataLoad(cfg,args):
    train_CMIP_input,train_CMIP_target = make_rolling_Data_monthly(time_interval=args.time_interval, is_train=True)
    valid_CMIP_input,valid_CMIP_target = make_rolling_Data_monthly(time_interval=args.time_interval, is_train=False)
    trainset = ENSODataset(data_tensor=train_CMIP_input,target_tensor=train_CMIP_target)
    validset = ENSODataset(data_tensor=valid_CMIP_input,target_tensor=valid_CMIP_target)
    train_loader = DataLoader(dataset=trainset, batch_size=cfg.PARA.train.batch_size,
                              shuffle=False, num_workers=cfg.PARA.train.num_workers)
    valid_loader = DataLoader(dataset=validset, batch_size=cfg.PARA.train.batch_size,
                              shuffle=False, num_workers=cfg.PARA.train.num_workers)
    return train_loader, valid_loader


def train(net, criterion, optimizer, train_loader, valid_loader, args, log, cfg):
    valid_pred = []
    train_pred = []
    inf = []
    min_loss = 100000  # 随便设置一个比较大的数
    for epoch in range(cfg.PARA.train.epochs):
        net.train()
        train_loss = 0.0
        train_total = 0.0
        valid_losses = []
        for i, data in enumerate(train_loader, 0):
            length = len(train_loader)  # length = 47500 / batch_size
            _, inputs, labels = data
            inputs, labels = Variable(inputs.cuda(args.device)), Variable(labels.cuda(args.device))
            optimizer.zero_grad()
            outputs, loss = net(inputs, labels)
            # loss = criterion(outputs, labels)
            # pdb.set_trace()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_pred.append([item.cpu().detach().numpy() for item in outputs])
            train_total += labels.size(0)
            if (i + 1 + epoch * length) % 100 == 0:
                log.logger.info('[Epoch:%d, iter:%d] Loss: %.5f '
                                % (epoch + 1, (i + 1 + epoch * length), train_loss / (i + 1)))
        with open(cfg.PARA.utils_paths.visual_path + args.net + '_train.txt', 'w+') as f:
            f.write('epoch=%d,loss=%.5f\n' % (epoch + 1, train_loss / length))

        net.eval()
        valid_loss = 0.0
        valid_total = 0.0
        with torch.no_grad():  # 强制之后的内容不进行计算图的构建，不使用梯度反传
            for i, data in enumerate(valid_loader, 0):
                length = len(valid_loader)
                _, inputs, labels = data
                inputs, labels = Variable(inputs.cuda(args.device)), Variable(labels.cuda(args.device))
                outputs = net(inputs)
                valid_pred.append([item.cpu().detach().numpy() for item in outputs])
                valid_total += labels.size(0)
                # correct += (predicted == labels).sum()
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                loss_avg = valid_loss / length
                valid_losses.append(loss_avg)
            log.logger.info('Validation | Loss: %.5f' % loss_avg)
            with open(cfg.PARA.utils_paths.visual_path + args.net + '_valid.txt', 'w+') as f:
                f.write('epoch=%d,loss=%.5f\n' % (epoch + 1, valid_loss / length))

        valid_loss_epoch = np.average(valid_losses)  # 每个epoch的loss均值
        if valid_loss_epoch < min_loss:  # 只保存valid loss最优的模型和最优inf
            min_loss = valid_loss_epoch
            '''save model's best net & epoch to checkpoint'''
            log.logger.info('Save model to checkpoint ')
            checkpoint = {'net': net.state_dict(), 'epoch': epoch}
            if not os.path.exists(cfg.PARA.utils_paths.checkpoint_path + args.net):
                os.makedirs(cfg.PARA.utils_paths.checkpoint_path + args.net)
            torch.save(checkpoint, cfg.PARA.utils_paths.checkpoint_path + args.net + '/' + str(epoch + 1) + 'ckpt.pth')
            '''inference from validating'''
            inf = np.array(valid_pred[-1])
            inf_valid = inf[-1, ...].reshape(12, 20, 50)
            inverse_inf = inverse_normalization(inf_valid)
            inference = {"epoch": epoch, "inf": inverse_inf}
            if not os.path.exists(cfg.PARA.utils_paths.result_path + args.net):
                os.makedirs(cfg.PARA.utils_paths.result_path + args.net)
            torch.save(inference, cfg.PARA.utils_paths.result_path + args.net + '/' + str(epoch + 1) + 'inf.pt')


def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    for path in cfg.PARA.utils_paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
    log = Logger(cfg.PARA.utils_paths.log_path + args.net + '_trainlog.txt', level='info')

    start_epoch = 0

    log.logger.info('==> Preparing dataset <==')
    train_loader, valid_loader = DataLoad(cfg,args)

    log.logger.info('==> Loading model <==')
    if args.pretrain:
        log.logger.info('Loading Pretrain Data')

    net = get_network(args, cfg).cuda(args.device)
    criterion = nn.MSELoss().cuda(args.device)
    optimizer = optim.Adam(net.parameters(), lr=cfg.PARA.train.lr)
    log.logger.info('==> SUM NET Params <==')
    get_model_params(net, args, cfg)

    # if torch.cuda.device_count()>1:#DataParallel is based on Parameter server
    #     net = nn.DataParallel(net, device_ids=cfg.PARA.train.device_ids)
    torch.backends.cudnn.benchmark = True

    '''断点续训否'''
    if args.resume:
        log.logger.info('Resuming from checkpoint')
        checkpoint = torch.load(cfg.PARA.utils_paths.checkpoint_path + args.net + '/' + args.epoch + 'ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']

    log.logger.info('==> Waiting Train <==')
    train(net=net, criterion=criterion, optimizer=optimizer,
          train_loader=train_loader, valid_loader=valid_loader, args=args, log=log, cfg=cfg)
    log.logger.info('==> Finish Train <==')

    log.logger.info('==> Plot Train_Vilid Loss & Save to Visual <==')
    plot_acc_loss(args, cfg=cfg)
    log.logger.info('*' * 25)


if __name__ == '__main__':
    main()
