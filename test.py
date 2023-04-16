import torch
import pdb
import argparse
import torch.nn as nn
import random
import os
from models import *
import numpy as np
from dataset.dataset import Cifar10Dataset
from mmcv import Config
from torch.utils.data import DataLoader
from torch.autograd import Variable
from log.logger import Logger
from utils.get_net import get_network
from dataset.dataset import ENSODataset, inverse_normalization, make_rolling_Data_monthly
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

def parser():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--config', '-c', default='./config/config.py', help='config file path')
    parser.add_argument('--net', '-n', type=str, default = 'Predrnn_ED', required=True, help='input which model to use')
    parser.add_argument('--time_interval', default='5', type=int, help='time interval of prediction')
    parser.add_argument('--pretrain', '-p', action='store_true', help='Location pretrain data')
    parser.add_argument('--device', '-d', type=str, default='cuda:1', help='GPU ID')
    args = parser.parse_args()
    return args

def DataLoad(cfg,args):
    valid_CMIP_input,valid_CMIP_target = make_rolling_Data_monthly(ENSO_year=args.ENSO_year,time_interval=args.time_interval, is_train=False)
    validset = ENSODataset(data_tensor=valid_CMIP_input,target_tensor=valid_CMIP_target)
    test_loader = DataLoader(dataset=validset, batch_size=cfg.PARA.train.batch_size,
                              shuffle=False, num_workers=cfg.PARA.train.num_workers)
    return test_loader


def test(net, epoch, test_loader, log, args, cfg):
    test_pred = []
    test_losses = []
    net.eval()
    test_loss = 0.0
    test_total = 0.0
    with torch.no_grad():  # 强制之后的内容不进行计算图的构建，不使用梯度反传
        for i, data in enumerate(test_loader, 0):
            length = len(test_loader)
            _, inputs, labels = data
            inputs, labels = Variable(inputs.cuda(args.device)), Variable(labels.cuda(args.device))
            outputs,loss = net(inputs,labels)
            test_pred.append([item.cpu().detach().numpy() for item in outputs])
            test_total += labels.size(0)
            test_loss += loss.item()
            loss_avg = test_loss / length
            test_losses.append(loss_avg)
        log.logger.info('Test | Loss: %.5f' % loss_avg)
        with open(cfg.PARA.utils_paths.visual_path + args.net + '_test.txt', 'w+') as f:
                f.write('epoch=%d,loss=%.5f\n' % (epoch + 1, test_loss / length))
        inf = np.array(test_pred)
        print('inf.shape:',inf.shape)
        inf_valid = inf
        inverse_inf = inverse_normalization(inf_valid)
        inference = {"epoch": epoch, "inf": inverse_inf}
        if not os.path.exists(cfg.PARA.utils_paths.result_path + args.net + '/'+'leadtime=' + str(args.time_interval)):
            os.makedirs(cfg.PARA.utils_paths.result_path + args.net + '/' +'leadtime='+ str(args.time_interval))
        torch.save(inference, cfg.PARA.utils_paths.result_path + args.net + '/' 
                    +'leadtime='+ str(args.time_interval) + '/' +  'test.pt')
            

def main():
    args = parser()
    cfg = Config.fromfile(args.config)
    log = Logger('./cache/log/' + args.net + '_testlog.txt', level='info')
    log.logger.info('==> Preparing data <==')
    test_loader = DataLoad(cfg)
    log.logger.info('==> Loading model <==')
    net = get_network(args,cfg).cuda()
    # net = torch.nn.DataParallel(net, device_ids=cfg.PARA.train.device_ids)
    log.logger.info("==> Waiting Test <==")
    for epoch in range(100, 101):
        # log.logger.info("==> Epoch:%d <=="%epoch)
        checkpoint = torch.load(cfg.PARA.utils_paths.checkpoint_path + args.net +
         '/'+'leadtime=' + str(args.time_interval)+ '/' +  'chekpoint.pth')
        # checkpoint = torch.load('./cache/checkpoint/' + args.net + '/' + str(60) + 'ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        test(net, epoch, test_loader, log, args, cfg)

    log.logger.info('*'*25)

if __name__ == '__main__':
    main()







