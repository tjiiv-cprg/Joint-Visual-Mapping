from ipdb import set_trace as bp
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import datetime
import numpy as np
import cv2
import random

import argparse
import os
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm
from datasets import LineTopologyData
from datasets import make_DT_GT, show_DT_heatmap, make_thata_GT, make_state_GT, make_state_GT_tensor
from model import TopologyRNN, NNUpsample4, SpatialSoftmax
from utils import check_mkdir, AverageMeter, getROI, computeloss, calculate_acc, back_dir2ori, post_process_predict,\
    normalize, back_pos_to_ori, recover_points, calculate_state_acc, check_any_point_already_in, find_nearest_forkline,\
    plot_state, draw_polyline_on_predictimg, draw_polyline_on_oriimg, colorize_mask, make_visaul_DTmap, \
    compute_fork_diff_loss, make_random_argument, is_continue_merge, get_next_direction
import time
from torch.utils.data import DataLoader
import copy
import torchvision.utils as vutils

cudnn.benchmark = True


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)

def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_folders', type=str, default='fork1p,fork2p,merge1p,merge2p', help='train data folders')
    parser.add_argument('--folders', type=str, default='fork1,fork2,merge1,merge2', help='test data folders')
    parser.add_argument('--train_dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--valid_dataset_dir', type=str, help='The validing dataset dir path')
    parser.add_argument('--epoch_num', type=int, help='The total epoch num')
    parser.add_argument('--lr', type=float, default=1e-5, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='The weight dacay')
    parser.add_argument('--momentum', type=float, default=0.95, help='The weight dacay')
    parser.add_argument('--device', type=str, default='cuda', help='training device')
    parser.add_argument('--lr_patience', type=int, default=200, help='The weight dacay')
    parser.add_argument('--ckpt_path', type=str, default='', help='The model checkpoint path')
    parser.add_argument('--exp_name', type=str, default='topologyRNN', help='experiment name')
    parser.add_argument('--snapshot', type=str, default='', help='The path of pretrained model to resum')
    parser.add_argument('--print_freq', type=int, default=1, help='print frequence')
    parser.add_argument('--RNNlength', type=int, default=50, help='max length of RNN queue')
    parser.add_argument('--visual_rate', type=float, default=1, help='frequence of visualization on tensorboard')
    parser.add_argument('--seed', type=int, default=1, help='The random seed')
    parser.add_argument('--gamma', type=int, default=5, help='gamma of focal-loss in training of state head')
    parser.add_argument('--focal_weights', action='store_true', default=True, help='with focal weights or not in training of state head')

    parser.add_argument('--ROI_size', nargs='+', type=int, default=[64, 32], help='the size of ROI')
    parser.add_argument('--state_ROI_ratio', nargs='+', type=float, default=[1.5, 1.5], help='the size of state ROI')
    parser.add_argument('--pos_ROI_ratio', nargs='+', type=float, default=[0.5, 1], help='the size of ROI')

    parser.add_argument('--weights', nargs='+', type=int, default=[10, 100, 100, 10], help='weight of position, theta, state, DT')
    parser.add_argument('--s_GT', action='store_true', default=False, help='provide state ground truth')
    parser.add_argument('--only_DT', action='store_true', default=False, help='just train DT head')
    parser.add_argument('--DT_ready', action='store_true', default=False, help='DT head is ready')
    parser.add_argument('--state_net', type=str, default='CNN', help='CNN or RNN, if CNN use non_normal_oversample, else use state_normal_less')
    parser.add_argument('--state_normal_less', nargs='+', type=float, default=[0.06, 1, 1, 0.1], help='less sample tario of normal, fork, merge, end')
    parser.add_argument('--non_normal_oversample', nargs='+', type=int, default=[1, 45, 25, 2], help='over sample tario of normal, fork, merge, end')

    parser.add_argument('--pos_net', type=str, default='RNN')
    parser.add_argument('--direction_net', type=str, default='RNN')
    parser.add_argument('--debug', action='store_true', default=False, help='debug means only valid and visualize every result')
    parser.add_argument('--true_train', action='store_true', default=False, help='if true_train validdataset != traindataset')
    parser.add_argument('--fork_weight', type=int, default=4, help='')
    return parser.parse_args()


def main():

    global best_record
    net = TopologyRNN(args.state_net, args.pos_net, args.direction_net).cuda()
    if args.fork_weight == 0: input('取消Fork_weight!!!')
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters()],
         'lr': args.lr, 'weight_decay': args.weight_decay, 'amsgrad': True}
    ])
    if len(args.snapshot) == 0:
        curr_epoch = 1
        best_record = {'val_loss':1e12, 'val_loss_epoch': 0,
                       'precision': 0, 'bset_precision_epoch': 0,
                       'recall' : 0, 'bset_recall_epoch': 0,
                       'F1': 0, 'bset_F1_epoch': 0, 'state_prec':0, 'state_prec_epoch':0, 'line_acc' : 0}

    else:
        print('training resumes from' + args.snapshot)
        pretrained_model = torch.load(args.snapshot)
        if args.DT_ready:
            print('DT and GlobalFeature modle resumes from' + args.snapshot)
            modle_dict = net.state_dict()
            DT_globale_feature_pretrained_modle_dict = {k :v for k , v in pretrained_model.items() if "globalTeatureNet" in k or "DTNet" in k}
            modle_dict.update(DT_globale_feature_pretrained_modle_dict)
            net.load_state_dict(modle_dict)

        else:
            net.load_state_dict(pretrained_model)
            print('opt resuming from {}'.format(args.snapshot))
            optimizer.load_state_dict(torch.load(os.path.join(args.snapshot.replace('epoch_', 'opt_epoch_'))))
            print("DT done small lr for them!")
            for p in net.globalTeatureNet.parameters():
                p.requires_grad = False
            for p in net.DTNet.parameters():
                p.requires_grad = False

        split_snapshot = args.snapshot.split('/')[-1].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        best_record = {'val_loss':float(split_snapshot[3]), 'val_loss_epoch': int(split_snapshot[1]),
                       'precision': float(split_snapshot[7]), 'bset_precision_epoch': int(split_snapshot[1]),
                       'recall' : float(split_snapshot[9]), 'bset_recall_epoch': int(split_snapshot[1]),
                       'F1': float(split_snapshot[5]), 'bset_F1_epoch': int(split_snapshot[1]),
                       'state_prec': float(split_snapshot[12]), 'state_prec_epoch': int(split_snapshot[1]),
                       'line_acc': float(split_snapshot[15]), 'state_prec_epoch': int(split_snapshot[1])}

    net = net.float()
    net.train()
    if args.only_DT:
        for p in net.directionHeader.parameters():
            p.requires_grad = False
        for p in net.stateHeader.parameters():
            p.requires_grad = False
        for p in net.positionHeader.parameters():
            p.requires_grad = False

    elif args.DT_ready:
        print("DT done small lr for them!")
        for p in net.globalTeatureNet.parameters():
            p.requires_grad = False
        for p in net.DTNet.parameters():
            p.requires_grad = False

    ROI_size = args.ROI_size
    folders = args.folders.split(',')
    test_folders = args.test_folders.split(',')
    train_set = LineTopologyData(datadir=args.train_dataset_dir, ROI_size=ROI_size, folders=folders)
    train_loader = DataLoader(train_set, batch_size=1,  num_workers=8, shuffle=True)
    if args.true_train:
        val_set = LineTopologyData(datadir=args.valid_dataset_dir, ROI_size=ROI_size, folders=test_folders)
    else:
        val_set = LineTopologyData(datadir=args.train_dataset_dir, ROI_size=ROI_size, folders=folders)
    val_loader = DataLoader(val_set, batch_size=1,  num_workers=8, shuffle=False)

    check_mkdir(args.ckpt_path)
    check_mkdir(os.path.join(args.ckpt_path, args.exp_name))
    check_mkdir(os.path.join(args.ckpt_path, exp_name))
    if args.only_DT:
        check_mkdir(os.path.join(args.ckpt_path, exp_name, 'only_DT'))
    with open(os.path.join(args.ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w')as file:
        file.write(str(args) + '\n\n')

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, min_lr=1e-14)
    for epoch in range(curr_epoch, args.epoch_num + 1):
        if not args.debug:
            train(train_loader, net, optimizer, epoch)
        valid_acc = validate(val_loader, net, optimizer, epoch)
        scheduler.step(valid_acc)

def train(train_loader, net, optimizer, epoch):
    ROI_size = args.ROI_size
    state_ROI = [int(ROI_size[0] * args.state_ROI_ratio[0]), int(ROI_size[1] * args.state_ROI_ratio[1])]
    pos_ROI = [int(ROI_size[0] * args.pos_ROI_ratio[0]), int(ROI_size[1] * args.pos_ROI_ratio[1])]
    spatial_softmax = SpatialSoftmax(pos_ROI[0], pos_ROI[1], 1, args.device, temperature=None, data_format='NCHW')
    train_loss = AverageMeter()
    train_position_loss = AverageMeter()
    train_direction_loss = AverageMeter()
    train_state_loss = AverageMeter()
    train_DT_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    time1 = time.time()
    weight_position, weight_theta, weight_state, weight_DT= args.weights
    for i, data in enumerate(train_loader):
        img, polylines, DT_GT, polylineMaps, init_nodes, imgName = data
        if len(polylines) > 0:
            loss_flag = False # calculate loss or not
            fork_directions_1 = []
            fork_directions_2 = []
            positionGTs, position_predict, directionGTs, direction_predict, stateGTs, state_predict, predictDT = [None for i in range(7)]

            img = img.float().cuda()
            DT_GT = DT_GT.float().cuda()
            polylineMaps = polylineMaps.cuda().float().squeeze(0)
            polylines = polylines.squeeze(0).cuda().float()
            img = img.unsqueeze(1)
            predictDT, concatFeature = net(img)
            img = img.squeeze(0).squeeze(0)  # shape H*W
            concatFeature = concatFeature.squeeze(0)  # shape C*H/4*W/4
            predictDT = predictDT.squeeze(0)

            this_img_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
            popsition_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
            direction_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
            state_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
            fork_diff_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
            # calculate DT_loss
            DT_loss_ = computeloss(positionGTs, position_predict, directionGTs, direction_predict, stateGTs, state_predict, DT_GT, predictDT, ROI_size)
            DT_loss = weight_DT * DT_loss_[4]
            this_img_loss += DT_loss

            if args.only_DT:
                pass
            else:
                fork_another_lines = []
                fork_another_init_nodes = []

                for id, init_node in enumerate(init_nodes):
                    if init_node["init_position"][1] < 1024 - 1.5 * ROI_size[0] and init_node["init_state"] == 1:
                        polyline = polylines[id]
                        fork_another_lines.append(polyline)
                        fork_another_init_nodes.append(init_node)

                for id, init_node in enumerate(init_nodes):
                    if id < len(polylines):
                        polyline = polylines[id]
                        direction = torch.tensor(init_node["init_direction"]).to(args.device).float()
                        position = torch.tensor(init_node["init_position"]).to(args.device).float()
                        state = init_node["init_state"].to(args.device).float()

                        directionGTs = direction.unsqueeze(0)
                        positionGTs = polyline.to(args.device).float()
                        stateGTs = state.unsqueeze(0)

                        direction_predict = direction.unsqueeze(0)
                        position_predict = position.unsqueeze(0)
                        state_predict = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                        state_predict[0][int(state.item())] = 1.0
                    else:
                        if len(fork_another_init_nodes) == 0:
                            break
                        polyline, true_init_node, picked_id = find_nearest_forkline(init_node, fork_another_init_nodes,
                                                                         fork_another_lines)
                        if polyline is None:
                            continue
                        fork_another_lines.pop(picked_id)
                        fork_another_init_nodes.pop(picked_id)

                        directionGTs = torch.tensor(true_init_node["init_direction"]).unsqueeze(0).to(args.device)
                        positionGTs = polyline.to(args.device).float()
                        # mannual fill 0
                        stateGTs = true_init_node["init_state"].to(args.device).float().unsqueeze(0).fill_(0)

                        direction = init_node["init_direction"]
                        position = init_node["init_position"]
                        state = init_node["init_state"].to(args.device).float()
                        direction_predict = direction.unsqueeze(0)
                        position_predict = position.unsqueeze(0)
                        state_predict = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                        state_predict[0][int(state.item())] = 1.0

                    if init_node["init_position"][1] < 1024 - 1.5 * ROI_size[0] and state == 1:
                        continue

                    direction_hidden = torch.zeros((1, 32, ROI_size[0], ROI_size[1]), dtype=torch.float,
                                                        device='cuda', requires_grad=True)
                    direction_cell = torch.zeros((1, 32, ROI_size[0], ROI_size[1]), dtype=torch.float,
                                                   device='cuda', requires_grad=True)

                    position_hidden = torch.zeros((1, 32, pos_ROI[0], pos_ROI[1]), dtype=torch.float,
                                                        device='cuda', requires_grad=True)
                    position_cell = torch.zeros((1, 32, pos_ROI[0], pos_ROI[1]), dtype=torch.float, device='cuda',
                                                   requires_grad=True)
                    state_hidden = torch.zeros((1, 32, state_ROI[0], state_ROI[1]), dtype=torch.float,
                                                        device='cuda', requires_grad=True)
                    state_cell = torch.zeros((1, 32, state_ROI[0], state_ROI[1]), dtype=torch.float, device='cuda',
                                                   requires_grad=True)

                    count = 0
                    using_pre_dir = False
                    while int(state.item()) < 2: # normal or fork continue
                        count = count + 1
                        if count == args.RNNlength:
                            break
                        # predict direction
                        curROI, angle = getROI(predictDT, position, direction, ROI_size, scale=4, mode='bottom')
                        stateGT_tensor = torch.zeros((4, curROI.shape[1], curROI.shape[2]), dtype=torch.float, device=curROI.device)
                        stateGT_tensor[int(state.item())] = 1.0
                        ROIFeature = torch.cat((curROI, stateGT_tensor), 0).unsqueeze(0)

                        if args.direction_net == "LSTM":
                            direction_out1, direction_out2, (direction_hidden, direction_cell) = net.directionHeader(ROIFeature,
                                                                                (direction_hidden, direction_cell))
                        elif args.direction_net == "RNN":
                            direction_out1, direction_out2, direction_hidden = net.directionHeader(ROIFeature, direction_hidden)

                        pre_directionGT, _, dis = make_thata_GT(position, positionGTs.unsqueeze(0))
                        if state == 0: # normal
                            direction_out = direction_out1
                            direction_out = back_dir2ori(angle, direction_out.squeeze(0))
                        elif state == 1: # fork
                            to_append_position = position
                            to_append_state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)

                            direction_out2_1, direction_out2_2 = direction_out2
                            direction_out2_1 = back_dir2ori(angle, direction_out2_1.squeeze(0))
                            direction_out2_2 = back_dir2ori(angle, direction_out2_2.squeeze(0))
                            fork_directions_1.append(direction_out2_1)
                            fork_directions_2.append(direction_out2_2)
                            next_direction = get_next_direction(position, positionGTs)
                            score1 = F.cosine_similarity(next_direction.unsqueeze(0), direction_out2_1.unsqueeze(0))
                            score2 = F.cosine_similarity(next_direction.unsqueeze(0), direction_out2_2.unsqueeze(0))
                            if score1 >= score2:
                                direction_out = direction_out2_1
                                to_append_direction = direction_out2_2
                            else:
                                direction_out = direction_out2_2
                                to_append_direction = direction_out2_1
                            init_nodes.append(
                                {'init_position': to_append_position, 'init_direction': to_append_direction,
                                 'init_state': to_append_state})

                        if dis > 20:
                            using_pre_dir = True
                            direction = pre_directionGT
                        else:
                            direction = direction_out
                        if direction[1] > 0:
                            using_pre_dir = True
                            direction = pre_directionGT

                        # predict position
                        curROI, angle = getROI(predictDT, position, direction, pos_ROI, scale=4, mode='bottom')
                        stateGT_tensor = torch.zeros((4, curROI.shape[1], curROI.shape[2]), dtype=torch.float,
                                                     device=curROI.device)
                        stateGT_tensor[int(state.item())] = 1.0
                        ROIFeature = torch.cat((curROI, stateGT_tensor), 0).unsqueeze(0)
                        if args.pos_net == "LSTM":
                            position_out, (position_hidden, position_cell) = net.positionHeader(ROIFeature,
                                                                                (position_hidden, position_cell))
                        elif args.pos_net == "RNN":
                            position_out, position_hidden = net.positionHeader(ROIFeature,position_hidden)

                        position_out = spatial_softmax(position_out)
                        position_out = recover_points(position_out, pos_ROI)

                        position_out = back_pos_to_ori(position, position_out, angle, pos_ROI)
                        position = position_out.squeeze(0)
                        if position[0]< 0 or position[0] >= img.shape[1] or position[1]< 0 or position[1]>=img.shape[0]:
                            break
                        position_predict = torch.cat((position_predict, position_out), dim=0)
                        direction_predict = torch.cat((direction_predict, direction_out.unsqueeze(0)), dim=0)

                        # get direction GT
                        directionGT, _, _ = make_thata_GT(position, positionGTs.unsqueeze(0))
                        directionGTs = torch.cat((directionGTs, directionGT.unsqueeze(0)), dim=0)

                        if using_pre_dir:
                            break
                        # get state GT
                        stateGT = make_state_GT_tensor(polylineMaps, polylines, position, direction, ROI_size, imgName, show=True)
                        print(stateGT)
                        if stateGT == 2 and is_continue_merge(positionGTs, position):
                            state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)
                        else:
                            state = torch.tensor([stateGT], dtype=stateGTs.dtype, device=stateGTs.device)
                        # predict state
                        if args.s_GT:
                            state_out = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                            state_out[0][int(stateGT)] = 1.0
                            state_predict = torch.cat((state_predict, state_out), dim=0)
                            stateGTs = torch.cat((stateGTs, torch.tensor([[stateGT]], dtype=stateGTs.dtype, device=stateGTs.device)),dim=0)
                        else:
                            if args.state_net == "RNN":
                                rand_ = random.random()
                                if (stateGT == 0 and rand_ > args.state_normal_less[0]) or (
                                        stateGT == 1 and rand_ > args.state_normal_less[1]) or (
                                        stateGT == 2 and rand_ > args.state_normal_less[2]) or (
                                        stateGT == 3 and rand_ > args.state_normal_less[3]):
                                     pass
                                else:
                                    curROI, angle = getROI(concatFeature, position, direction, state_ROI, scale=4, mode='middle')
                                    state_out, state_hidden = net.stateHeader(curROI.unsqueeze(0), state_hidden)
                                    stateGTs = torch.cat((stateGTs, torch.tensor([[stateGT]], dtype=stateGTs.dtype,
                                                                                  device=stateGTs.device)), dim=0)
                                    state_predict = torch.cat((state_predict, state_out), dim=0)

                            elif args.state_net == "CNN":
                                if stateGT == 0:  # normal
                                    loop_times = args.non_normal_oversample[0]
                                elif stateGT == 1: # fork
                                    loop_times = args.non_normal_oversample[1]
                                elif stateGT == 2: # merge
                                    loop_times = args.non_normal_oversample[2]
                                elif stateGT == 3: # end
                                    loop_times = args.non_normal_oversample[3]
                                loop_count = 0
                                curROI_batch = None
                                while loop_count < loop_times:
                                    loop_count += 1
                                    position, direction = make_random_argument(position, direction, pos_x_thresh=2, pos_y_thresh=4, dire_thresh=0.05)
                                    stateGT = make_state_GT_tensor(polylineMaps, polylines, position, direction, ROI_size, imgName, show=True)
                                    if loop_times > 2 and stateGT == 0:
                                        continue
                                    stateGTs = torch.cat((stateGTs, torch.tensor([[stateGT]], dtype=stateGTs.dtype, device=stateGTs.device)),dim=0)
                                    curROI, angle = getROI(concatFeature, position, direction, state_ROI, scale=4,mode='middle')
                                    if curROI_batch is None:
                                        curROI_batch = curROI.unsqueeze(0)
                                    else:
                                        curROI_batch = torch.cat((curROI_batch, curROI.unsqueeze(0)), dim=0)

                                if curROI_batch is not None:
                                    state_out = net.stateHeader(curROI_batch)
                                    state_predict = torch.cat((state_predict, state_out), dim=0)

                    point_size = position_predict.size(0)
                    if point_size > 1:
                        loos_flag = True
                    print('[has %d points computed]'%(point_size))
                    if args.s_GT: # train DT, direction, position
                        loss_ = computeloss(positionGTs, position_predict, directionGTs, direction_predict, None, None, None, None, ROI_size)

                    else: # train DT, direction, position, state
                        loss_ = computeloss(positionGTs, position_predict, directionGTs, direction_predict, stateGTs, state_predict, None, None, ROI_size, args.gamma, args.focal_weights)
                    if point_size == 0:
                        continue
                    thisnode_loss = weight_position * loss_[1] / (loss_[-1][1].size(0) + loss_[-1][0].size(0)) + (weight_theta * loss_[2] + weight_state * loss_[3])/point_size
                    this_img_loss += thisnode_loss
                    popsition_loss += weight_position * loss_[1].item() / (loss_[-1][1].size(0) + loss_[-1][0].size(0))
                    direction_loss += weight_theta * loss_[2].item()/position_predict.size(0)
                    state_loss += weight_state * loss_[3].item()/state_predict.size(0)

            if len(fork_directions_1) > 0:
                fork_diff_loss = compute_fork_diff_loss(fork_directions_1, fork_directions_2)
                this_img_loss += fork_diff_loss * args.fork_weight

            if not args.only_DT and not loos_flag:
                continue
            this_img_loss.backward()
            optimizer.step()
            if torch.any(torch.isnan(net.globalTeatureNet.conv1.weight)):
                bp()

            train_loss.update(this_img_loss.data.item(), 1)
            train_position_loss.update(popsition_loss.data.item(), 1)
            train_direction_loss.update(direction_loss.data.item(), 1)
            train_state_loss.update(state_loss.data.item(), 1)
            train_DT_loss.update(DT_loss.data.item(), 1)

            curr_iter += 1
            writer.add_scalar('train_loss', this_img_loss.data.item(), curr_iter)
            writer.add_scalar('train_popsition_loss', popsition_loss.data.item(), curr_iter)
            writer.add_scalar('train_direction_loss', direction_loss.data.item(), curr_iter)
            writer.add_scalar('train_state_loss', state_loss.data.item(), curr_iter)
            writer.add_scalar('train_DT_loss', DT_loss.data.item(), curr_iter)
            writer.add_scalar('fork_diff_loss', fork_diff_loss.data.item(), curr_iter)
            if (i + 1) % args.print_freq == 0:
                time2 = time.time()
                time_delta = time2 - time1
                time1 = time2
                print('[epoch %d],[iter %d / %d],[train loss %.5f],[position_loss %.5f],[theta_loss %.5f],[state_loss %.5f],[DT_loss %.5f],[avg_train loss %.5f],[time %.3f s]' % (
                    epoch, i + 1, len(train_loader), this_img_loss.data.item(), popsition_loss.data.item(), direction_loss.data.item(),
                    state_loss.data.item(),  DT_loss.data.item(), train_loss.avg, time_delta))

def validate(val_loader, net, optimizer, epoch):
    net.eval()
    ROI_size = args.ROI_size
    state_ROI = [int(ROI_size[0] * args.state_ROI_ratio[0]), int(ROI_size[1] * args.state_ROI_ratio[1])]
    pos_ROI = [int(ROI_size[0] * args.pos_ROI_ratio[0]), int(ROI_size[1] * args.pos_ROI_ratio[1])]
    check_ROI = [int(ROI_size[0] * 0.5), int(ROI_size[1] * 0.5)]
    spatial_softmax = SpatialSoftmax(pos_ROI[0], pos_ROI[1], 1, args.device, temperature=None, data_format='NCHW')
    val_loss = AverageMeter()
    val_position_loss = AverageMeter()
    val_direction_loss = AverageMeter()
    val_state_loss = AverageMeter()
    val_DT_loss = AverageMeter()
    curr_iter = (epoch - 1) * len(val_loader)
    time1 = time.time()
    weight_position, weight_theta, weight_state, weight_DT= args.weights

    precision_count =0
    recall_count =0
    precision_sum =0
    recall_sum =0

    normal_state_sum = 0
    end_state_sum = 0
    fork_state_sum = 0
    merge_state_sum = 0

    normal_state_posi = 0
    end_state_posi = 0
    fork_state_posi = 0
    merge_state_posi = 0

    to_visual_datas = []
    lines_num_sum = 0
    posi_lines_num = 0
    for i, data in enumerate(val_loader):
        with torch.no_grad():
            img, polylines, DT_GT, polylineMaps, init_nodes = data
            polylines_predict = []
            polylines_label = polylines.squeeze(0)
            states_label = []
            states_predict = []

            if len(polylines) > 0:
                positionGTs, position_predict, directionGTs, direction_predict, stateGTs, state_predict, predictDT = [None for i in range(7)]
                img = img.float().cuda()
                DT_GT = DT_GT.float().cuda()
                polylineMaps = polylineMaps.cuda().float().squeeze(0)
                polylines = polylines.squeeze(0).cuda().float()
                img = img.unsqueeze(1)
                predictDT, concatFeature = net(img)

                img = img.squeeze(0).squeeze(0)  # shape H*W
                concatFeature = concatFeature.squeeze(0)  # shape C*H/4*W/4
                this_img_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
                popsition_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
                direction_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
                state_loss = torch.tensor([0.0], dtype=torch.float, device=args.device)
                predictDT = predictDT.squeeze(0)

                DT_loss_ = computeloss(positionGTs, position_predict, directionGTs, direction_predict, stateGTs, state_predict, DT_GT, predictDT, ROI_size)
                DT_loss = weight_DT * DT_loss_[4]
                this_img_loss += DT_loss
                if args.only_DT:
                    pass
                else:
                    fork_another_lines = []
                    fork_another_init_nodes = []
                    already_polylines = []
                    for id, init_node in enumerate(init_nodes):
                        if init_node["init_position"][1] < 1024 - 1.5 * ROI_size[0] and init_node["init_state"] == 1:
                            polyline = polylines[id]
                            fork_another_lines.append(polyline)
                            fork_another_init_nodes.append(init_node)

                    picked_ids = []
                    after_forkmerge_position = None
                    after_forkmerge_state = None
                    after_forkmerge_stateGT = None
                    for id, init_node in enumerate(init_nodes):
                        after_forkmerge_flag = False
                        if id < len(polylines):
                            polyline = polylines[id]
                            direction = torch.tensor(init_node["init_direction"]).to(args.device).float()
                            position = torch.tensor(init_node["init_position"]).to(args.device).float()
                            state = init_node["init_state"].to(args.device).float()

                            directionGTs = direction.unsqueeze(0)
                            positionGTs = polyline.to(args.device).float()
                            stateGTs = state.unsqueeze(0)

                            direction_predict = direction.unsqueeze(0)
                            position_predict = position.unsqueeze(0)
                            state_predict = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                            state_predict[0][int(state.item())] = 1.0
                        else:
                            if len(fork_another_init_nodes) == 0:
                                break
                            polyline, true_init_node, picked_id = find_nearest_forkline(init_node, fork_another_init_nodes, fork_another_lines)
                            if polyline is None:
                                continue
                            fork_another_lines.pop(picked_id)
                            fork_another_init_nodes.pop(picked_id)

                            directionGTs = torch.tensor(true_init_node["init_direction"]).unsqueeze(0).to(args.device)
                            positionGTs = polyline.to(args.device).float()
                            stateGTs = true_init_node["init_state"].to(args.device).float().unsqueeze(0).fill_(0)

                            direction = init_node["init_direction"]
                            position = init_node["init_position"]
                            state = init_node["init_state"].to(args.device).float()

                            direction_predict = direction.unsqueeze(0)
                            position_predict = position.unsqueeze(0)
                            state_predict = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                            state_predict[0][int(state.item())] = 1.0

                        if init_node["init_position"][1] < 1024 - 1.5 * ROI_size[0] and state == 1:
                            continue

                        direction_hidden = torch.zeros((1, 32, ROI_size[0], ROI_size[1]), dtype=torch.float,
                                                       device='cuda', requires_grad=True)
                        direction_cell = torch.zeros((1, 32, ROI_size[0], ROI_size[1]), dtype=torch.float,
                                                     device='cuda', requires_grad=True)

                        position_hidden = torch.zeros((1, 32, pos_ROI[0], pos_ROI[1]), dtype=torch.float,
                                                      device='cuda', requires_grad=True)
                        position_cell = torch.zeros((1, 32, pos_ROI[0], pos_ROI[1]), dtype=torch.float, device='cuda',
                                                    requires_grad=True)

                        state_hidden = torch.zeros((1, 32, state_ROI[0], state_ROI[1]), dtype=torch.float,
                                                   device='cuda', requires_grad=True)
                        state_cell = torch.zeros((1, 32, state_ROI[0], state_ROI[1]), dtype=torch.float, device='cuda',
                                                 requires_grad=True)

                        count = 0
                        while int(state.item()) < 2:#normal and fork
                            count = count + 1
                            if count == args.RNNlength:
                                break
                            curROI, angle = getROI(predictDT, position, direction, ROI_size, scale=4, mode='bottom')
                            state_predict_tensor = torch.zeros((4, curROI.shape[1], curROI.shape[2]), dtype=torch.float, device=curROI.device)
                            state_predict_tensor[int(state.item())] = 1.0
                            ROIFeature = torch.cat((curROI, state_predict_tensor), 0).unsqueeze(0)
                            if args.direction_net == "LSTM":
                                direction_out1, direction_out2, (direction_hidden, direction_cell) =\
                                    net.directionHeader(ROIFeature,(direction_hidden,direction_cell))
                            elif args.direction_net == "RNN":
                                direction_out1, direction_out2, direction_hidden = net.directionHeader(ROIFeature,
                                                                                                       direction_hidden)
                            if state == 0:
                                direction_out = direction_out1
                                direction_out = back_dir2ori(angle, direction_out.squeeze(0))
                            elif state == 1:
                                to_append_position = position
                                to_append_state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)
                                direction_out2_1, direction_out2_2 = direction_out2
                                direction_out2_1 = back_dir2ori(angle, direction_out2_1.squeeze(0))
                                direction_out2_2 = back_dir2ori(angle, direction_out2_2.squeeze(0))

                                next_direction = get_next_direction(position, positionGTs)
                                score1 = F.cosine_similarity(next_direction.unsqueeze(0), direction_out2_1.unsqueeze(0))
                                score2 = F.cosine_similarity(next_direction.unsqueeze(0), direction_out2_2.unsqueeze(0))
                                if score1 >= score2:
                                    direction_out = direction_out2_1
                                    to_append_direction = direction_out2_2
                                else:
                                    direction_out = direction_out2_2
                                    to_append_direction = direction_out2_1
                                init_nodes.append(
                                    {'init_position': to_append_position, 'init_direction': to_append_direction,
                                     'init_state': to_append_state})
                            if direction_out[1]>0:
                                break
                            direction = direction_out
                            # predict position
                            curROI, angle = getROI(predictDT, position, direction, pos_ROI, scale=4, mode='bottom')
                            state_predict_tensor = torch.zeros((4, curROI.shape[1], curROI.shape[2]),
                                                               dtype=torch.float, device=curROI.device)
                            state_predict_tensor[int(state.item())] = 1.0
                            ROIFeature = torch.cat((curROI, state_predict_tensor), 0).unsqueeze(0)
                            if args.pos_net == "LSTM":
                                position_out, (position_hidden, position_cell) = net.positionHeader(ROIFeature,
                                                                                (position_hidden,position_cell))
                            elif args.pos_net == "RNN":
                                position_out, position_hidden = net.positionHeader(ROIFeature, position_hidden)

                            position_out = spatial_softmax(position_out)
                            position_out = recover_points(position_out, pos_ROI)
                            position_out = back_pos_to_ori(position, position_out, angle, pos_ROI)
                            position = position_out.squeeze(0)
                            if position[0]< 0 or position[0] >= img.shape[1] or position[1]< 0 or position[1]>=img.shape[0]:
                                break
                            direction_predict = torch.cat((direction_predict, direction_out.unsqueeze(0)), dim=0)
                            if after_forkmerge_flag:
                                after_forkmerge_position = torch.cat((after_forkmerge_position, position_out), dim=0)
                            else:
                                position_predict = torch.cat((position_predict, position_out), dim=0)
                            # get direction GT
                            directionGT, closest_id, _ = make_thata_GT(position, positionGTs.unsqueeze(0))
                            directionGTs = torch.cat((directionGTs, directionGT.unsqueeze(0)), dim=0)

                            stateGT = make_state_GT_tensor(polylineMaps, polylines, position, direction, ROI_size, ratio=0.875)
                            if id >= len(polylines) and (stateGT == 1 or stateGT == 2):
                                stateGT = 0
                            # stateGTs = torch.cat((stateGTs, torch.tensor([[stateGT]], dtype=stateGTs.dtype, device=stateGTs.device)), dim=0)

                            if args.s_GT:
                                state_out = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                                state_out[0][int(stateGT)] = 1.0
                                state_predict = torch.cat((state_predict, state_out), dim=0)
                                state = torch.tensor([stateGT], dtype=stateGTs.dtype, device=stateGTs.device)
                                stateGTs = torch.cat((stateGTs, torch.tensor([[stateGT]], dtype=stateGTs.dtype, device=stateGTs.device)), dim=0)
                            else:
                                curROI, angle = getROI(concatFeature, position, direction, state_ROI, scale=4,
                                                       mode='middle')
                                if args.state_net == "RNN":
                                    state_out, state_hidden = net.stateHeader(curROI.unsqueeze(0), state_hidden)
                                elif args.state_net == "LSTM":
                                    state_out, (state_hidden, state_cell) = net.stateHeader(curROI.unsqueeze(0), (
                                    state_hidden, state_cell))
                                elif args.state_net == "CNN":
                                    state_out = net.stateHeader(curROI.unsqueeze(0))
                                elif args.state_net == "CNN_A":
                                    state_out = net.stateHeader(curROI.unsqueeze(0))

                                if after_forkmerge_flag:
                                    after_forkmerge_state = torch.cat((after_forkmerge_state, state_out), dim=0)
                                    after_forkmerge_stateGT = torch.cat((after_forkmerge_stateGT, torch.tensor([[stateGT]], dtype=stateGTs.dtype,
                                                                                 device=stateGTs.device)), dim=0)
                                else:
                                    state_predict = torch.cat((state_predict, state_out), dim=0)
                                    stateGTs = torch.cat((stateGTs, torch.tensor([[stateGT]], dtype=stateGTs.dtype,
                                                                                 device=stateGTs.device)), dim=0)
                                state = torch.tensor([state_out.argmax()], dtype=stateGTs.dtype, device=stateGTs.device)

                            if state == 1:
                                if id >= len(polylines): # already in fork branch
                                    state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)
                                    state_out_ = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                                    state_out_[0][int(state.item())] = 1.0
                                    state_predict[-1] = state_out_
                                    stateGTs[-1].fill_(0)
                                else:
                                    to_append_state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)
                                    to_add_init_node = {'init_position': position, 'init_direction': direction, 'init_state': to_append_state}
                                    polyline, true_init_node, picked_id = find_nearest_forkline(to_add_init_node, fork_another_init_nodes, fork_another_lines, thresh=17)
                                    if polyline is None or picked_id in picked_ids:
                                        state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)
                                        state_out_ = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                                        state_out_[0][int(state.item())] = 1.0
                                        state_predict[-1] = state_out_
                                        stateGTs[-1].fill_(0)
                                    else:
                                        picked_ids.append(picked_id)
                            if state == 2:
                                if id >= len(polylines): # already in fork branch
                                    state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)
                                    state_out_ = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                                    state_out_[0][int(state.item())] = 1.0
                                    state_predict[-1] = state_out_
                                else:
                                    point_already_in = check_any_point_already_in(position, direction, angle, check_ROI, already_polylines)
                                    if point_already_in:
                                        if is_continue_merge(positionGTs, position) and after_forkmerge_position is not None:
                                            position_predict = torch.cat((position_predict, after_forkmerge_position), dim=0)
                                            # b()
                                            state_predict = torch.cat((state_predict, after_forkmerge_state), dim=0)
                                            stateGTs = torch.cat((stateGTs, after_forkmerge_stateGT), dim=0)
                                        break
                                    else:
                                        # this merge point has not already been accessed, keep going
                                        state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)
                                        if stateGT == 0:
                                            state_out_ = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                                            state_out_[0][int(state.item())] = 1.0
                                            state_predict[-1] = state_out_

                                        elif stateGT == 2:
                                            if is_continue_merge(positionGTs, position):
                                                pass
                                            else:
                                                after_forkmerge_flag = True
                                                after_forkmerge_position = position.unsqueeze(0)
                                                after_forkmerge_state = state_predict[-1].unsqueeze(0)
                                                after_forkmerge_stateGT = torch.tensor([[stateGT]], dtype=stateGTs.dtype, device=stateGTs.device)


                        already_polylines.append(position_predict)
                        if after_forkmerge_position is not None and after_forkmerge_flag:
                            already_polylines.append(after_forkmerge_position)
                        print('[has %d points computed]'%(position_predict.size(0)))
                        loss_ = computeloss(positionGTs, position_predict, directionGTs, direction_predict, stateGTs,
                                            state_predict, DT_GT, predictDT, ROI_size, args.gamma,  args.focal_weights)
                        # weight_position, weight_theta, weight_state, weight_DT
                        thisnode_loss = weight_position * loss_[1] / (loss_[-1][1].size(0) + loss_[-1][0].size(0)) + (weight_theta * loss_[2] + weight_state * loss_[3])/position_predict.size(0) # + weight_DT * loss_[4]
                        this_img_loss += thisnode_loss
                        popsition_loss += weight_position * loss_[1].item()/(loss_[-1][1].size(0) + loss_[-1][0].size(0))
                        direction_loss += weight_theta * loss_[2].item()/position_predict.size(0)
                        state_loss += weight_state * loss_[3].item()/position_predict.size(0)
                        positionGTs_dense = loss_[-1][0]
                        positionpredict_dense = loss_[-1][1]
                        # post process
                        # position_predict = post_process_predict(position_predict, ROI_size, img)
                        polylines_predict.append(position_predict)
                        states_label.append(stateGTs)
                        states_predict.append(state_predict)
                        acc_data = calculate_acc(positionGTs, position_predict, 10)
                        precision_count += acc_data[0]
                        recall_count += acc_data[1]
                        precision_sum += acc_data[2]
                        recall_sum += acc_data[3]

                        acc_data_width = calculate_acc(positionGTs, position_predict, 20)
                        recall_20 = acc_data_width[1] / acc_data_width[3]
                        if recall_20 < 0.7:
                            pass
                        else:
                            posi_lines_num += 1

                        state_acc_nums = calculate_state_acc(stateGTs, state_predict)
                        normal_state_sum += state_acc_nums[0]
                        normal_state_posi += state_acc_nums[1]
                        fork_state_sum += state_acc_nums[2]
                        fork_state_posi += state_acc_nums[3]
                        merge_state_sum += state_acc_nums[4]
                        merge_state_posi += state_acc_nums[5]
                        end_state_sum += state_acc_nums[6]
                        end_state_posi += state_acc_nums[7]

                val_loss.update(this_img_loss.data.item(), 1)
                val_position_loss.update(popsition_loss.data.item(), 1)
                val_direction_loss.update(direction_loss.data.item(), 1)
                val_state_loss.update(state_loss.data.item(), 1)
                val_DT_loss.update(DT_loss.data.item(), 1)
                curr_iter += 1
                writer.add_scalar('val_popsition_loss', popsition_loss.data.item(), curr_iter)
                writer.add_scalar('val_direction_loss', direction_loss.data.item(), curr_iter)
                writer.add_scalar('val_state_loss', state_loss.data.item(), curr_iter)
                writer.add_scalar('val_DT_loss', DT_loss.data.item(), curr_iter)
                if (i + 1) % args.print_freq == 0:
                    time2 = time.time()
                    time_delta = time2 - time1
                    time1 = time2
                    print('[epoch %d],[iter %d / %d],[valid loss %.5f],[position_loss %.5f],[theta_loss %.5f],[state_loss %.5f],[DT_loss %.5f],[avg_val loss %.5f],[time %.3f s]' % (
                        epoch, i + 1, len(val_loader), this_img_loss.data.item(), popsition_loss.data.item(), direction_loss.data.item(),
                        state_loss.data.item(),  DT_loss.data.item(), val_loss.avg, time_delta))
        lines_num_sum += len(polylines_predict)
        if random.random() < args.visual_rate:
            to_visual_data = [polylines_label, polylines_predict, img, predictDT, DT_GT, states_label, states_predict]
            to_visual_datas.append(to_visual_data)
    if args.only_DT:
        line_acc = 0
    else:
        line_acc = posi_lines_num / lines_num_sum

    if precision_sum > 0 and recall_sum>0 and (precision_count>0 or recall_count>0):
        precision = precision_count / precision_sum
        recall = recall_count / recall_sum
        F1 = 2*(precision*recall)/(precision+recall)
    else:
        precision = 0
        recall = 0
        F1 = 0
    if args.only_DT:
        normal_acc = 0
        fork_acc = 0
        merge_acc = 0
        end_acc = 0
        state_prec = 0
    else:
        if normal_state_sum == 0:
            normal_acc = 0
        else:
            normal_acc = normal_state_posi / normal_state_sum
        if merge_state_sum == 0:
            merge_acc = 0
        else:
            merge_acc = merge_state_posi / merge_state_sum
        if fork_state_sum == 0:
            fork_acc = 0
        else:
            fork_acc = fork_state_posi / fork_state_sum
        if end_state_sum == 0:
            end_acc = 0
        else:
            end_acc = end_state_posi / end_state_sum
        state_prec = (normal_acc + fork_acc + merge_acc + end_acc) / 4
    print("normal_sum, normal_posi, fork_sum, fork_posi, merge_sum, merge_posi, end_sum, end_posi, line_acc")
    print(normal_state_sum, "  ", normal_state_posi, "  ",fork_state_sum, "  ",fork_state_posi,"  ",
          merge_state_sum, "  ", merge_state_posi, "  ", end_state_sum, "  ", end_state_posi, "  ", line_acc)
    print(
        '-----------------------------------------------------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f] [precision %.5f] [recall %.5f] [F1 %.5f] [state_prec %.5f] [line_acc %.5f]'
          %(epoch, val_loss.avg, precision, recall, F1, state_prec, line_acc))

    if F1 < 0.5:
        state_prec = 0
    if F1 > best_record['F1'] or state_prec > best_record['state_prec'] or args.debug or val_loss.avg < best_record['val_loss'] or line_acc > best_record['line_acc']:

        if F1 > best_record['F1']:
            best_record['F1'] = F1
            best_record['bset_F1_epoch'] = epoch
        if precision > best_record['precision']:
            best_record['precision'] = precision
            best_record['bset_precision_epoch'] = epoch
        if recall >best_record['recall']:
            best_record['recall'] = recall
            best_record['bset_recall_epoch'] = epoch

        if val_loss.avg < best_record['val_loss']:
            best_record['val_loss'] = val_loss.avg
            best_record['val_loss_epoch'] = epoch

        if state_prec > best_record['state_prec']:
            best_record['state_prec'] = state_prec
            best_record['state_prec_epoch'] = epoch

        if line_acc > best_record['line_acc']:
            best_record['line_acc'] = line_acc
            best_record['line_acc_epoch'] = epoch

        snapshot_name = 'epoch_%d_valloss_%.5f_F1_%.5f_prec_%.5f_recall_%.5f_state_prec_%.5f_line_acc_%.5f_normal_prec_%.5f_fork_prec_%.5f_merge_prec_%.5f_end_prec_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, F1, precision, recall, state_prec, line_acc, normal_acc, fork_acc, merge_acc, end_acc,
            optimizer.param_groups[0]['lr'])
        ckpt_file = os.path.join(args.ckpt_path, exp_name, snapshot_name + '.pth')
        opt_pth_file = os.path.join(args.ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth')
        if args.only_DT:
            ckpt_file = os.path.join(args.ckpt_path, exp_name, 'only_DT',snapshot_name + '.pth')
            opt_pth_file = os.path.join(args.ckpt_path, exp_name,'only_DT', 'opt_' + snapshot_name + '.pth')
        torch.save(net.state_dict(), ckpt_file, _use_new_zipfile_serialization=False)
        torch.save(optimizer.state_dict(), opt_pth_file,_use_new_zipfile_serialization=False)

        val_visual = []
        for vis_data in to_visual_datas:
            polylines_label, polylines_predict, img, DT_predict, DT_GT, states_label, states_predict = vis_data
            val_visual = visualize(polylines_label, polylines_predict, img, DT_predict, DT_GT, states_label, states_predict, val_visual)
        if len(to_visual_datas[0][-2]) > 0:
            val_visual = torch.stack(val_visual, 0)
            val_visual = vutils.make_grid(val_visual, nrow=5, padding=5)
            writer.add_image(snapshot_name, val_visual)
        else:
            val_visual = torch.stack(val_visual, 0)
            val_visual = vutils.make_grid(val_visual, nrow=4, padding=5)
            writer.add_image(snapshot_name, val_visual)

        print('best record: [val loss %.5f_epoch_%d], [precision %.5f_epoch_%d] [recall %.5f_epoch_%d] [F1 %.5f_epoch_%d] [state_prec %.5f_epoch_%d]' % (
            best_record['val_loss'], best_record['val_loss_epoch'],
            best_record['precision'], best_record['bset_precision_epoch'],
            best_record['recall'], best_record['bset_recall_epoch'],
            best_record['F1'], best_record['bset_F1_epoch'],
            best_record['state_prec'], best_record['state_prec_epoch']))
        print('-----------------------------------------------------------------------------------------------------------')

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('precision', precision, epoch)
    writer.add_scalar('recall', recall, epoch)
    writer.add_scalar('F1', F1, epoch)
    writer.add_scalar('state_prec', state_prec, epoch)
    writer.add_scalar('normal_acc', normal_acc, epoch)
    writer.add_scalar('fork_acc', fork_acc, epoch)
    writer.add_scalar('merge_acc', merge_acc, epoch)
    writer.add_scalar('end_acc', end_acc, epoch)
    writer.add_scalar('line_acc', line_acc, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    net.train()
    return val_loss.avg


def visualize(polylines_label, polylines_predict, img, predictDT, DT_GT, states_label, states_predict, val_visual):
    DT_GT_heat_img = make_visaul_DTmap(DT_GT)
    predictDT_heat_img = make_visaul_DTmap(predictDT)
    color_img = img.detach().cpu().numpy()
    color_img = np.asarray(colorize_mask(color_img))
    poly_label_img = copy.deepcopy(color_img)
    poly_label_img = draw_polyline_on_oriimg(polylines_label, poly_label_img)
    poly_predict_img = copy.deepcopy(color_img)
    poly_predict_img = draw_polyline_on_predictimg(polylines_predict, poly_predict_img, states_predict)
    if len(states_label) > 0:
        state_plot = plot_state(states_label, states_predict, polylines_label, polylines_predict)
        val_visual.extend([torch.from_numpy(np.transpose(DT_GT_heat_img, (2,0,1))),
                      torch.from_numpy(np.transpose(predictDT_heat_img, (2,0,1))),
                      torch.from_numpy(np.transpose(poly_label_img, (2,0,1))),
                      torch.from_numpy(np.transpose(poly_predict_img, (2,0,1))),
                      torch.from_numpy(np.transpose(cv2.resize(state_plot, (poly_predict_img.shape[1], poly_predict_img.shape[0])), (2,0,1)))])

    else:
        val_visual.extend([torch.from_numpy(np.transpose(DT_GT_heat_img, (2,0,1))),
                      torch.from_numpy(np.transpose(predictDT_heat_img, (2,0,1))),
                      torch.from_numpy(np.transpose(poly_label_img, (2,0,1))),
                      torch.from_numpy(np.transpose(poly_predict_img, (2,0,1)))])
    return val_visual

if __name__ == '__main__':
    args = init_args()
    setup_seed(args.seed)
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = os.path.join(args.exp_name, now)
    writer = SummaryWriter(os.path.join(args.ckpt_path, exp_name, 'summary'))
    main()
