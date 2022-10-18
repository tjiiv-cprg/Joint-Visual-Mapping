from ipdb import set_trace as bp
from torch.backends import cudnn
import torch
torch.set_default_tensor_type(torch.FloatTensor)
import datetime
import numpy as np
import cv2
import random

import argparse
import os

from torch.utils.data import DataLoader
from torch.nn import functional as F
from datasets import LineTopologyData
from datasets import make_DT_GT, show_DT_heatmap, make_thata_GT, make_state_GT, make_state_GT_tensor
from model.topologyRNN import TopologyRNN, NNUpsample4, SpatialSoftmax
from utils import check_mkdir, AverageMeter, getROI, computeloss, check_any_point_already_in, find_nearest_forkline, \
    calculate_acc, calculate_acc2, back_dir2ori, normalize, back_pos_to_ori, post_process_predict, recover_points, plot_state, \
    draw_polyline_on_predictimg, draw_polyline_on_oriimg, colorize_mask, make_visaul_DTmap, calculate_state_acc, plotROI, is_continue_merge

import time
import math
from tqdm import tqdm
import copy
from PIL import Image
from xml.dom.minidom import parse
from shutil import copyfile
import csv
cudnn.benchmark = True

palette = [0, 0, 0, 128, 64, 128, 157, 234, 50]

for i in range(256*3-len(palette)):
    palette.append(0)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_root', type=str, help='The test dataset root path')
    parser.add_argument('--dataset_dir', type=str, help='test data folders')
    parser.add_argument('--ckpt_root', type=str, help='checkpoint root path')
    parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint file')
    parser.add_argument('--exp_name', type=str, default='', help='experimental name')
    parser.add_argument('--diff_threshs', type=str, default='10', help='precision at which pixel')

    parser.add_argument('--device', type=str, default='cuda', help='device for train')
    parser.add_argument('--RNNlength', type=int, default=50, help='The ckpt')
    parser.add_argument('--seed', type=int, default=1, help='The random seed')
    parser.add_argument('--plot_state_ids', nargs='+', type=int, default=[])
    parser.add_argument('--ROI_size', nargs='+', type=int, default=[64, 32], help='the size of ROI')
    parser.add_argument('--state_net', type=str, default='CNN')
    parser.add_argument('--pos_net', type=str, default='RNN')
    parser.add_argument('--direction_net', type=str, default='RNN')
    parser.add_argument('--result_file', type=str, default='result.xml',  help='the result file')
    parser.add_argument('--posmax', type=str, default='softmax', help='position valid use argmax or softmax')
    parser.add_argument('--s_GT', action='store_true', default=True)
    parser.add_argument('--withpost', action='store_true', default=False)


    return parser.parse_args()

def main():

    net = TopologyRNN().cuda()
    print('training resumes from' + os.path.join(args.ckpt_path))
    pretrained_model = torch.load(args.ckpt_path)

    modle_dict = net.state_dict()
    DT_globale_feature_pretrained_modle_dict = {k: v for k, v in pretrained_model.items() if
                                                "globalTeatureNet" in k or "DTNet" in k or "directionHeader" in k or "positionHeader" in k}
    modle_dict.update(DT_globale_feature_pretrained_modle_dict)
    net.load_state_dict(modle_dict)

    net = net.float()
    net.eval()
    ROI_size = args.ROI_size

    if ',' in args.dataset_dir:
        dataset_dirs = args.dataset_dir.split(',')
        data_paths = [os.path.join(args.dataset_root, dataset_dir) for dataset_dir in dataset_dirs]
    else:
        data_paths = [os.path.join(args.dataset_root, args.dataset_dir)]
    if ',' in args.diff_threshs:
        diff_threshs = args.diff_threshs.split(',')
    else:
        diff_threshs = list()
        diff_threshs.append(int(args.diff_threshs))

    check_mkdir(os.path.join(work_dir, "mybest_result_img"))

    data_path = data_paths[0]
    with open(os.path.join(data_path, 'eval_reault_orimanner.csv'), "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["roadID", "precision_count", "precision_sum", "recall_count", "recall_sum", "precision", "reacll", "F1",
             "normal_acc", "fork_acc", "merge_acc", "end_acc", "acc_total", "acc_mean", "lane_acc"])
        first_save = True
        for diff_thresh in diff_threshs:
            writer.writerow(["dis_thresh = ", diff_thresh])
            total_03_precision_count = 0
            total_03_precision_sum = 0
            total_03_recall_count = 0
            total_03_recall_sum = 0
            for i in range(1):

                dataset_dir = data_path
                print("Now processing: ", dataset_dir)
                # check_mkdir(os.path.join(dataset_dir, "orimanner_result_img"))
                # check_mkdir(os.path.join(args.dataset_root, "mybest_result_img"))
                test_set = LineTopologyData(datadir=args.dataset_root, ROI_size=ROI_size, folders=dataset_dirs, mode='test', sample_ratio=1)
                # test_set = LineTopologyData(datadir=data_path, ROI_size=ROI_size, mode='test', sample_ratio=2)
                test_loader = DataLoader(test_set, batch_size=1, num_workers=8, shuffle=True)
                precision, recall, F1, precision_count, precision_sum, recall_count, recall_sum, state_acc = test(test_loader, net,
                                                                                                       dataset_dir,
                                                                                                       diff_thresh,
                                                                                                       first_save)
                (normal_acc, fork_acc, merge_acc, end_acc, acc_total, acc_mean, lane_acc) = state_acc
                print("normal_acc = %.5f  fork_acc = %.5f  merge_acc = %.5f  end_acc = %.5f  acc_total = %.5f  "
                      "acc_mean = %.5f  lane_acc  = %.5f" % (normal_acc, fork_acc, merge_acc, end_acc, acc_total, acc_mean, lane_acc))
                print("precision = %.5f    recall = %.5f   F1 = %.5f " % (precision, recall, F1))
                writer.writerow([i, precision_count, precision_sum, recall_count, recall_sum, precision, recall, F1,
                                 normal_acc, fork_acc, merge_acc, end_acc, acc_total, acc_mean, lane_acc])
                total_03_precision_count += precision_count
                total_03_precision_sum += precision_sum
                total_03_recall_count += recall_count
                total_03_recall_sum += recall_sum
            total_03_precision = total_03_precision_count / total_03_precision_sum
            total_03_recall = total_03_recall_count / total_03_recall_sum
            total_03_F1 = 2 * (total_03_precision * total_03_recall) / (total_03_precision + total_03_recall)
            writer.writerow(["total03", total_03_precision_count, total_03_precision_sum, total_03_recall_count,
                             total_03_recall_sum, total_03_precision, total_03_recall, total_03_F1])
        writer.writerow([])

def test(test_loader, net, dataset_dir, diff_thresh, first_save):
    ROI_size = args.ROI_size

    pos_ROI = [int(ROI_size[0] * 0.5), int(ROI_size[1] * 1)]
    check_ROI = [int(ROI_size[0] * 0.5), int(ROI_size[1] * 0.5)]

    spatial_softmax = SpatialSoftmax(pos_ROI[0], pos_ROI[1], 1, args.device, temperature=None, data_format='NCHW')


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

    # acc_lane
    posi_lines_num = 0
    lines_num_sum = 0

    to_acc_positionGTs = []
    to_acc_position_predict = []

    for i, data in enumerate(test_loader):
        with torch.no_grad():
            img, polylines, DT_GT, polylineMaps, init_nodes, img_name = data
            print(img_name)
            # bp()
            # N = img.size(0)
            polylines_predict = []
            polylines_label = polylines.squeeze(0)
            states_label = []
            states_predict = []
            ROI_params = []

            if len(polylines) > 0:
                img = img.float().cuda()
                DT_GT = DT_GT.float().cuda()

                polylineMaps = polylineMaps.cuda().float().squeeze(0)
                polylines = polylines.squeeze(0).cuda().float()
                img = img.unsqueeze(1)

                predictDT, concatFeature = net(img)

                img = img.squeeze(0).squeeze(0)  # shape H*W
                concatFeature = concatFeature.squeeze(0)  # shape C*H/4*W/4
                predictDT = predictDT.squeeze(0)
                fork_another_lines = []
                fork_another_init_nodes = []
                already_polylines = []
                to_acc_position_predict = []

                for id, init_node in enumerate(init_nodes):
                    if id < len(polylines):
                        polyline = polylines[id]
                    else:
                        if len(fork_another_init_nodes) == 0:
                            break
                        polyline, true_init_node, pick_id = find_nearest_forkline(init_node, fork_another_init_nodes,
                                                                         fork_another_lines)
                        if polyline is None:
                            continue
                    direction = torch.tensor(init_node["init_direction"])
                    direction_norm = direction.norm(dim=0)
                    direction = (direction / direction_norm).to(args.device).float()
                    position = torch.tensor(init_node["init_position"]).to(args.device).float()
                    state = init_node["init_state"].to(args.device).float()

                    directionGTs = direction.unsqueeze(0)
                    positionGTs = polyline.to(args.device).float()
                    stateGTs = state.unsqueeze(0)

                    direction_predict = direction.unsqueeze(0)
                    position_predict = position.unsqueeze(0)
                    state_predict = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                    state_predict[0][int(state.item())] = 1.0

                    direction_hidden = torch.zeros((1, 32, ROI_size[0], ROI_size[1]), dtype=torch.float,
                                                   device='cuda', requires_grad=True)
                    direction_cell = torch.zeros((1, 32, ROI_size[0], ROI_size[1]), dtype=torch.float,
                                                 device='cuda', requires_grad=True)

                    position_hidden = torch.zeros((1, 32, pos_ROI[0], pos_ROI[1]), dtype=torch.float,
                                                  device='cuda', requires_grad=True)
                    position_cell = torch.zeros((1, 32, pos_ROI[0], pos_ROI[1]), dtype=torch.float, device='cuda',
                                                requires_grad=True)

                    state_hidden = torch.zeros((1, 32, ROI_size[0], ROI_size[1]), dtype=torch.float,
                                               device='cuda', requires_grad=True)
                    state_cell = torch.zeros((1, 32, ROI_size[0], ROI_size[1]), dtype=torch.float, device='cuda',
                                             requires_grad=True)

                    count = 0
                    while int(state.item()) <2:
                        #b()
                        count = count + 1
                        if count == args.RNNlength:
                            break

                        # predict direction
                        curROI, angle = getROI(predictDT, position, direction, ROI_size, scale=4, mode='bottom')

                        state_predict_tensor = torch.zeros((4, curROI.shape[1], curROI.shape[2]), dtype=torch.float,
                                                           device=curROI.device)
                        state_predict_tensor[int(state.item())] = 1.0

                        ROIFeature = torch.cat((curROI, state_predict_tensor), 0).unsqueeze(0)

                        if args.direction_net == "LSTM":
                            direction_out1, direction_out2, (direction_hidden, direction_cell) = \
                                net.directionHeader(ROIFeature, (direction_hidden, direction_cell))

                        elif args.direction_net == "RNN":
                            direction_out1, direction_out2, direction_hidden = net.directionHeader(ROIFeature, direction_hidden)
                        if state == 0:
                            direction_out = direction_out1
                            direction_out = back_dir2ori(angle, direction_out.squeeze(0))
                        elif state == 1:
                            to_append_position = position
                            to_append_state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)

                            direction_out2_1, direction_out2_2 = direction_out2
                            direction_out2_1 = back_dir2ori(angle, direction_out2_1.squeeze(0))
                            direction_out2_2 = back_dir2ori(angle, direction_out2_2.squeeze(0))
                            # b()
                            score1 = F.cosine_similarity(direction.unsqueeze(0), direction_out2_1.unsqueeze(0))
                            score2 = F.cosine_similarity(direction.unsqueeze(0), direction_out2_2.unsqueeze(0))
                            if score1 >= score2:
                                direction_out = direction_out2_1
                                to_append_direction = direction_out2_2
                            else:
                                direction_out = direction_out2_2
                                to_append_direction = direction_out2_1

                            init_nodes.append(
                                {'init_position': to_append_position, 'init_direction': to_append_direction,
                                 'init_state': to_append_state})

                        if direction_out[1] > 0:
                            if img_name[0] == "3228#4510#3996#3440#1024.png" or img_name[0] == "3033#4587#3850#3353#1024.png":
                                pass
                            else:
                                break
                        direction = direction_out

                        # predict position
                        curROI, angle = getROI(predictDT, position, direction, pos_ROI, scale=4, mode='bottom')
                        state_predict_tensor = torch.zeros((4, curROI.shape[1], curROI.shape[2]),
                                                           dtype=torch.float, device=curROI.device)
                        state_predict_tensor[int(state.item())] = 1.0
                        ROIFeature = torch.cat((curROI, state_predict_tensor), 0).unsqueeze(0)
                        if args.pos_net == "LSTM":
                            position_out, (position_hidden, position_cell) = net.positionHeader(ROIFeature,(position_hidden,position_cell))
                        elif args.pos_net == "RNN":
                            position_out, position_hidden = net.positionHeader(ROIFeature, position_hidden)
                        if args.posmax == "argmax":
                            position_out_argmax_ = position_out.argmax()
                            position_out = torch.tensor(
                                [[position_out_argmax_ % pos_ROI[1], position_out_argmax_ // pos_ROI[1]]],
                                dtype=position_out.dtype, device=position_out.device)
                        elif args.posmax == "softmax":
                            position_out = spatial_softmax(position_out)
                            position_out = recover_points(position_out, pos_ROI)

                        position_out = back_pos_to_ori(position, position_out, angle, pos_ROI)
                        position = position_out.squeeze(0)
                        if count > 1 and(position[0] < 0 or position[0] > img.shape[1] or position[1] < 0 or position[1] >img.shape[0] or \
                                DT_GT[0][int(position[1].item()/4)][int(position[0].item()/4)] < 1):
                            bp()
                            break

                        direction_predict = torch.cat((direction_predict, direction_out.unsqueeze(0)), dim=0)
                        position_predict = torch.cat((position_predict, position_out), dim=0)

                        # get direction GT
                        directionGT, closest_id, _ = make_thata_GT(position, positionGTs.unsqueeze(0))

                        directionGTs = torch.cat((directionGTs, directionGT.unsqueeze(0)), dim=0)


                        stateGT = make_state_GT_tensor(polylineMaps, polylines, position, direction, check_ROI, img_name, ratio=1)
                        stateGTs = torch.cat(
                            (stateGTs, torch.tensor([[stateGT]], dtype=stateGTs.dtype, device=stateGTs.device)), dim=0)

                        # predict state
                        if args.s_GT:
                            state_out = torch.zeros((1, 4), dtype=torch.float, device=args.device)
                            state_out[0][int(stateGT)] = 1.0
                            state_predict = torch.cat((state_predict, state_out), dim=0)
                            state = torch.tensor([stateGT], dtype=stateGTs.dtype, device=stateGTs.device)
                        else:
                            curROI, angle = getROI(concatFeature, position, direction, ROI_size, scale=4, mode='middle')
                            ROIFeature = torch.cat((curROI, state_predict_tensor), 0).unsqueeze(0)

                            # state_out, state_hidden = net.stateHeader(ROIFeature, state_hidden)
                            if args.state_net == "RNN":
                                state_out, state_hidden = net.stateHeader(ROIFeature, state_hidden)
                            elif args.state_net == "LSTM":
                                state_out, (state_hidden, state_cell) = net.stateHeader(ROIFeature, (state_hidden, state_cell))
                            elif args.state_net == "CNN":
                                state_out = net.stateHeader(ROIFeature)
                            # state_out, state_hidden= net.stateHeader(ROIFeature, state_hidden)
                            state_predict = torch.cat((state_predict, state_out), dim=0)
                            state = torch.tensor([state_out.argmax()], dtype=stateGTs.dtype, device=stateGTs.device)

                        for to_plot_state in args.plot_state_ids:
                            if state == to_plot_state:
                                ROI_param = [position, direction, "middle", int(state.item())]
                                ROI_params.append(ROI_param)

                        if state == 2:

                            if is_continue_merge(positionGTs, position):
                                state = torch.tensor([0], dtype=stateGTs.dtype, device=stateGTs.device)
                            else:
                                break

                    if args.withpost:
                        position_predict = post_process_predict(position_predict, ROI_size, img, search_radius=15)
                    already_polylines.append(position_predict)
                    print('[has %d points computed]' % (position_predict.size(0)))
                    loss_ = computeloss(positionGTs, position_predict, directionGTs, direction_predict, stateGTs,
                                        state_predict, DT_GT, predictDT, ROI_size)

                    positionGTs_dense = loss_[-1][0]
                    positionpredict_dense = loss_[-1][1]

                    polylines_predict.append(position_predict)
                    states_label.append(stateGTs)
                    states_predict.append(state_predict)

                    to_acc_position_predict.append(positionpredict_dense)
                    acc_data = calculate_acc(positionGTs, position_predict, threshhold=diff_thresh, cut_end=True)
                    precision_count += acc_data[0]
                    recall_count += acc_data[1]
                    precision_sum += acc_data[2]
                    recall_sum += acc_data[3]

                    # acc_lane
                    acc_data_width = calculate_acc(positionGTs, position_predict, 20)
                    recall_20 = acc_data_width[1] / acc_data_width[3]
                    if recall_20 < 0.7:
                        pass
                    else:
                        posi_lines_num += 1

                    state_acc_nums = calculate_state_acc(stateGTs, state_predict)

                    # state_acc calculate
                    normal_state_sum += state_acc_nums[0]
                    normal_state_posi += state_acc_nums[1]
                    fork_state_sum += state_acc_nums[2]
                    fork_state_posi += state_acc_nums[3]
                    merge_state_sum += state_acc_nums[4]
                    merge_state_posi += state_acc_nums[5]
                    end_state_sum += state_acc_nums[6]
                    end_state_posi += state_acc_nums[7]


                if first_save:
                    to_save_data = [polylines_label, to_acc_position_predict, img, predictDT, DT_GT, states_label, states_predict, ROI_params, img_name[0]]
                    save_result(to_save_data, dataset_dir)

        lines_num_sum += len(polylines_predict)

    if precision_sum > 0 and recall_sum>0:
        precision = precision_count / precision_sum
        recall = recall_count / recall_sum
        F1 = 2*(precision*recall)/(precision+recall)
    else:
        precision = 0
        recall = 0
        F1 = 0
    if normal_state_sum == 0:
        normal_acc = 1
    else:
        normal_acc = normal_state_posi / normal_state_sum

    if merge_state_sum == 0:
        merge_acc = 1
    else:
        merge_acc = merge_state_posi / merge_state_sum

    if fork_state_sum == 0:
        fork_acc = 1
    else:
        fork_acc = fork_state_posi / fork_state_sum

    if end_state_sum == 0:
        end_acc = 1
    else:
        end_acc = end_state_posi / end_state_sum

    acc_total = (normal_state_posi + fork_state_posi + merge_state_posi + end_state_posi) / (normal_state_sum + fork_state_sum + merge_state_sum + end_state_sum)
    acc_mean = (normal_acc + fork_acc + merge_acc + end_acc) / 4
    lane_acc = posi_lines_num / lines_num_sum
    state_acc = (normal_acc, fork_acc, merge_acc, end_acc, acc_total, acc_mean, lane_acc)

    return precision, recall, F1, precision_count, precision_sum, recall_count, recall_sum, state_acc


def save_result(to_save_data, dataset_dir):

    polylines_label, polylines_predict, img, predictDT, DT_GT, states_label, states_predict, ROI_params, img_name = to_save_data
    DT_GT_heat_img = make_visaul_DTmap(DT_GT)
    predictDT_heat_img = make_visaul_DTmap(predictDT)

    color_img = img.detach().cpu().numpy()
    color_img = np.asarray(colorize_mask(color_img))

    poly_label_img = copy.deepcopy(color_img)
    poly_label_img = draw_polyline_on_oriimg(polylines_label, poly_label_img, states_label)

    poly_predict_img = copy.deepcopy(color_img)
    poly_predict_img = draw_polyline_on_predictimg(polylines_predict, poly_predict_img, states_predict)

    poly_predict_img = plotROI(ROI_params, args.ROI_size, poly_predict_img)

    poly_predict_img_withGTstate = copy.deepcopy(color_img)
    poly_predict_img_withGTstate = draw_polyline_on_predictimg(polylines_predict, poly_predict_img_withGTstate, states_label)

    state_plot = plot_state(states_label, states_predict, polylines_label, polylines_predict)
    state_plot = cv2.resize(state_plot, (poly_predict_img.shape[1], poly_predict_img.shape[0]))
    res = torch.cat((torch.tensor(DT_GT_heat_img),torch.tensor(predictDT_heat_img),torch.tensor(poly_label_img),
                     torch.tensor(poly_predict_img), torch.tensor(poly_predict_img_withGTstate), torch.tensor(state_plot)), dim=1)
    res = res.cpu().numpy()
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(work_dir, 'mybest_result_img', img_name.replace('.png', '_res.png')), res)
    write_xml_reslut(polylines_predict, img_name, dataset_dir)


def write_xml_reslut(polylines_predict, img_name, dataset_dir):

    ori_anno_file = os.path.join(dataset_dir, "annotations.xml")
    result_file_name = os.path.join(dataset_dir, args.result_file)
    if not os.path.exists(result_file_name):
        copyfile(ori_anno_file, result_file_name)

    domTree = parse(result_file_name)
    rootNode = domTree.documentElement
    imagesNodes = rootNode.getElementsByTagName("image")
    for imageNode in imagesNodes:
        imagename = imageNode.getAttribute("name")
        if not img_name == imagename:
            continue

        polylines = imageNode.getElementsByTagName("polyline")
        for id, polyline in enumerate(polylines):
            polyline.setAttribute('source', 'net_predict')
            points_reult_str = ""
            if id < len(polylines_predict):
                for ii, point in enumerate(polylines_predict[id]):
                    points_reult_str += "%.2f,%.2f"%(point[0], point[1])
                    if ii == len(polylines_predict[id]) - 1:
                        pass
                    else:
                        points_reult_str += ";"
            else:
                print("One line has not been predicted")
            polyline.setAttribute('points', points_reult_str)
    try:
        with open(result_file_name,'w',encoding='UTF-8') as fh:
            domTree.writexml(fh,indent='',encoding='UTF-8')
            print('Write OK')
    except Exception as err:
        print('Error：{err}'.format(err=err))


if __name__ == '__main__':
    args = init_args()
    work_dir = os.path.join(args.ckpt_root, args.exp_name, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    check_mkdir(os.path.join(args.ckpt_root, args.exp_name))
    os.mkdir(os.path.join(work_dir))
    with open(os.path.join(work_dir, str(datetime.datetime.now()) + '.txt'), 'w')as file:
        file.write(str(args) + '\n\n')
    setup_seed(args.seed)
    main()
