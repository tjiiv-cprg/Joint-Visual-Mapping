import os
import torch
from torch.nn import functional as F
import math
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from ipdb import set_trace as bp
import numpy as np
import cv2
from torch.autograd import Variable
import copy
import time
import random
from io import StringIO, BytesIO

ROI_colors = [(38, 70, 255), (7, 246, 23), (248, 20, 32), (34, 1, 0)] # blue green red gray
# ROI_colors = [(255, 70, 38), (7, 246, 23), (248, 20, 32), (34, 1, 0)]

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

class AverageMeter(object):
    def __init__(self):
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

def normalize(input):
    if torch.is_tensor(input):
        input_norm = input.norm(dim=0)
        output = (input/input_norm)
        return output  # torch.tensor
    if type(input) is np.ndarray:
        pass

    if isinstance(input, list):
        input = np.array(input)
    input_norm = np.linalg.norm(input)
    output = input / input_norm
    return output  # np.ndarray


def back_dir2ori(ROI_angle, direction):
    dx = direction[0]*math.cos(ROI_angle) - direction[1]*math.sin(ROI_angle)
    dy = -1*direction[1]*math.cos(ROI_angle) - direction[0]*math.sin(ROI_angle)
    direction[0] = dx
    direction[1] = -1 * dy
    return normalize(direction)

def back_dir2ori_wrong(ROI_angle, direction):
    dx = direction[0]*math.cos(ROI_angle) + direction[1]*math.sin(ROI_angle)
    dy = direction[1]*math.cos(ROI_angle) - direction[0]*math.sin(ROI_angle)
    direction[0] = dx
    direction[1] = dy
    return normalize(direction)

def recover_points(point, ROI_size):
    """

    :param point: from SpatialSoftmax[x,y] x in width, y in height
    :param ROI_size:
    :return:
    """
    if not (-1-1e-5 <= point[0][0] <= 1+1e-5 and -1-1e-5 <= point[0][1] <= 1+1e-5):
        bp()
    point[0][0] = (point[0][0] + 1) * (ROI_size[1]-1)/2
    point[0][1] = (point[0][1] + 1) * (ROI_size[0]-1)/2
    try:
        0 <= point[0][0] < ROI_size[1] and 0 <= point[0][1] < ROI_size[0]
    except:
        bp()
    return point


def back_pos_to_ori(pre_position, cur_position, angle, ROI_size):

    t1 = pre_position[0] + ROI_size[0]*math.sin(angle) - ROI_size[1]/2*math.cos(angle)
    t2 = pre_position[1] - ROI_size[0]*math.cos(angle) - ROI_size[1]/2*math.sin(angle)
    M = torch.tensor([
        [math.cos(angle),-math.sin(angle),t1],
        [math.sin(angle),math.cos(angle) ,t2]
        ], dtype=torch.float, device=cur_position.device)
    cur_position = torch.cat((cur_position, torch.tensor([[1.0]], device=cur_position.device)), dim=1)
    cur_position = cur_position.transpose(0,1)
    position_out_ = torch.mm(M, cur_position)
    position_out = position_out_.permute(1,0)
    return position_out



def getROI(inputtensor, ROI_anchor, direction, ROI_size, scale=4, mode='bottom'):
    """

    :param inputtensor: shape C*H*W
    :param ROI_anchor: tensor[x,y]
    :param direction: tensor[d_x, d_y] norm=1
    :param ROI_size: [h, w] in origin scale
    :param scale: downsample scale
    :param mode: middle or bottom
    :return:
    """
    if mode == "middle":
        ratio = ROI_size[0] * 0.5

        ROI_anchor = ROI_anchor - ratio*direction
    ROI_size = [int(ROI_size[i]/scale) for i in range(len(ROI_size))]

    max_len = round((ROI_size[0]**2 + ROI_size[1]**2) ** 0.5)

    ROI_anchor = [int(ROI_anchor[i].item()/scale) for i in range(len(ROI_anchor))]
    upborder = ROI_anchor[1] - max_len
    bottomborder = ROI_anchor[1] + max_len # including
    leftborder = ROI_anchor[0] - max_len
    rightborder = ROI_anchor[0] + max_len
    if upborder >=0 and bottomborder < inputtensor.shape[1] and leftborder >=0 and rightborder < inputtensor.shape[2]:
        crop_tensor = inputtensor[:, upborder: bottomborder+1, leftborder : rightborder+1]

    elif upborder < 0:
        if leftborder < 0:
            crop_tensor = inputtensor[:, 0: bottomborder+1, 0 : rightborder+1]
            tocat_tensor_up = torch.zeros((crop_tensor.shape[0], 2*max_len + 1 -crop_tensor.shape[1], crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((tocat_tensor_up, crop_tensor), dim=1)
            tocat_tensor_left = torch.zeros((crop_tensor.shape[0], crop_tensor.shape[1], 2*max_len + 1-crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((tocat_tensor_left, crop_tensor), dim=2)
        elif rightborder >= inputtensor.shape[2]:
            crop_tensor = inputtensor[:, 0: bottomborder+1, leftborder : inputtensor.shape[2]]
            tocat_tensor_up = torch.zeros((crop_tensor.shape[0], 2*max_len + 1 -crop_tensor.shape[1], crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((tocat_tensor_up, crop_tensor), dim=1)
            tocat_tensor_right = torch.zeros((crop_tensor.shape[0], crop_tensor.shape[1], 2*max_len + 1-crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((crop_tensor, tocat_tensor_right), dim=2)
        else:
            crop_tensor = inputtensor[:, 0: bottomborder+1, leftborder : rightborder+1]
            tocat_tensor_up = torch.zeros((crop_tensor.shape[0], 2*max_len + 1 -crop_tensor.shape[1], crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((tocat_tensor_up, crop_tensor), dim=1)

    elif bottomborder >= inputtensor.shape[1]:
        if leftborder < 0:
            crop_tensor = inputtensor[:, upborder: inputtensor.shape[1], 0 : rightborder+1]
            tocat_tensor_bottom = torch.zeros((crop_tensor.shape[0], 2*max_len + 1 -crop_tensor.shape[1], crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((crop_tensor, tocat_tensor_bottom), dim=1)
            tocat_tensor_left = torch.zeros((crop_tensor.shape[0], crop_tensor.shape[1], 2*max_len + 1-crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((tocat_tensor_left, crop_tensor), dim=2)
        elif rightborder >= inputtensor.shape[2]:
            crop_tensor = inputtensor[:, upborder: inputtensor.shape[1], leftborder : inputtensor.shape[2]]
            tocat_tensor_bottom = torch.zeros((crop_tensor.shape[0], 2*max_len + 1 -crop_tensor.shape[1], crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((crop_tensor, tocat_tensor_bottom), dim=1)
            tocat_tensor_right = torch.zeros((crop_tensor.shape[0], crop_tensor.shape[1], 2*max_len + 1-crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((crop_tensor, tocat_tensor_right), dim=2)
        else:
            crop_tensor = inputtensor[:, upborder: inputtensor.shape[1], leftborder : rightborder+1]
            tocat_tensor_bottom = torch.zeros((crop_tensor.shape[0], 2*max_len + 1 -crop_tensor.shape[1], crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
            crop_tensor = torch.cat((crop_tensor, tocat_tensor_bottom), dim=1)

    elif leftborder < 0:
        crop_tensor = inputtensor[:, upborder: bottomborder+1, 0 : rightborder+1]
        tocat_tensor_left = torch.zeros((crop_tensor.shape[0], crop_tensor.shape[1], 2*max_len + 1 -crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
        crop_tensor = torch.cat((tocat_tensor_left, crop_tensor), dim=2)

    elif rightborder >= inputtensor.shape[2]:
        crop_tensor = inputtensor[:, upborder: bottomborder+1, leftborder : inputtensor.shape[2]]
        tocat_tensor_right = torch.zeros((crop_tensor.shape[0], crop_tensor.shape[1], 2*max_len + 1 -crop_tensor.shape[2]), dtype=torch.float, device=crop_tensor.device)
        crop_tensor = torch.cat((crop_tensor, tocat_tensor_right), dim=2)

    try:
        crop_tensor.size(1) == crop_tensor.size(2) == 2*max_len+1
    except:
        bp()
    # OK!!
    if direction[0] == 0:
        angle = math.pi / 2
    else:
        angle = math.atan((-1 * direction[1]) / (direction[0]))
    if direction[0] < 0:
        angle = math.pi + angle

    angle = math.pi/2 - angle

    theta = torch.tensor([[math.cos(angle), math.sin(-angle), 0],
                          [math.sin(angle), math.cos(angle), 0]], dtype=torch.float, device=crop_tensor.device)

    grid = F.affine_grid(theta.unsqueeze(0), crop_tensor.unsqueeze(0).size(), align_corners=False).to(crop_tensor.device)
    output = F.grid_sample(crop_tensor.unsqueeze(0), grid, align_corners=False)
    # now take one more right pixel, tobi better
    ROI = output[:, :, max_len-(ROI_size[0]-1): max_len+1, max_len - int((ROI_size[1]-1)/2): max_len + int((ROI_size[1]-1)/2)+2]
    ROI = ROI.squeeze(0)

    return ROI, angle


def compute_class_weights(histogram):
    classWeights = np.ones(4, dtype=np.float32)
    normHist = histogram / np.sum(histogram)
    for i in range(4):
        classWeights[i] = 1 / (np.log(1.10 + normHist[i]))
    return classWeights


def focal_loss_my(input,target, gamma=2, focal_weights=False):
    '''
    :param input: shape [batch_size,num_classes,H,W] 仅仅经过卷积操作后的输出，并没有经过任何激活函数的作用
    :param target: shape [batch_size,H,W]
    :return:
    '''
    n, c, h, w = input.size()
    target = target.long()
    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()
    number_3 = torch.sum(target == 3).item()

    frequency = torch.tensor((number_0, number_1, number_2, number_3), dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)

    weights=torch.from_numpy(classWeights).float().cuda()
    focal_frequency = F.nll_loss(F.softmax(input, dim=1), target, reduction='none')

    focal_frequency += 1.0 # shape [num_samples] 1-P（gt_classes）

    focal_frequency = torch.pow(focal_frequency, gamma) # torch.Size([75])
    focal_frequency = focal_frequency.repeat(c, 1)

    focal_frequency = focal_frequency.transpose(1, 0)
    if focal_weights:
        loss = F.nll_loss(focal_frequency * (F.log_softmax(input, dim=1)), target, weight=weights, reduction='sum')
        return loss / weights.sum()
    else:
        loss = F.nll_loss(focal_frequency * (F.log_softmax(input, dim=1)), target, weight=None, reduction='sum')
        return loss


def focal_loss_zhihu(input, target, gamma):
    n, c, h, w = input.size()

    target = target.long()
    inputs = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.contiguous().view(-1)

    N = inputs.size(0)
    C = inputs.size(1)

    number_0 = torch.sum(target == 0).item()
    number_1 = torch.sum(target == 1).item()
    number_2 = torch.sum(target == 2).item()
    number_3 = torch.sum(target == 3).item()

    frequency = torch.tensor((number_0, number_1, number_2, number_3), dtype=torch.float32)
    frequency = frequency.numpy()
    classWeights = compute_class_weights(frequency)

    weights = torch.from_numpy(classWeights).float()
    # bp()
    weights=weights[target.view(-1)].float().cuda()
    P = F.softmax(inputs, dim=1) # shape [num_samples,num_classes]

    class_mask = inputs.data.new(N, C).fill_(0)
    class_mask = Variable(class_mask)
    ids = target.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.) # shape [num_samples,num_classes]  one-hot encoding

    probs = (P * class_mask).sum(1).view(-1, 1) # shape [num_samples,]
    log_p = probs.log()

    print('in calculating batch_loss',weights.shape,probs.shape,log_p.shape)

    batch_loss = -weights * (torch.pow((1 - probs), gamma)) * log_p

    print(batch_loss.shape)

    loss = batch_loss.mean()
    return loss

def oversample_polyline(polyline, gap):
    i = 0
    pointnum = len(polyline) - 1
    while polyline[pointnum][0] == -1:
        polyline = polyline[:pointnum]
        pointnum = pointnum - 1

    end_point = polyline[pointnum]
    while 1:
        point = polyline[i]
        if i == len(polyline) - 1:
            break
        point_ = polyline[i + 1]
        dis = F.pairwise_distance(point.unsqueeze(0), point_.unsqueeze(0), p=2).item()
        if dis >= gap:
            to_add_point_num = int(dis // gap)
            x_gap = (point_[0] - point[0]) / dis * gap
            y_gap = (point_[1] - point[1]) / dis * gap
            for j in range(to_add_point_num):
                new_point = torch.tensor([[point[0] + (j + 1) * x_gap, point[1] + (j + 1) * y_gap]],
                                         dtype=polyline.dtype, device=polyline.device)
                polyline = torch.cat((polyline[:i+j+1], new_point, polyline[i+j+1:]), dim=0)
            i = i + to_add_point_num
            polyline = torch.cat((polyline[:i + 1], polyline[i + 2:]), dim=0)
        else:
            polyline = torch.cat((polyline[:i + 1], polyline[i + 2:]), dim=0)
    polyline = torch.cat((polyline[:], end_point.unsqueeze(0)), dim=0)
    return polyline


def position_loss(positionGTs, position_predict, ROI_size):
    gap = ROI_size[0] # pixels
    positionGTs = oversample_polyline(positionGTs, gap)
    position_predict = oversample_polyline(position_predict, gap)
    loss = torch.tensor([0.0], dtype=torch.float, device=positionGTs.device)

    m = len(positionGTs)
    n = len(position_predict)
    distance = torch.zeros((m, n), dtype=loss.dtype, device=loss.device).fill_(1000000)
    for m_, point_GT in enumerate(positionGTs):
        if point_GT[0] == -1:
            break
        for n_, point_predict in enumerate(position_predict):
            dis = F.pairwise_distance(point_GT.unsqueeze(0), point_predict.unsqueeze(0), p=2)
            distance[m_][n_] = dis

    for m_ in range(m):
        if positionGTs[m_][0] == -1:
            break
        loss = loss + distance[m_, :].min()
    for n_ in range(n):
        loss = loss + distance[:, n_].min()
    loss = loss / (m+n)

    return loss, positionGTs, position_predict

def compute_fork_diff_loss(fork_directions_1, fork_directions_2):
    loss_fork_diff = torch.tensor([0.0], dtype=torch.float, device='cuda')
    for i in range(len(fork_directions_1)):
        loss_fork_diff = loss_fork_diff + F.cosine_similarity(fork_directions_1[i].unsqueeze(0), fork_directions_2[i].unsqueeze(0))
    return loss_fork_diff

def computeloss(positionGTs, position_predict, directionGTs, direction_predict, stateGTs, state_predict, DT_GT, predictDT, ROI_size=[64,32], gamma=2,  focal_weights = False):
    loss_DT = torch.tensor([0.0], dtype=torch.float, device='cuda')
    loss_theta =torch.tensor([0.0], dtype=torch.float, device='cuda')
    loss_state = torch.tensor([0.0], dtype=torch.float, device='cuda')
    loss_position = torch.tensor([0.0], dtype=torch.float, device='cuda')
    positionGTs_ = None
    position_predict_ = None
    if DT_GT is not None and predictDT is not None:
        L2loos = torch.nn.MSELoss()
        loss_DT = L2loos(predictDT, DT_GT)
    if directionGTs is not None and direction_predict is not None:
        assert directionGTs.shape == direction_predict.shape
        num_nodes = len(directionGTs)
        for i in range(num_nodes):
            loss_theta = loss_theta + (1 - F.cosine_similarity(directionGTs[i].unsqueeze(0), direction_predict[i].unsqueeze(0)))

    if stateGTs is not None and state_predict is not None:
        assert stateGTs.shape[0] == state_predict.shape[0]
        loss_state = focal_loss_my(state_predict.permute(1,0).unsqueeze(0).unsqueeze(-1), stateGTs.view(1,1,-1,1), gamma, focal_weights)

    if positionGTs is not None and position_predict is not None:
        loss_position, positionGTs_, position_predict_ = position_loss(positionGTs, position_predict, ROI_size)

    loss_total = loss_position + loss_DT + loss_state + loss_theta
    # weight_position, weight_theta, weight_state, weight_DT
    return loss_total, loss_position, loss_theta, loss_state, loss_DT, [positionGTs_, position_predict_]


def calculate_acc(positionGTs_dense, positionpredict_dense, threshhold=10, cut_end=False):
    if positionpredict_dense is None:
        return 0, 0, 1, 1

    precision_count = 0
    recall_count = 0

    precision_sum = len(positionpredict_dense)
    recall_sum = 0

    for predict_point in positionpredict_dense:
        point_num = len(positionGTs_dense)
        dis_min = 100000
        for point_id in range(point_num-1):
            if positionGTs_dense[point_id][0] < 0:
                break
            point1 = positionGTs_dense[point_id]
            point2 = positionGTs_dense[point_id+1]
            vector1 = point2 - point1
            vector2 = predict_point - point1
            vector3 = predict_point - point2
            if vector1.dot(vector2) < 0 or (-1*vector1).dot(vector3) < 0:
                continue
            dis = abs(vector1[0] * vector2[1] - vector1[1]*vector2[0]) / vector1.norm(dim=0)
            if dis < dis_min:
                dis_min = dis
            if dis_min < threshhold:
                precision_count = precision_count +1
                break

    if cut_end:
        pointnum = len(positionGTs_dense) - 1
        while positionGTs_dense[pointnum][0] == -1:
            positionGTs_dense = positionGTs_dense[:pointnum]
            pointnum = pointnum - 1
        positionGTs_dense = positionGTs_dense[: -4]

    for DT_point in positionGTs_dense:
        point_num = len(positionpredict_dense)
        dis_min = 100000
        if DT_point[0] < 0:
            break
        recall_sum = recall_sum + 1
        for point_id in range(point_num-1):
            point1 = positionpredict_dense[point_id]
            point2 = positionpredict_dense[point_id+1]
            vector1 = point2 - point1
            vector2 = DT_point - point1
            vector3 = DT_point - point2
            if vector1.dot(vector2) < 0 or (-1*vector1).dot(vector3) < 0:
                continue
            dis = abs(vector1[0] * vector2[1] - vector1[1]*vector2[0]) / vector1.norm(dim=0)
            if dis < dis_min:
                dis_min = dis
            if dis_min < threshhold:
                recall_count = recall_count +1
                break
    return precision_count, recall_count, precision_sum, recall_sum


def calculate_acc2(positionGTs, positionpredicts, threshhold=10):
    if positionpredicts is None:
        return 0, 0, 1, 1

    precision_count = 0
    recall_count = 0
    precision_sum = 0
    recall_sum = 0

    for predicts in positionpredicts:
        precision_sum += len(predicts)
        for predict_point in predicts:
            dis_min = 100000
            found =False
            for GT in positionGTs:
                if found:
                    break
                point_num = len(GT)
                for point_id in range(point_num-1):
                    if GT[point_id][0] < 0:
                        break
                    point1 = GT[point_id]
                    point2 = GT[point_id+1]
                    vector1 = point2 - point1
                    vector2 = predict_point - point1
                    vector3 = predict_point - point2
                    if vector1.dot(vector2) < 0 or (-1*vector1).dot(vector3) < 0:
                        continue
                    dis = abs(vector1[0] * vector2[1] - vector1[1]*vector2[0]) / vector1.norm(dim=0)
                    if dis < dis_min:
                        dis_min = dis
                    if dis_min < threshhold:
                        precision_count = precision_count +1
                        found = True
                        break

    for GT in positionGTs:
        for DT_point in GT:
            dis_min = 100000
            found = False
            if DT_point[0] < 0:
                break
            for predicts in positionpredicts:
                if found:
                    break
                point_num = len(predicts)
                recall_sum += 1
                for point_id in range(point_num-1):
                    point1 = predicts[point_id]
                    point2 = predicts[point_id+1]
                    vector1 = point2 - point1
                    vector2 = DT_point - point1
                    vector3 = DT_point - point2

                    if vector1.dot(vector2) < 0 or (-1*vector1).dot(vector3) < 0:
                        continue
                    dis = abs(vector1[0] * vector2[1] - vector1[1]*vector2[0]) / vector1.norm(dim=0)
                    if dis < dis_min:
                        dis_min = dis
                    if dis_min < threshhold:
                        recall_count = recall_count +1
                        found = True
                        break

    return precision_count, recall_count, precision_sum, recall_sum

def draw_polyline_onimg(polyline, img):

    num = polyline.shape[0]
    i = 0
    while i < num:
        if (polyline[i] < 0).any():
            break
        i += 1
    polyline = polyline[0:i]

    polyline = np.round(polyline).astype(np.int).reshape(-1, 1, 2)

    for point in polyline:
        cv2.circle(img, (point[0][0], point[0][1]), 2, (248, 20, 32), 2)

    cv2.polylines(img, [polyline], False, (13,217,58), thickness=1)

    return img

def colorize_mask(mask):
    palette = [0, 0, 0, 128,64,128,157,234,50]
    for i in range(256*3-len(palette)):
        palette.append(0)
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask = new_mask.convert('RGB')
    return new_mask

def post_process_predict(position_predict, ROI_size, img, search_radius=20):
    """

    :param position_predict: shape line_num * point_num
    :param ROI_size:
    :param img:
    :return:
    """
    gray_img = img.detach().cpu().numpy().astype(np.uint8)

    position_predict = position_predict.detach().cpu().numpy()

    gap = ROI_size[0]/2 # pixels
    for i, point in enumerate(position_predict):
        dis_matrix = np.ones((img.shape[0], img.shape[1]), dtype=np.float)
        dis_matrix = dis_matrix*img.shape[0] * img.shape[1]
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                if gray_img[y][x] != 2:
                    continue
                cur_img_point = np.array([x, y])
                dis = np.linalg.norm(point - cur_img_point)
                dis_matrix[y][x] = dis
        min_dis = dis_matrix.min()
        if min_dis > 2 and min_dis < search_radius:
            print("min_dis = ", min_dis)
            [y],[x] = np.where(dis_matrix == min_dis)
            position_predict[i] = np.array([x,y])

    return torch.tensor(position_predict).cuda()

def plotROI(ROI_params, ROI_size, img):
    for ROI_param in ROI_params:
        anchor_point, direction, mode, state = ROI_param
        if direction[0] == 0:
            angle = math.pi / 2
        else:
            angle = math.atan((-1 * direction[1]) / (direction[0]))
        if direction[0] < 0:
            angle = math.pi + angle
        angle = math.pi/2 - angle
        if mode == "middle":
            ratio = ROI_size[0] * 0.5
            anchor_point = anchor_point - ratio*direction

        corner_points_inROI = torch.tensor([[0,0], [0, ROI_size[0]-1], [ROI_size[1]-1, ROI_size[0]-1], [ROI_size[1]-1, 0]], dtype=torch.float)
        corner_points_ori = torch.zeros_like(corner_points_inROI)
        for ii in range(4):
            corner_points_ori[ii] = back_pos_to_ori(anchor_point, corner_points_inROI[ii].unsqueeze(0), angle, ROI_size).squeeze(0)
        color = ROI_colors[state]

        polyline = corner_points_ori.numpy().astype(np.int).reshape(-1, 1, 2)

        cv2.polylines(img, [polyline], True, color, thickness=2)

    return img

def calculate_state_acc(stateGTs, state_predicts, weight=[33,5]):
    normal_state_count = 0
    end_state_count = 0
    fork_state_count = 0
    merge_state_count = 0
    normal_state_bingo = 0
    end_state_bingo = 0
    fork_state_bingo = 0
    merge_state_bingo = 0
    assert len(stateGTs) == len(state_predicts)
    for id in range(len(stateGTs)):
        stateGT = stateGTs[id]
        state_predict = state_predicts[id].argmax()
        state_predict_after = -1
        state_predict_pre = -1
        if id > 0:
            state_predict_pre = state_predicts[id-1].argmax()
        if id < len(stateGTs) - 1:
            state_predict_after = state_predicts[id+1].argmax()
        if stateGT == 3:
            if state_predict == stateGT or state_predict_pre == stateGT or state_predict_after == stateGT:
                end_state_bingo += 1
            end_state_count += 1
        elif stateGT == 2:
            if state_predict == stateGT or state_predict_pre == stateGT or state_predict_after == stateGT:
                merge_state_bingo += 1
            merge_state_count += 1
        elif stateGT == 1:
            if state_predict == stateGT or state_predict_pre == stateGT or state_predict_after == stateGT:
                fork_state_bingo += 1
            fork_state_count += 1
        elif stateGT == 0:
            if state_predict == stateGT:
                normal_state_bingo += 1
            normal_state_count += 1

    return normal_state_count, normal_state_bingo, fork_state_count, fork_state_bingo, merge_state_count, merge_state_bingo, end_state_count, end_state_bingo


def if_in_box(corner_points, tar_point):
    """

    :param corner_points: [borttomleft, upleft, upright, bottomright]
    :param tar_point:
    :return:
    """
    x = tar_point[0]
    y = tar_point[1]
    if corner_points[0][0] <= x <= corner_points[2][0] and corner_points[1][1] <= y <= corner_points[3][1]:
        return True
    else:
        return False

def check_any_point_already_in(position, direction, angle, ROI_size, already_polylines):
    ratio = ROI_size[0] * 0.5

    gap = ROI_size[0] / 8
    ROI_width = ROI_size[1]
    ROI_height = ROI_size[0]
    anchor_point = position - ratio * direction

    t1 = anchor_point[0] + ROI_size[0] * math.sin(angle) - ROI_size[1] / 2 * math.cos(angle)
    t2 = anchor_point[1] - ROI_size[0] * math.cos(angle) - ROI_size[1] / 2 * math.sin(angle)
    M = torch.tensor([
        [math.cos(angle), -math.sin(angle), t1],
        [math.sin(angle), math.cos(angle), t2]
    ], dtype=torch.float, device=anchor_point.device)
    M = torch.cat((M, torch.tensor([[0, 0, 1]], dtype=M.dtype, device=M.device)), dim=0)
    M = M.inverse()

    for polyline in already_polylines:
        polyline_ = oversample_polyline(polyline, gap)
        for point in polyline_:
            point_ = torch.tensor([[point[0]], [point[1]], [1.0]], dtype=point.dtype, device=point.device)
            point_ = torch.mm(M, point_)
            ROI_corners = [[0, ROI_height - 1], [0, 0], [ROI_width - 1, 0], [ROI_width - 1, ROI_height - 1]]
            if if_in_box(ROI_corners, point_):
                return True
    return False

def find_nearest_forkline(init_node, fork_another_init_nodes, fork_another_lines, thresh=30):
    num = len(fork_another_init_nodes)
    pos_tar = init_node["init_position"]
    dis_min = 100000
    pick_id = -1
    for i in range(num):
        pos_ori = torch.tensor(fork_another_init_nodes[i]["init_position"]).to("cuda").float()
        dis = F.pairwise_distance(pos_tar.unsqueeze(0), pos_ori.unsqueeze(0), p=2).item()

        if dis < dis_min:
            dis_min = dis
            pick_id = i
    if dis_min < thresh:
        return fork_another_lines[pick_id], fork_another_init_nodes[pick_id], pick_id
    else:
        return None, None, None


def make_visaul_DTmap(DT_tensor):
    DT_numpy = DT_tensor.squeeze(0).detach().cpu().numpy()
    DT_numpy = cv2.resize(DT_numpy, (DT_numpy.shape[1]*4, DT_numpy.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    DT_norm_img = np.zeros(DT_numpy.shape)
    DT_norm_img = cv2.normalize(DT_numpy, DT_norm_img, 0,255, cv2.NORM_MINMAX)
    DT_norm_img = np.asarray(DT_norm_img, dtype=np.uint8)
    DT_heat_img = cv2.applyColorMap(DT_norm_img, cv2.COLORMAP_JET)
    DT_heat_img = cv2.cvtColor(DT_heat_img,cv2.COLOR_BGR2RGB)
    return DT_heat_img

palette = [0, 0, 0, 128, 64, 128, 157, 234, 50]
for i in range(256 * 3 - len(palette)):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    new_mask = new_mask.convert('RGB')
    return new_mask


def draw_polyline_on_oriimg(polylines, img, states):

    set_state = False
    if not len(polylines) == len(states):
        set_state = True

    for id, polyline in enumerate(polylines):

        if set_state:
            state = np.zeros((len(polyline)))
        else:
            state = states[id]
            state = state.detach().cpu().numpy()
            if state.shape[1] > 1:
                state = np.argmax(state, axis=1)

        polyline = polyline.detach().cpu().numpy()
        num = polyline.shape[0]
        i = 0
        while i < num:
            if (polyline[i] < 0).any():
                break
            i += 1
        polyline = polyline[0:i]
        polyline = np.round(polyline).astype(np.int).reshape(-1, 1, 2)

        for ii, point in enumerate(polyline):
            if ii == len(state):
                break
            if state[ii] == 0:
                cv2.circle(img, (point[0][0], point[0][1]), 2, ROI_colors[0], 12)
            elif state[ii] == 1:
                cv2.rectangle(img, (point[0][0]-8, point[0][1]-8),(point[0][0]+8, point[0][1]+8), ROI_colors[1], 4)
            elif state[ii] == 2:
                cv2.rectangle(img, (point[0][0] - 8, point[0][1] - 8),(point[0][0] + 8, point[0][1] + 8),ROI_colors[2],4)
            elif state[ii] == 3:
                cv2.rectangle(img, (point[0][0] - 8, point[0][1] - 8), (point[0][0] + 8, point[0][1] + 8),ROI_colors[3], 4)

        for ii, point in enumerate(polyline):
            #
            if ii == len(polyline)-1:
                cv2.rectangle(img, (point[0][0] - 8, point[0][1] - 8), (point[0][0] + 8, point[0][1] + 8),
                              ROI_colors[3], 4)
            else:
                cv2.circle(img, (point[0][0], point[0][1]), 2, ROI_colors[0], 12)

        cv2.polylines(img, [polyline], False, (125, 244, 244), thickness=4)
    return img


def draw_polyline_on_predictimg(polylines, img, states):

    set_state = False
    post_process_predict = False
    if not len(polylines) == len(states):
        set_state = True
    for id, polyline in enumerate(polylines):
        if set_state:
            state = np.zeros((len(polyline)))
        else:
            state = states[id]
            state = state.detach().cpu().numpy()
            if state.shape[1] > 1:
                state = np.argmax(state, axis=1)
        polyline = polyline.detach().cpu().numpy()
        num = polyline.shape[0]
        i = 0
        while i < num:
            if (polyline[i] < 0).any():
                break
            i += 1
        polyline = polyline[0:i]
        polyline = np.round(polyline).astype(np.int).reshape(-1, 1, 2)

        for ii, point in enumerate(polyline):
            if ii == len(state):
                break

            if state[ii] == 0:
                cv2.circle(img, (point[0][0], point[0][1]), 2, ROI_colors[0], 12)
            elif state[ii] == 1:
                cv2.rectangle(img, (point[0][0]-8, point[0][1]-8),(point[0][0]+8, point[0][1]+8), ROI_colors[1], 4)
            elif state[ii] == 2:
                cv2.rectangle(img, (point[0][0] - 8, point[0][1] - 8),(point[0][0] + 8, point[0][1] + 8),ROI_colors[2],4)
            elif state[ii] == 3:
                cv2.rectangle(img, (point[0][0] - 8, point[0][1] - 8), (point[0][0] + 8, point[0][1] + 8),ROI_colors[3], 4)

        if len(polyline) >1:
            cv2.polylines(img, [polyline], False, (125,244,244), thickness=4)

    return img


def plot_state(states_label, states_predict,  polylines_label, polylines_predict):
    plt.figure()
    line_num = len(states_label)
    for line_id in range(line_num):
        label = states_label[line_id]
        predict = states_predict[line_id]
        label = label.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()
        predict = np.argmax(predict, axis=1)
        label_circle_x = []
        label_circle_y = []
        for point_id, state in enumerate(label):
            # draw circles
            state = state + line_id*5
            label_circle_x.append(point_id)
            label_circle_y.append(state)

        label_circle_x = np.array(label_circle_x)
        label_circle_y = np.array(label_circle_y)
        plt.plot(label_circle_x, label_circle_y, 'go', label='label_%d'%line_id, linewidth=1)#green o

        predict_triangle_x = []
        predict_triangle_y = []
        for point_id, state in enumerate(predict):
            # draw triangle
            state = state + line_id*5
            predict_triangle_x.append(point_id)
            predict_triangle_y.append(state)

        predict_triangle_x = np.array(predict_triangle_x)
        predict_triangle_y = np.array(predict_triangle_y)
        plt.plot(predict_triangle_x, predict_triangle_y, 'rx', label='label_%d'%line_id, linewidth=1)#red x
    plt.grid(which='both')
    plt.title("state_predictVSlabel")
    plt.xlabel("point_id")
    plt.ylabel("state")
    # plt.show()
    buffer_ = BytesIO() # using buffer
    plt.savefig(buffer_,format='png')
    buffer_.seek(0)
    dataPIL = Image.open(buffer_)
    img = np.asarray(dataPIL)
    # cv2.imshow('image', data)
    buffer_.close()
    img = img[:, :, 0:3]
    plt.close()
    return img

def make_random_argument(src_pos, src_dire, pos_x_thresh=2, pos_y_thresh=4, dire_thresh=0.05):
    """
    
    :param src_pos: 
    :param src_dire: 
    :param pos_x_thresh: +-10pos_x_thresh
    :param pos_y_thresh: +-16
    :param dire_thresh: +-20du
    :return: 
    """
    x_diff = (random.random() - 0.5) * 2 * pos_x_thresh
    y_diff = (random.random() - 0.5) * 2 * pos_y_thresh
    dire_diff = (random.random() - 0.5) * 2 * dire_thresh
    tarpos = torch.zeros_like(src_pos)
    tarpos[0] = src_pos[0] + x_diff
    tarpos[1] = src_pos[1] + y_diff

    tar_dire = torch.zeros_like(src_dire)
    if src_dire[0] == 0:
        angle = math.pi / 2
    else:
        angle = math.atan((-1 * src_dire[1]) / (src_dire[0]))
    if src_dire[0] < 0:
        angle = math.pi + angle
    angle = math.pi/2 - angle
    tar_angle = angle + dire_diff

    tar_dire[0] = 1*math.sin(tar_angle)
    tar_dire[1] = -1 * 1*math.cos(tar_angle)
    return tarpos, tar_dire


def is_continue_merge(positionGTs, position):
    pointnum = len(positionGTs) - 1
    while positionGTs[pointnum][0] == -1:
        positionGTs = positionGTs[:pointnum]
        pointnum = pointnum - 1
    end_point = positionGTs[pointnum]
    if position[1] - end_point[1] > 64:
        return True
    else:
        return False

def get_next_direction(position, positionGTs):
    y = position[1]
    ii = 0
    while positionGTs[ii][1] > y:
        ii = ii + 1
    next_direction = normalize(positionGTs[ii+2] - positionGTs[ii+1])

    return next_direction
