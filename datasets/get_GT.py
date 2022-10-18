"""
provide ground-truth of DT, state, direction
"""
import numpy as np
import cv2
import math
from tools import getROI, normalize
import torch
from ipdb import set_trace as bp

def make_DT_GT(img, polylines, dis_thresh=16, line_value=8):

    heatMap = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for polyline in polylines:
        points = np.round(np.array(polyline))
        points = points.reshape(-1,1,2)
        points = points.astype(np.int32)
        heatMap = cv2.polylines(heatMap, [points], False, line_value)
    gap = line_value/dis_thresh
    for row in range(heatMap.shape[0]):
        for col in range(heatMap.shape[1]):
            if(heatMap[row][col] == line_value):
                for i in range(2*dis_thresh+1):
                    for j in range(2*dis_thresh+1):
                        row_ = row + dis_thresh - i
                        col_ = col + dis_thresh - j
                        if row_<0 or row_>=heatMap.shape[0] or col_ <0 or col_>=heatMap.shape[1]:
                            continue
                        else:
                            distance = round(np.linalg.norm(np.array([row-row_, col-col_])))
                            if distance >= dis_thresh:
                                continue
                            else:
                                value = line_value- distance*gap
                                heatMap[row_][col_] = max(value, heatMap[row_][col_])
    return heatMap

def show_DT_heatmap(graymap):

    norm_img = np.zeros(graymap.shape)
    norm_img = cv2.normalize(graymap, norm_img, 0,255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

    cv2.namedWindow("GT", 0)
    cv2.imshow("GT", heat_img)
    cv2.waitKey(0)


def make_thata_GT(anchor_point, polylines):
    dis_min = 10000
    heading = torch.tensor([0., 1.], dtype=torch.float, device=anchor_point.device)
    closest_id = -1
    for id, polyline in enumerate(polylines):
        point_num = len(polyline)
        for point_id in range(point_num-1):
            if polyline[point_id+1][0] < 0:
                break
            point1 = polyline[point_id]
            point2 = polyline[point_id+1]
            vector1 = point2 - point1
            vector2 = anchor_point - point1
            vector3 = anchor_point - point2
            # judge if can project point on this segment

            if vector1.dot(vector2) < 0 or (-1*vector1).dot(vector3) < 0:
                continue
            dis = abs(vector1[0] * vector2[1] - vector1[1]*vector2[0]) / vector1.norm(dim=0)
            if dis < dis_min:
                dis_min = dis
                heading = vector1
                closest_id = id

    heading_norm = heading.norm(dim=0)
    if heading_norm == 0:
        bp()
    heading = heading/heading_norm
    return heading, closest_id, dis_min


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



def make_state_GT_tensor(polylineMaps, polylines, anchor_point, heading, ROI_size, imgName, show=False, ratio=0.75):
    """

    :param polylineMaps:
    :param anchor_point:
    :param heading:
    :param ROI_size: [height, width]
    :return:
    """
    if isinstance(polylineMaps, list):
        polylineMaps = torch.tensor(np.array(polylineMaps)).float()
    if isinstance(polylines, list):
        for id, polyline in enumerate(polylines):
            if len(polyline) < 80:
                polyline += [[-1,-1] for i in range(80-len(polyline))]
                polylines[id] = polyline
        polylines = torch.tensor(np.array(polylines)).float()
    if isinstance(anchor_point, list):
        anchor_point = torch.tensor(np.array(anchor_point)).float()
    if isinstance(heading, list):
        heading = torch.tensor(heading).float()
    state = 0 # 0:normal, 1:fork, 2:merge, 3:end
    count_innerline = 0
    count_startpoint = 0
    count_endpoint = 0
    # todo
    ROI_size = [ROI_size[0]*ratio, ROI_size[1]*ratio]

    ROI_width = ROI_size[1]
    ROI_height = ROI_size[0]

    curROI_multy, angle = getROI(polylineMaps, anchor_point, heading, ROI_size, scale=1, mode="middle")
    ratio = ROI_size[0] * 0.5
    anchor_point = anchor_point - ratio*heading
    t1 = anchor_point[0] + ROI_size[0]*math.sin(angle) - ROI_size[1]/2*math.cos(angle)
    t2 = anchor_point[1] - ROI_size[0]*math.cos(angle) - ROI_size[1]/2*math.sin(angle)
    M = torch.tensor([
        [math.cos(angle),-math.sin(angle),t1],
        [math.sin(angle),math.cos(angle) ,t2]
        ], dtype=torch.float, device=anchor_point.device)
    M = torch.cat((M, torch.tensor([[0,0,1]], dtype=M.dtype, device=M.device)), dim=0)
    M = M.inverse()
    ROI_toshow = torch.zeros_like(curROI_multy[0])

    for id, curROI in enumerate(curROI_multy):

        if show:
            show_DT_heatmap(curROI.detach().cpu().numpy())

        if curROI.max() > 0:
            if show:
                ROI_toshow += curROI
            count_innerline = count_innerline + 1
            polyline = polylines[id]
            start_point = polyline[0]
            pointnum = len(polyline)-1
            while polyline[pointnum][0]==-1:
                pointnum = pointnum -1

            end_point = polyline[pointnum]

            start_point_ = torch.tensor([[start_point[0]], [start_point[1]], [1.0]], dtype=start_point.dtype, device=start_point.device)
            start_point_ = torch.mm(M, start_point_)

            end_point_ = torch.tensor([[end_point[0]], [end_point[1]], [1.0]], dtype=end_point.dtype, device=end_point.device)
            end_point_ = torch.mm(M, end_point_)

            ROI_corners = [[0, ROI_height-1], [0, 0], [ROI_width-1, 0], [ROI_width-1, ROI_height-1]]
            if if_in_box(ROI_corners, start_point_):
                count_startpoint = count_startpoint + 1
                if count_innerline > 1:
                    show_DT_heatmap(curROI.detach().cpu().numpy())

            if if_in_box(ROI_corners, end_point_):
                show_DT_heatmap(curROI.detach().cpu().numpy())
                count_endpoint = count_endpoint + 1

    if show:
        print("count_innerline", count_innerline, "   count_startpoint", count_startpoint, "   count_endpoint", count_endpoint)
        if count_innerline == 2 and count_endpoint ==1:
            show_DT_heatmap(ROI_toshow.detach().cpu().numpy())
    if count_innerline > 1 and count_startpoint > 0:
        state = 1 #fork
    elif count_innerline > 1 and count_endpoint == 1:
        state = 2 #merge
    elif count_innerline == 0 or (count_innerline == 1 and count_endpoint == 1) or (count_innerline == 2 and count_endpoint == 2):
        state = 3 #end
    return state


def make_state_GT(polylineMaps, polylines, anchor_point, heading, ROI_size):
    """

    :param polylineMaps:
    :param anchor_point:
    :param heading:
    :param ROI_size: [height, width]
    :return:
    """
    state = 0 # 0:normal, 1:fork, 2:merge
    ROI_width = ROI_size[1]
    ROI_height = ROI_size[0]
    if heading[0] == 0:
        angle = math.pi / 2
    else:
        angle = math.atan((-1 * heading[1]) / (heading[0]))
    if heading[0] < 0:
        angle = math.pi + angle

    M = cv2.getRotationMatrix2D((anchor_point[0], anchor_point[1]), 90 - angle*(180/ math.pi), 1)#ni shi zhen
    count_innerline = 0
    count_startpoint = 0
    count_endpoint = 0

    for id, polylineMap in enumerate(polylineMaps):
        img_rotated = cv2.warpAffine(polylineMap, M, (polylineMap.shape[1], polylineMap.shape[0]), borderValue=(0,0,0))  # M为上面的旋转矩阵

        curROI = img_rotated[round(anchor_point[1])-(ROI_height-1): round(anchor_point[1])+1, int(round(anchor_point[0]-(ROI_width-1)/2)) : int(round(anchor_point[0] + (ROI_width-1)/2+1))] # anchor at bottom

        if curROI.max() > 0:
            count_innerline = count_innerline + 1
            polyline = polylines[id]
            start_point = polyline[0]
            end_point = polyline[-1]
            start_point = np.array([[start_point[0]], [start_point[1]], [1]])
            end_point = np.array([[end_point[0]], [end_point[1]], [1]])
            start_point = np.matmul(M , start_point)
            end_point = np.matmul(M , end_point)
            start_point = [start_point[0] - (anchor_point[0] - (ROI_width-1)/2), start_point[1] - (anchor_point[1] - (ROI_height-1))]
            end_point = [end_point[0] - (anchor_point[0] - (ROI_width-1)/2), end_point[1] - (anchor_point[1] - (ROI_height-1))]

            ROI_corners = [[0, ROI_height-1], [0, 0], [ROI_width-1, 0], [ROI_width-1, ROI_height-1]]
            if(if_in_box(ROI_corners, start_point)):
                count_startpoint = count_startpoint + 1
            if(if_in_box(ROI_corners, end_point)):
                count_endpoint = count_endpoint + 1

    if count_innerline > 1:
        if count_startpoint > 0:
            state = 1
        elif count_endpoint > 0:
            state = 2
    return state









