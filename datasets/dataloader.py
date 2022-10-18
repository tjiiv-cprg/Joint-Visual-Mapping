import os.path as ops
import cv2
import numpy as np
from torch.utils.data import Dataset
from xml.dom.minidom import parse
from tqdm import tqdm
from .get_GT import make_DT_GT, make_state_GT, make_state_GT_tensor
import json
from tools import check_mkdir, normalize

class LineTopologyData(Dataset):

    def __init__(self, datadir, ROI_size, folders, mode="train", sample_ratio=2):

        assert ops.exists(datadir), '{:s} does not exist'.format(datadir)
        self.datadir = datadir
        self.sample_ratio = sample_ratio
        self.mode = mode
        self.annoFilePaths = [ops.join(self.datadir, folder, "annotations.xml") for folder in folders]
        self.ROI_size = ROI_size
        self.img_polylines_dict, self.keys = self.readXML(self.annoFilePaths)
        self.init_nodes_info = {}
        self.make_DT_polylineMaps()


    def oversample_polyline_list(self, polyline, gap):
        i = 0
        end_point = polyline[-1]
        while 1:
            point = polyline[i]
            if i == len(polyline)-1:
                break
            point_ = polyline[i+1]
            dis = np.linalg.norm(np.array([point[0]-point_[0], point[1]-point_[1]]))

            if dis >= gap:
                to_add_point_num = int(dis // gap)
                x_gap = (point_[0] - point[0]) / dis * gap
                y_gap = (point_[1] - point[1]) / dis * gap
                for j in range(to_add_point_num):
                    new_point = [point[0] + (j+1)*x_gap, point[1] + (j+1)*y_gap]
                    polyline.insert(i+j+1, new_point)
                i = i + to_add_point_num
                del polyline[i+1]
            else:
                del polyline[i+1]
        polyline.append(end_point)
        return polyline


    def sort_polylines(self, polylines):
        polylines.sort(key=lambda x:x[0][0])
        return polylines


    def readXML(self, fileName):
        img_polylines_dict = {}
        keys = []
        for file in fileName:

            folderName = file.split('/')[-2]
            domTree = parse(file)
            rootNode = domTree.documentElement
            imagesNodes = rootNode.getElementsByTagName("image")
            for imageNode in tqdm(imagesNodes, desc='Processing'):
                id = imageNode.getAttribute("id")
                imgName = imageNode.getAttribute("name")
                width = imageNode.getAttribute("width")
                height = imageNode.getAttribute("height")
                keys.append((folderName, imgName))
                polylines = imageNode.getElementsByTagName("polyline")
                points_multylines = []
                for polyine in polylines:
                    label = polyine.getAttribute("label")
                    points_str = polyine.getAttribute("points")
                    points_num = []
                    for point_str in points_str.split(';'):
                        point_pair = []
                        for point_xy in point_str.split(','):
                            point_pair.append(float(point_xy))
                        points_num.append(point_pair)
                    points_num = self.oversample_polyline_list(points_num, self.ROI_size[0]/self.sample_ratio) #just for only state
                    points_multylines.append(points_num)
                points_multylines = self.sort_polylines(points_multylines)

                img_polylines_dict[imgName] = points_multylines

        return img_polylines_dict, keys

    def make_DT_polylineMaps(self):
        init_node_json_file = ops.join(self.datadir, "init_node_info.json")
        if ops.exists(init_node_json_file):
            print("init_node json alerady exists")
            with open(init_node_json_file, "r") as load_f:
                self.init_nodes_info = json.load(load_f)
            return
        print("this is the first time processing these data, please wait with patience!")
        check_mkdir(ops.join(self.datadir, "DT_maps"))
        check_mkdir(ops.join(self.datadir, "polylineMaps"))
        for folderName, imgName in tqdm(self.keys):
            polylines = self.img_polylines_dict[imgName]
            imgFilePath = ops.join(self.datadir, folderName, "gray_map", imgName)

            img = cv2.imread(imgFilePath, cv2.IMREAD_GRAYSCALE)
            if len(polylines) > 0:

                DT_GT = make_DT_GT(img, polylines)
                DT_GT = cv2.resize(DT_GT, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
                DT_img_name = ops.join(self.datadir, "DT_maps", imgName)
                cv2.imwrite(DT_img_name, DT_GT)
                polylineMaps = []
                init_nodes = []
                for id, polyline in enumerate(polylines):
                    polylineMap = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                    points = np.round(np.array(polyline))
                    points = points.reshape(-1, 1, 2)
                    points = points.astype(np.int32)
                    polylineMap = cv2.polylines(polylineMap, [points], False, 1)
                    polylineMapName = ops.join(self.datadir, "polylineMaps", imgName.replace(".png", "line_{:d}.png".format(id)))
                    cv2.imwrite(polylineMapName, polylineMap)
                    polylineMaps.append(polylineMap)
                for id, polyline in enumerate(polylines):
                    init_position = polyline[0]
                    init_direction = (np.array(polyline[1]) - np.array(polyline[0])).tolist()
                    init_direction = normalize(init_direction).tolist()
                    init_state = make_state_GT_tensor(polylineMaps, polylines, init_position, init_direction, self.ROI_size)
                    init_nodes.append({'init_position': init_position, 'init_direction': init_direction, 'init_state': init_state})
                self.init_nodes_info[imgName] = init_nodes

        with open(init_node_json_file, "w") as f:
            json.dump(self.init_nodes_info, f)
            print("Node information writing done!")


    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):

        folderName = self.keys[item][0]
        imgName = self.keys[item][1]
        polylines = self.img_polylines_dict[imgName]
        for id, polyline in enumerate(polylines):
            if len(polyline) < 50:
                polyline += [[-1,-1] for i in range(80-len(polyline))]
                polylines[id] = polyline

        polylines = np.array(polylines)
        imgFilePath = ops.join(self.datadir, folderName, "gray_map", imgName)
        img = cv2.imread(imgFilePath, cv2.IMREAD_GRAYSCALE)
        polylineMaps = []
        init_nodes = []
        DT_GT = []
        if len(polylines) > 0:
            DT_img_name = ops.join(self.datadir, "DT_maps", imgName)
            DT_GT = cv2.imread(DT_img_name, cv2.IMREAD_GRAYSCALE)
            for id, polyline in enumerate(polylines):
                polylineMapName = ops.join(self.datadir, "polylineMaps", imgName.replace(".png", "line_{:d}.png".format(id)))
                polylineMap = cv2.imread(polylineMapName, cv2.IMREAD_GRAYSCALE)
                polylineMaps.append(polylineMap)
            init_nodes = self.init_nodes_info[imgName]
        else:
            polylines = []

        if self.mode == 'test':
            return img, polylines, DT_GT, np.array(polylineMaps), init_nodes, imgName
        else:
            if img is None:
                print("img is None")
                print(imgName)
            if polylines is None:
                print("polylines is None")
            if DT_GT is None:
                print("DT_GT is None")
            if polylineMaps is None:
                print("polylineMaps is None")
            if init_nodes is None:
                print("init_nodes is None")
            try:
                return img, polylines, DT_GT, np.array(polylineMaps), init_nodes, imgName
            except:
                raise ValueError('')
