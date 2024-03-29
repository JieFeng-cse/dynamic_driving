import xml.etree.ElementTree as xml
import pyproj
import math
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from utils import dict_utils
import os
import pickle
class Point:
    def __init__(self):
        self.x = None
        self.y = None

def get_x_y_lists(element, point_dict):
    x_list = list()
    y_list = list()
    for nd in element.findall("nd"):
        pt_id = int(nd.get("ref"))
        point = point_dict[pt_id]
        x_list.append(point.x)
        y_list.append(point.y)
    return x_list, y_list
class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin+180.)/6)+1  # works for most tiles, and for all in the dataset
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x-self.x_origin, y-self.y_origin]
def set_visible_area(point_dict, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for id, point in dict_utils.get_item_iterator(point_dict):
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 10, max_x + 10])
    axes.set_ylim([min_y - 10, max_y + 10])

def get_type(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "type":
            return tag.get("v")
    return None


def get_subtype(element):
    for tag in element.findall("tag"):
        if tag.get("k") == "subtype":
            return tag.get("v")
    return None

def draw_map_without_lanelet(filename, axes, lat_origin, lon_origin, true_point, pred_point,epoch_id):

    assert isinstance(axes, matplotlib.axes.Axes)

    axes.set_aspect('equal', adjustable='box')
    axes.patch.set_facecolor('lightgrey')

    projector = LL2XYProjector(lat_origin, lon_origin)

    e = xml.parse(filename).getroot()

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point

    set_visible_area(point_dict, axes)

    unknown_linestring_types = list()

    for way in e.findall('way'):
        way_type = get_type(way)
        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        elif way_type == "curbstone":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "line_thin":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=1, zorder=10)
        elif way_type == "line_thick":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                type_dict = dict(color="white", linewidth=2, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="white", linewidth=2, zorder=10)
        elif way_type == "pedestrian_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif way_type == "bike_marking":
            type_dict = dict(color="white", linewidth=1, zorder=10, dashes=[5, 10])
        elif way_type == "stop_line":
            type_dict = dict(color="white", linewidth=3, zorder=10)
        elif way_type == "virtual":
            type_dict = dict(color="blue", linewidth=1, zorder=10, dashes=[2, 5])
        elif way_type == "road_border":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "guard_rail":
            type_dict = dict(color="black", linewidth=1, zorder=10)
        elif way_type == "traffic_sign":
            continue
        else:
            if way_type not in unknown_linestring_types:
                unknown_linestring_types.append(way_type)
            continue

        x_list, y_list = get_x_y_lists(way, point_dict)
        plt.plot(x_list, y_list, **type_dict)
    type_dict = dict(color="green", linewidth=1,linestyle="dashed", marker='x', markersize=2)
    plt.plot((true_point[:,0]), true_point[:,1], **type_dict)

    type_dict = dict(color="red", linewidth=1,linestyle="dashed", marker='x',  markersize=2)
    plt.plot(pred_point[:,0], pred_point[:,1], **type_dict)
    print(unknown_linestring_types)
    plot_pth = '/home/jonathon/Documents/new_project/interaction-dataset-master/traj'
    pth = os.path.join(plot_pth,'epoch'+epoch_id+'traj.png')
    plt.savefig(pth)
    plt.close()
    # plt.ion()
    # plt.pause(10)
    

def main_drawer(map_path, truth_point, pred_point, epoch_id):
    fig, axes = plt.subplots(1, 1)
    e = xml.parse(map_path).getroot()
    draw_map_without_lanelet(map_path, axes, 0,0, truth_point, pred_point, epoch_id)

def main_vector(map_path):
    r"""
    returns a numpy array.
    """
    e = xml.parse(map_path).getroot()
    projector = LL2XYProjector(0,0) # because coord in osm has already been pre-processed

    point_dict = dict()
    for node in e.findall("node"):
        point = Point()
        point.x, point.y = projector.latlon2xy(float(node.get('lat')), float(node.get('lon')))
        point_dict[int(node.get('id'))] = point
    
    P = find_all_ways(e, point_dict)
    # print(P[-1])
    P = np.array(P).astype(np.float32)
    return P

def find_all_ways(e, point_dict):
    '''
    get polylines and their type
    '''
    unknown_linestring_types = list()
    j_id = -1
    P = [] # holds all ways. eg: P=[[way 1's vectors], [way 2's vectors], ..., [way n's vectors]] or like: P=[p1, p2, ..., pn]
    for way in e.findall('way'):
        p_j = []# holds all vectors, eg: p_j = [v1, v2, ..., vn]
        j_id += 1 # vi=[ds, de, a, j], j id start with 0
        # feature_encode = [0,0,0,0,0,0,0,0,0,0,0] # len = 11
        feature_encode = [0,0,0,0,0,0,0,0] # len = 8
        way_type = get_type(way)
        if way_type is None:
            raise RuntimeError("Linestring type must be specified")
        elif way_type == "curbstone" or way_type == 'guard_rail' or way_type == 'road_border':
            feature_encode[0] = 1
        elif way_type == "line_thin":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                feature_encode[1] = 1
                feature_encode[3] = 1
            else:
                feature_encode[1] = 1
                feature_encode[3] = 0
        elif way_type == "line_thick":
            way_subtype = get_subtype(way)
            if way_subtype == "dashed":
                feature_encode[2] = 1
                feature_encode[3] = 1
            else:
                feature_encode[2] = 1
        elif way_type == "pedestrian_marking":
            feature_encode[4] = 1
        elif way_type == "bike_marking":
            feature_encode[5] = 1
        elif way_type == "stop_line":
            feature_encode[6] = 1
        elif way_type == "virtual":
            feature_encode[7] = 1
        # elif way_type == "road_border":
        #     feature_encode[8] = 1
        # elif way_type == "guard_rail":
        #     feature_encode[9] = 1
        # elif way_type == "traffic_sign":
        #     feature_encode[10] = 1
        else:
            if way_type not in unknown_linestring_types:
                unknown_linestring_types.append(way_type)
            continue

        x_list, y_list = get_x_y_lists(way, point_dict)
        assert len(x_list) == len(y_list), 'x list and y list must be matched.'
        for i in range(len(x_list)-1):
            d_s_x, d_s_y = x_list[i], y_list[i]
            d_e_x, d_e_y = x_list[i+1], y_list[i+1]
            vi = [d_s_x,d_s_y ,d_e_x, d_e_y, 0.0]+feature_encode # here 0.0 means this vector is a line on the map
            vi.append(j_id)
            p_j.append(vi)
        # P.append(p_j)
        pj_group = norm_nodes_polyline(p_j)
        for each_pj in pj_group:
            P.append(each_pj)
    return P
def norm_nodes_polyline(pj):
    to_nodes_num = 1
    pj_group = []
    if len(pj) < to_nodes_num:
        to_add = to_nodes_num - len(pj)
        for i in range(to_add):
            pj.append(pj[i])#copy
        pj_group.append(pj)
    elif len(pj) > to_nodes_num:
        new_pj = []
        multiple_times = int(len(pj) / to_nodes_num)
        remain_item = len(pj) % to_nodes_num
        for i in range(multiple_times):
            new_pj = pj[i*to_nodes_num:(i+1)*to_nodes_num]
            pj_group.append(new_pj)
        if remain_item != 0:
            to_add = to_nodes_num - remain_item
            new_pj = pj[-remain_item:]
            for i in range(to_add):
                new_pj.append(new_pj[i]) #copy
        pj_group.append(new_pj)
    elif len(pj) == to_nodes_num:
        pj_group.append(pj)
    # for each in pj_group:
    #     print(len(each))
    return pj_group
if __name__ == "__main__":
    # projector = LL2XYProjector(0,0) # because coord in osm has already been pre-processed
    pth = '/home/jonathon/Documents/new_project/interaction-dataset-master/vec_dir/DR_USA_Intersection_EP0/40framesperseg_000.pickle'
    map_pth = '/home/jonathon/Documents/new_project/interaction-dataset-master/maps/DR_USA_Intersection_EP0.osm'
    with open(pth, "rb") as fp:
        dic = pickle.load(fp)
    print(len(dic))
    for Bs in dic:
        iR =dic[Bs][1]
        # print(iR)
        tmp = dic[Bs][0]
        for i, track_id in enumerate(tmp):
            print(track_id)
            track = np.array(tmp[track_id])
            track_tmp = track[:,1:5].copy()
            track[:,1] = iR[0,0,0]*track_tmp[:,0] + iR[0,0,1]*track_tmp[:,1] + iR[0,0,2]
            track[:,2] = iR[0,1,0]*track_tmp[:,0] + iR[0,1,1]*track_tmp[:,1] + iR[0,1,2]
            pa = np.concatenate((np.expand_dims(track[:,1],(1)),np.expand_dims(track[:,2],(1))),axis=1)
            print(pa.shape)
            main_drawer(map_pth,pa,pa,str(Bs)+str(track_id))
            # track[:,3] = R[0,0,0]*track_tmp[:,2] + R[0,0,1]*track_tmp[:,3]
            # track[:,4] = R[0,1,0]*track_tmp[:,2] + R[0,1,1]*track_tmp[:,3]
    pa = np.array([[0,0]])
    pb = np.array([[0,0]])
    # main_drawer('/home/jonathon/Documents/new_project/interaction-dataset-master/maps/DR_CHN_Merging_ZS.osm',pa,pb) # draw the map with matplot
    # main_vector('./maps/DR_USA_Intersection_EP0.osm')
    
