import torch
import os
import csv
import copy
import random
from utils.dataset_types import MotionState, Track
from utils.dataset_reader import Key
import numpy as np
import math
from utils import segmentation
from build_graph import Key_seg
import pickle

import networkx as nx
import matplotlib.pyplot as plt
# from model import proj_mat_inverse, proj_mat


def proj_mat_inverse(theta, X, Y):
    X = np.array(X)
    Y = np.array(Y)
    theta = np.array(theta)
    X = np.expand_dims(X, axis=(0, 1, 2))
    Y = np.expand_dims(Y, axis=(0, 1, 2))
    R11 = np.expand_dims(np.cos(theta), axis=(0, 1, 2))
    R12 = np.expand_dims(-np.sin(theta), axis=(0, 1, 2))
    R21 = np.expand_dims(np.sin(theta), axis=(0, 1, 2))
    R22 = np.expand_dims(np.cos(theta), axis=(0, 1, 2))

    R1 = np.concatenate((R11, R12, X), axis=2)
    R2 = np.concatenate((R21, R22, Y), axis=2)
    R3 = np.zeros([X.shape[0], 1, 3])
    R3[:, :, 2] = 1
    R = np.concatenate([R1, R2, R3], axis=1)
    return R


def proj_mat(R):
    R_inversed = R.copy().transpose((0, 2, 1))
    R_inversed[:, 2, :] = np.array([0.0, 0.0, 1.0])
    R_inversed[:, 0, 2] = -R_inversed[:, 0, 0] * \
        R[:, 0, 2] - R_inversed[:, 0, 1]*R[:, 1, 2]
    R_inversed[:, 1, 2] = -R_inversed[:, 1, 0] * \
        R[:, 0, 2] - R_inversed[:, 1, 1]*R[:, 1, 2]
    return R_inversed


class traj_generator(object):
    def __init__(self, frame_n, frame_gap):
        self.frame_n = frame_n  # how many agents involved
        self.frame_gap = frame_gap  # the number of gap
        self.check_set = set()

    def index_model(self):
        self.track_id_pos = 0
        self.x_pos = 1
        self.y_pos = 2
        self.vx_pos = 3
        self.vy_pos = 4
        self.agent_type_pos = 5
        self.wid_pos = 6
        self.len_pos = 7
        self.psi_pos = 8
        self.agrole_pos = 9
        self.frame_id = 10
    def set_path(self, pth):
        self.data_path = pth
        base_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/vec_dir'
        sce_dir = os.path.join(base_path, self.data_path.split('/')[-2])
        save_path = os.path.join(sce_dir, str(
            self.frame_n)+'framesperseg'+os.path.split(self.data_path)[1].split('.')[0]+'.pickle')
        return save_path

    def load(self):
        with open(self.data_path, 'rb') as fp:
            self.data = pickle.load(fp)

    def traj_generate(self):
        #self.data: B * seq_len * agent_num * dim
        whole_feature_map = dict()
        tracks = dict()
        for Bs in self.data:
            tracks = dict()
            track_set = set()
            for frame_id in self.data[Bs]:
                frame = self.data[Bs][frame_id]
                for i, lines in enumerate(frame):
                    lines.append(frame_id)
                    # if i == 0:
                    #     assert lines[self.agrole_pos] == 1, "the first one is not agent"
                    #     iR = proj_mat_inverse(lines[self.psi_pos])
                    #     R = proj_mat(iR)

                    if not (lines[0] in track_set):
                        tracks[lines[0]] = []
                        tracks[lines[0]].append(lines)
                        track_set.add(lines[0])
                    else:
                        tracks[lines[0]].append(lines)
            whole_feature_map[Bs] = tracks
        self.whole_feature_map = whole_feature_map
        self.transform()
        # return save_path

    def transform(self):
        transed_feature_map = dict()
        for Bs in self.whole_feature_map:
            transed_tracks = dict()
            for i, track_id in enumerate(self.whole_feature_map[Bs]):
                track = np.array(self.whole_feature_map[Bs][track_id])
                if i == 0:
                    # print(track[0:10,1:3])
                    assert track[:, 9].all() == 1, 'not the agent!' + \
                        str(track[0, 9])
                    iR = proj_mat_inverse(
                        track[9][8], track[9][1], track[9][2])
                    R = proj_mat(iR)
                    # print(np.matmul(R,iR))
                track_tmp = track[:, 1:5].copy()
                track[:, self.x_pos] = R[0, 0, 0]*track_tmp[:, 0] + \
                    R[0, 0, 1]*track_tmp[:, 1] + R[0, 0, 2]
                track[:, self.y_pos] = R[0, 1, 0]*track_tmp[:, 0] + \
                    R[0, 1, 1]*track_tmp[:, 1] + R[0, 1, 2]
                track[:, self.vx_pos] = R[0, 0, 0]*track_tmp[:, 2] + \
                    R[0, 0, 1]*track_tmp[:, 3]
                track[:, self.vy_pos] = R[0, 1, 0]*track_tmp[:, 2] + \
                    R[0, 1, 1]*track_tmp[:, 3]
                transed_tracks[track_id] = track
            transed_feature_map[Bs] = (transed_tracks.copy(), iR)
            transed_tracks.clear()
        base_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/vec_dir'
        sce_dir = os.path.join(base_path, self.data_path.split('/')[-2])
        if not os.path.exists(sce_dir):
            os.makedirs(sce_dir)
        save_path = os.path.join(sce_dir, str(
            self.frame_n)+'framesperseg'+os.path.split(self.data_path)[1].split('.')[0][-4:]+'.pickle')
        with open(save_path, "wb") as fp:  # Pickling
            pickle.dump(transed_feature_map, fp,
                        protocol=pickle.HIGHEST_PROTOCOL)
        # return save_path


if __name__ == '__main__':
    # with open("/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_USA_Intersection_EP0/40framespersegtracks_000.pickle", "rb") as fp:   #Pickling
    #     mydict = pickle.load(fp)
    # print(len(mydict[0]))
    # loader = traj_generator(40,10)
    # pth = "/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_USA_Intersection_EP0/40framespersegtracks_000.pickle"
    # loader.load(pth)
    # loader.traj_generate()
    # pth = '/home/jonathon/Documents/new_project/interaction-dataset-master/vec_dir/DR_USA_Intersection_EP0/40framesperseg40framespersegtracks_000.pickle'
    # with open(pth, "rb") as fp:
    #     dic = pickle.load(fp)
    # for Bs in dic:
    #     for i, track_id in enumerate(dic[Bs]):
    #         track = np.array(dic[Bs][track_id])
    #         if i == 0:
    #             # print(track[0:10,1:3])
    #             assert track[:,9].all() == 1, 'not the agent!'+str(track[0,9])
    #             iR = proj_mat_inverse(track[9][8],track[9][1],track[9][2])
    #             R = proj_mat(iR)
    #             # print(np.matmul(R,iR))
    #         track_tmp = track[:,1:5].copy()
    #         track[:,1] = R[0,0,0]*track_tmp[:,0] + R[0,0,1]*track_tmp[:,1] + R[0,0,2]
    #         track[:,2] = R[0,1,0]*track_tmp[:,0] + R[0,1,1]*track_tmp[:,1] + R[0,1,2]
    #         track[:,3] = R[0,0,0]*track_tmp[:,2] + R[0,0,1]*track_tmp[:,3]
    #         track[:,4] = R[0,1,0]*track_tmp[:,2] + R[0,1,1]*track_tmp[:,3]
    loader = traj_generator(40, 10)
    loader.index_model()
    dir_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data'
    for scenarios in os.listdir(dir_path):
        if scenarios[0] == 'D':
            scenarios = os.path.join(dir_path, scenarios)
            for files in sorted(os.listdir(scenarios)):
                if '.pickle' in files:
                    pickle_path = os.path.join(scenarios,files)
                    print(pickle_path)
                    ex = loader.set_path(pickle_path)
                    # if os.path.exists(ex):
                    #     print("done")
                    #     continue

                    loader.load()
                    loader.traj_generate()
    #/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_DEU_Merging_MT/40framespersegtracks_000.pickle
    # problems maybe exists                
