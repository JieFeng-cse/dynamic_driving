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

import networkx as nx
import matplotlib.pyplot as plt

class feature_generator(object):
    def __init__(self, data_path, frame_n, frame_gap):
        self.data_path = data_path
        self.frame_n = frame_n #3
        self.frame_gap = frame_gap #5
        self.track_num = 1000
        self.data_iter_num = 0
        self.check_set = set()
        self.frame_order = 1
        self.last_frame = -1
        self.current_frame = 0
        self.next_frame = 1
        self.final_set_list = []
        self.agent_list = []
        self.end = False
    def smallest_A_size_check(self):
        title, self.data =  segmentation.load_csv(self.data_path)
        self.get_title_pos(title)
        self.data_len = len(self.data)
        if self.data_len <= 1:
            self.track_pos = None
            return None
        i = 0
        while i in range(self.data_len):
            # print(str(self.data_len) + ' ' + str(i))
            self.frame_order = 1
            set_list = []
            if self.end:
                print("end")
                break
            j = 0
            agent_flag = 0
            # print("into the while")
            while True:
                # print(set_list)
                line = self.data[min(i + j, len(self.data) - 1)]
                if line[self.agrole_pos] == 'agent' and not agent_flag:
                    agent_flag = 1
                    self.agent_list.append(int(line[self.track_pos]))
                if i + j >= len(self.data) - 1:
                    set_list.append(copy.copy(self.check_set))
                    self.end = True
                    set_num = len(set_list)
                    if set_num < self.frame_n:
                        del self.agent_list[-1]
                        break
                    final_set = set_list[0]
                    for set_idx in range(1, set_num):
                        final_set = final_set & set_list[set_idx]
                    self.final_set_list.append(final_set)
                    self.data_iter_num += 1
                    if len(final_set) < self.track_num:
                        self.track_num = len(final_set)
                        print(str(self.track_num)+"*******************")
                    break
                next_line = self.data[i + j + 1]
                self.next_frame = int(next_line[self.frame_pos])
                self.last_frame = int(self.current_frame)
                self.current_frame = int(line[self.frame_pos])
                self.check_set.add(int(line[self.track_pos]))
                j += 1
                # print(self.check_set)
                # print(str(self.current_frame)+ '  ' + str(self.next_frame))
                if self.next_frame - self.current_frame == 1:
                    # print("new frame")
                    self.frame_order += 1
                    set_list.append(copy.copy(self.check_set))
                    self.check_set.clear()
                elif abs(self.next_frame - self.current_frame) > 1:
                    self.frame_order += 1
                    set_list.append(copy.copy(self.check_set))
                    self.check_set.clear()

                    set_num = len(set_list)
                    # print("*****************" + str(set_num))
                    if set_num < self.frame_n:
                        del self.agent_list[-1]
                        break
                    final_set = set_list[0]
                    for set_idx in range(1, set_num):
                        final_set = final_set & set_list[set_idx]
                    self.data_iter_num += 1
                    self.final_set_list.append(final_set)
                    if len(final_set) < self.track_num:
                        self.track_num = len(final_set)
                        print(str(self.track_num)+"*******************")
                    break
            i = i + j
        self.end = False
        return self.track_num                           

    def get_title_pos(self,title):
        self.time_pos = title.index(Key_seg.time_stamp_ms)
        self.track_pos = title.index(Key_seg.track_id)
        self.frame_pos = title.index(Key_seg.frame_id)
        self.agrole_pos = title.index(Key_seg.agent_role)
        self.len_pos = title.index(Key_seg.length)
        self.wid_pos = title.index(Key_seg.width)
        self.x_pos = title.index(Key_seg.x)
        self.y_pos = title.index(Key_seg.y)
        self.vx_pos = title.index(Key_seg.vx)
        self.vy_pos = title.index(Key_seg.vy)
        self.psi_pos = title.index(Key_seg.psi_rad)
        self.agent_type_pos = title.index(Key_seg.agent_type)
    def get_k_tracks(self):
        assert len(self.agent_list) == self.data_iter_num, 'agent number is not correct'
        zipped = zip(self.agent_list, self.final_set_list)
        for sets in zipped:
            assert len(sets[1]) >= self.track_num
            for i in range(len(sets[1]) - self.track_num):
                sets[1].remove(random.choice(tuple(sets[1])))
            if sets[0] not in sets[1]:
                sets[1].remove(random.choice(tuple(sets[1])))
                sets[1].add(sets[0])
        # print(self.final_set_list[0])    
           
       
    def construct_features(self):
        whole_feature_map = []
        i = 0
        clip_id = 0
        while i in range(self.data_len):
            j = 0
            feature_map_per_chip = []
            if self.end:
                print("end")
                break
            feature_map_per_frame = []
            while True:
                line = self.data[min(i + j, len(self.data) - 1)]
                feature_vec = [int(line[self.track_pos]),float(line[self.x_pos])/1000.0,float(line[self.y_pos])/1000.0,float(line[self.vx_pos]),\
                    float(line[self.vy_pos]),int(line[self.agent_type_pos] != 'car'),float(line[self.wid_pos]),float(line[self.len_pos]),\
                        float(line[self.psi_pos]),int(line[self.agrole_pos]=='agent')]

                if i + j >= len(self.data) - 1:
                    feature_map_per_frame.append(copy.copy(feature_vec))
                    feature_map_per_chip.append(copy.copy(feature_map_per_frame))
                    self.end = True
                    frame_num = len(feature_map_per_chip)
                    if frame_num < self.frame_n:
                        feature_map_per_chip.clear()
                        feature_map_per_frame.clear()
                        break
                    new_frame = []
                    new_frame_chip = []
                    agent_track =None
                    for idx_f, frame in enumerate(feature_map_per_chip):
                        idx_v = 0
                        for single_track in frame:
                            if int(single_track[0]) in self.final_set_list[clip_id]:
                                if int(single_track[0]) == self.agent_list[clip_id]:
                                    agent_track = single_track
                                    agent_track_id = idx_v
                                single_track[0] = float(single_track[0])/100.0
                                new_frame.append(copy.copy(single_track))
                                idx_v += 1
                        if agent_track_id != 0:
                            new_frame[agent_track_id] = new_frame[0]
                            new_frame[0] = agent_track

                        new_frame_chip.append(copy.copy(new_frame))                      
                        new_frame.clear()
                    whole_feature_map.append(copy.copy(new_frame_chip))
                    clip_id += 1
                    break
                next_line = self.data[i + j + 1]
                self.next_frame = int(next_line[self.frame_pos])
                self.last_frame = int(self.current_frame)
                self.current_frame = int(line[self.frame_pos])
                feature_map_per_frame.append(copy.copy(feature_vec))
                j += 1
                if self.next_frame - self.current_frame == 1:
                    self.frame_order += 1
                    feature_map_per_chip.append(copy.copy(feature_map_per_frame))
                    feature_map_per_frame.clear()
                elif abs(self.next_frame - self.current_frame) > 1:
                    self.frame_order += 1
                    feature_map_per_chip.append(copy.copy(feature_map_per_frame))
                    feature_map_per_frame.clear()
                    # print(np.array(feature_map_per_chip).shape)

                    frame_num = len(feature_map_per_chip)
                    if frame_num < self.frame_n:
                        feature_map_per_chip.clear()
                        feature_map_per_frame.clear()
                        break
                    new_frame = []
                    new_frame_chip = []
                    agent_track =None
                    for idx_f, frame in enumerate(feature_map_per_chip):
                        idx_v = 0
                        for single_track in frame:
                            if int(single_track[0]) in self.final_set_list[clip_id]:
                                if int(single_track[0]) == self.agent_list[clip_id]:
                                    agent_track = single_track
                                    agent_track_id = idx_v
                                single_track[0] = float(single_track[0])/100.0
                                new_frame.append(single_track)
                                idx_v += 1
                        if agent_track_id != 0:
                            new_frame[agent_track_id] = new_frame[0]
                            new_frame[0] = agent_track
                            # print(str(round(new_frame[0][0]*100))+' ' + str(self.agent_list[clip_id]))
                            # print(new_frame[agent_track_id][0])
                        assert round(new_frame[0][0]*100) == int(self.agent_list[clip_id]),"agent is not at the first"
                        new_frame_chip.append(copy.copy(new_frame))                        
                        new_frame.clear()
                    whole_feature_map.append(copy.copy(new_frame_chip))
                    # if clip_id >= 21780:
                    #     print(np.array(whole_feature_map).shape)
                    #     print(np.array(whole_feature_map)[0][0][0][0])
                    # print(np.array(whole_feature_map).shape)
                    clip_id += 1
                    # print(clip_id)
                    break
            i = i + j
        self.end = False
        data = np.array(whole_feature_map)
        base_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data'
        sce_dir = os.path.join(base_path,'DR_CHN_Merging_ZS')
        if not os.path.exists(sce_dir):
            os.makedirs(sce_dir)
        save_path = os.path.join(sce_dir,str(self.frame_n)+'framesperseg'+os.path.split(self.data_path)[1].split('.')[0]+'.npy')
        np.save(save_path,data)
        print(data.shape)

    def prune(self,path,node_num):
        data = torch.tensor(np.load(path))
        if data.shape[2] == node_num:
            return
        data = data[:,:,0:node_num,:]
        np.save(path,data)
        print(data.shape)

        


if __name__ == '__main__':
    csv_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/recorded_trackfiles/DR_CHN_Merging_ZS/train/segmented/tracks_001.csv'
    ff = feature_generator(csv_path,40,10)
    print(ff.smallest_A_size_check())
    # print(ff.data_iter_num)
    # print(len(ff.agent_list))
    ff.get_k_tracks()
    print("get k tracks")
    # ff.construct_features()
    # pp = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/track_1_feature.npy'
    pp = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_CHN_Merging_ZS/40framespersegtracks_001.npy'
    ff.prune(pp,13)
    

