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

class feature_generator(object):
    def __init__(self, frame_n, frame_gap):       
        self.frame_n = frame_n #3
        self.frame_gap = frame_gap #5
        self.track_num = 1000
        self.data_iter_num = 0
        self.check_set = set()
        self.frame_order = 0
        self.last_frame = -1
        self.current_frame = 0
        self.next_frame = 1
        self.final_set_list = []
        self.agent_list = []
        self.end = False
    def init(self):
        title, self.data =  segmentation.load_csv(self.data_path)
        self.get_title_pos(title)
        self.data_len = len(self.data)
        if self.data_len <= 1:
            self.track_pos = None
            return None
    def set_path(self,data_path):
        self.data_path = data_path
        base_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data'
        sce_dir = os.path.join(base_path,self.data_path.split('/')[-4])
        expected_path = os.path.join(sce_dir,str(self.frame_n)+'framesperseg'+os.path.split(self.data_path)[1].split('.')[0]+'.pickle')
        return expected_path
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
            set_list = []
            if self.end:
                print("agent list is built")
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
                    if set_num != self.frame_n:
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
                self.check_set.add(-int(line[self.track_pos][1:]) 
                                    if line[self.track_pos][0] == 'P' else int(line[self.track_pos]))
                j += 1
                if self.next_frame - self.current_frame == 1:
                    set_list.append(copy.copy(self.check_set))
                    self.check_set.clear()
                elif abs(self.next_frame - self.current_frame) > 1:
                    set_list.append(copy.copy(self.check_set))
                    self.check_set.clear()

                    set_num = len(set_list)
                    # print("*****************" + str(set_num))
                    if set_num != self.frame_n:
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
        self.check_set.clear()
        self.final_set_list.clear()
        return 
        # return self.track_num                           

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
           
    def consrtruct_features_all_car(self):
        whole_feature_map = dict()
        i = 0
        chip_id = 0
        self.frame_order = 0
        while i in range(self.data_len):
            j = 0
            feature_map_per_chip = {}
            if self.end:
                print("end")
                break
            feature_map_per_frame = []
            while True:
                line = self.data[min(i + j, len(self.data) - 1)]
                if len(line) == 12:
                    feature_vec = [ int(line[self.track_pos]),\
                                    float(line[self.x_pos]),float(line[self.y_pos]),float(line[self.vx_pos]),\
                                    float(line[self.vy_pos]),int(line[self.agent_type_pos] != 'car'),float(line[self.wid_pos]),float(line[self.len_pos]),\
                                    float(line[self.psi_pos]),int(line[self.agrole_pos]=='agent')]
                elif len(line) == 9:
                    feature_vec = [-int(line[self.track_pos][1:]),\
                                    float(line[self.x_pos]),float(line[self.y_pos]),float(line[self.vx_pos]),\
                                    float(line[self.vy_pos]),int(line[self.agent_type_pos] != 'car'),float(0.0),float(0.0),\
                                    float(np.arctan2(float(line[self.vy_pos]),float(line[self.vx_pos]))),int(0)]

                if i + j >= len(self.data) - 1:
                    feature_map_per_frame.append(copy.copy(feature_vec))
                    feature_map_per_chip[self.frame_order] = copy.copy(feature_map_per_frame)
                    self.end = True
                    frame_num = len(feature_map_per_chip)
                    if frame_num != self.frame_n:
                        feature_map_per_chip.clear()
                        feature_map_per_frame.clear()
                        break
                    new_frame = []
                    new_frame_chip = {}
                    agent_track =None
                    for idx_f in feature_map_per_chip:
                        idx_v = 0
                        frame = feature_map_per_chip[idx_f]
                        for single_track in frame:
                            if int(single_track[0]) == self.agent_list[chip_id]:
                                agent_track = single_track
                                agent_track_id = idx_v #delete the operation: track id/100
                            new_frame.append(copy.copy(single_track))
                            idx_v += 1
                        if agent_track_id != 0:
                            new_frame[agent_track_id] = new_frame[0]
                            new_frame[0] = agent_track

                        new_frame_chip[idx_f] = copy.copy(new_frame)                      
                        new_frame.clear()
                    whole_feature_map[chip_id] = copy.copy(new_frame_chip)
                    chip_id += 1
                    break
                next_line = self.data[i + j + 1]
                self.next_frame = int(next_line[self.frame_pos])
                self.last_frame = int(self.current_frame)
                self.current_frame = int(line[self.frame_pos])
                feature_map_per_frame.append(copy.copy(feature_vec))
                j += 1
                if self.next_frame - self.current_frame == 1:
                    feature_map_per_chip[self.frame_order]=copy.copy(feature_map_per_frame)
                    feature_map_per_frame.clear()
                    self.frame_order += 1
                elif abs(self.next_frame - self.current_frame) > 1:
                    feature_map_per_chip[self.frame_order]=copy.copy(feature_map_per_frame)
                    feature_map_per_frame.clear()
                    self.frame_order = 0

                    frame_num = len(feature_map_per_chip)
                    if frame_num != self.frame_n:
                        feature_map_per_chip.clear()
                        feature_map_per_frame.clear()
                        break
                    new_frame = []
                    new_frame_chip = {}
                    agent_track = None
                    normal_flag = True # if agent is suddenly changed, then false
                    for idx_f in feature_map_per_chip:
                        idx_v = 0
                        frame = feature_map_per_chip[idx_f]
                        agent_count = 0
                        for single_track in frame:
                            # print(single_track)
                            if single_track[0] == self.agent_list[chip_id]:
                                agent_count += 1
                                if agent_count > 1:
                                    normal_flag = False
                                    break
                                agent_track = single_track
                                agent_track_id = idx_v
                            new_frame.append(single_track)
                            idx_v += 1
                        if agent_count == 0:
                            normal_flag = False
                        if normal_flag == False:
                            print("hhh")
                            break
                        if agent_track_id != 0:
                            new_frame[agent_track_id] = new_frame[0]
                            new_frame[0] = agent_track
                            # print(str(round(new_frame[0][0]*100))+' ' + str(self.agent_list[chip_id]))
                            # print(new_frame[agent_track_id][0])
                        if normal_flag:
                            assert round(new_frame[0][0]) == int(self.agent_list[chip_id]),\
                                "agent is not at the first"+str(new_frame[0][0])+' '+str(self.agent_list[chip_id])
                            assert round(new_frame[0][-1]) == 1,"agent is not at the first"
                            new_frame_chip[idx_f] = copy.copy(new_frame)                       
                            new_frame.clear()
                    if not normal_flag:
                        feature_map_per_chip.clear()
                        feature_map_per_frame.clear()
                        chip_id += 1
                        break
                    whole_feature_map[chip_id] = copy.copy(new_frame_chip)
                    new_frame_chip.clear()
                    chip_id += 1
                    break
            i = i + j
        self.end = False
        base_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data'
        sce_dir = os.path.join(base_path,self.data_path.split('/')[-4])
        if not os.path.exists(sce_dir):
            os.makedirs(sce_dir)
        save_path = os.path.join(sce_dir,str(self.frame_n)+'framesperseg'+os.path.split(self.data_path)[1].split('.')[0]+'.pickle')
        with open(save_path, "wb") as fp:   #Pickling
            pickle.dump(whole_feature_map, fp, protocol = pickle.HIGHEST_PROTOCOL)
        
        self.agent_list.clear()
        self.check_set.clear()
        self.final_set_list.clear()
        print(len(whole_feature_map))
        return save_path 
        



    def construct_features(self):
        whole_feature_map = []
        i = 0
        chip_id = 0
        while i in range(self.data_len):
            j = 0
            feature_map_per_chip = []
            if self.end:
                print("end")
                break
            feature_map_per_frame = []
            while True:
                line = self.data[min(i + j, len(self.data) - 1)]
                feature_vec = [-int(line[self.track_pos][1:]) if line[self.track_pos][0] == 'P' else int(line[self.track_pos]),\
                                float(line[self.x_pos]),float(line[self.y_pos]),float(line[self.vx_pos]),\
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
                            if int(single_track[0]) in self.final_set_list[chip_id]:
                                if int(single_track[0]) == self.agent_list[chip_id]:
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
                    chip_id += 1
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
                            if int(single_track[0]) in self.final_set_list[chip_id]:
                                if int(single_track[0]) == self.agent_list[chip_id]:
                                    agent_track = single_track
                                    agent_track_id = idx_v
                                single_track[0] = float(single_track[0])/100.0
                                new_frame.append(single_track)
                                idx_v += 1
                        if agent_track_id != 0:
                            new_frame[agent_track_id] = new_frame[0]
                            new_frame[0] = agent_track
                            # print(str(round(new_frame[0][0]*100))+' ' + str(self.agent_list[chip_id]))
                            # print(new_frame[agent_track_id][0])
                        assert round(new_frame[0][0]*100) == int(self.agent_list[chip_id]),"agent is not at the first"
                        new_frame_chip.append(copy.copy(new_frame))                        
                        new_frame.clear()
                    whole_feature_map.append(copy.copy(new_frame_chip))
                    # if chip_id >= 21780:
                    #     print(np.array(whole_feature_map).shape)
                    #     print(np.array(whole_feature_map)[0][0][0][0])
                    # print(np.array(whole_feature_map).shape)
                    chip_id += 1
                    # print(chip_id)
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
    # csv_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/recorded_trackfiles/DR_CHN_Merging_ZS/train/segmented/tracks_000.csv'
    # csv_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/recorded_trackfiles/DR_USA_Intersection_EP0/train/segmented/tracks_000.csv'
    # csv_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/recorded_trackfiles/DR_DEU_Roundabout_OF/train/segmented/tracks_000.csv'
    # ff = feature_generator(40,10)
    # ff.set_path(csv_path)
    # ff.smallest_A_size_check()
    # print(ff.smallest_A_size_check())
    # # print(ff.data_iter_num)
    # # print(len(ff.agent_list))
    # ff.get_k_tracks()
    # print("get k tracks")
    # ff.consrtruct_features_all_car()
    # pp = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/track_1_feature.npy'
    # pp = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_CHN_Merging_ZS/40framespersegtracks_000.npy'
    # ff.prune(pp,13)
    dir_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/recorded_trackfiles'
    ff = feature_generator(40,10)
    for scenarios in os.listdir(dir_path):
        if scenarios[0] == 'D':
            scenarios = os.path.join(dir_path, scenarios)
            seg_dir = os.path.join(scenarios,'train/segmented')
            for files in sorted(os.listdir(seg_dir)):
                csv_path = os.path.join(seg_dir,files)
                if '/home/jonathon/Documents/new_project/interaction-dataset-master/recorded_trackfiles/DR_CHN_Merging_ZS' in csv_path:
                    continue
                print(csv_path)
                ex = ff.set_path(csv_path)
                if os.path.exists(ex):
                    print("done")
                    continue
                ff.smallest_A_size_check()
                ff.consrtruct_features_all_car()


                
            
    

