import numpy as np
import torch
import torch.utils.data as Data
from extract_osm import *

class vectors_car_road(Data.Dataset):
    "just for now"
    def __init__(self, osm_path, npy_path, train_frame, predict_frame):
        self.train_frame = train_frame
        self.predict_frame = predict_frame
        self.data_set = torch.tensor(np.load(npy_path))
        self.osm_data = torch.from_numpy(main_vector(osm_path))
        # print(self.data_set.shape)
        self.all_data_set = self.get_osm_style(self.data_set[:,:self.train_frame,:,:], train_frame, predict_frame)
        self.all_data_set = self.all_data_set.float()
        # print(self.all_data_set.shape)
        self.osm_data = self.osm_data.reshape([-1, 14]).unsqueeze(0).float()
        self.osm_data = self.osm_data.expand([self.all_data_set.shape[0], self.osm_data.shape[1], self.osm_data.shape[2]])
        # print(self.osm_data.shape)
        self.all_data_set = torch.cat([self.all_data_set, self.osm_data], dim=1)
        # print(self.all_data_set.shape)
    def __getitem__(self, index):
        X = self.all_data_set[index]
        label_list = []
        last_agent_loc_list = []
        # for j in range(self.train_frame, self.train_frame+self.predict_frame):    
        #     for i in range(self.data_set[index][j].shape[0]):
        #         if self.data_set[index][j][i][-1] == 1 :
        #             Lable = torch.stack([self.data_set[index][j][i][1],self.data_set[index][j][i][2]]).unsqueeze(0)
        #             label_list.append(Lable)
        #             # break
        # for j in range(0, self.train_frame):    
        #     for i in range(self.data_set[index][j].shape[0]):
        #         if self.data_set[index][j][i][-1] == 1 :
        #             last_agent_loc = torch.stack([self.data_set[index][j][i][1],self.data_set[index][j][i][2]]).unsqueeze(0)
        #             last_agent_loc_list.append(last_agent_loc)
                    # break
        Labels = []
        for i in range(0, self.train_frame + self.predict_frame):
            assert self.data_set[index,i,0,-1] == 1, 'the first node is not agent!'
            Labels.append(torch.tensor([self.data_set[index,i,0,1],self.data_set[index,i,0,2],self.data_set[index,i,0,3],self.data_set[index,i,0,4],self.data_set[index,i,0,8]]))
        Labels = torch.stack(Labels)

        # Lable = torch.cat(label_list, dim=0)
        # last_agent_locs = torch.cat(last_agent_loc_list, dim=0)
        return (X, Labels)
    def __len__(self):
        return self.data_set.shape[0]
    def get_osm_style(self, data, train_frame, predict_frame):
        data_copy = data.clone().detach()
        bu = torch.zeros_like(self.data_set[:,:train_frame,:,:4])
        data = torch.cat([data, bu], dim=3)
        # print(data_copy.shape)
        data[:,:,:,0] = data_copy[:,:,:,1]
        data[:,:,:,1] = data_copy[:,:,:,1] + data_copy[:,:,:,3]*0.0001 # 0.001 scale, 0.1s
        data[:,:,:,2] = data_copy[:,:,:,2]
        data[:,:,:,3] = data_copy[:,:,:,2] + data_copy[:,:,:,4]*0.0001

        data[:,:,:,4] = torch.ones_like(data_copy[:,:,:,0]) # for cars and peds. not map
        for i in range(data.shape[1]):
            data[:,i,:,5] = torch.ones_like(data_copy[:,i,:,0])*(i+1)*0.1 #timestamp using frame id
        # data[:,:,:,5] = data_copy[:,:,:,0]
        data[:,:,:,6:13] = data_copy[:,:,:,3:]
        data[:,:,:,13] = data_copy[:,:,:,0]
        # print(data[0,38,6,:])
        data = data.permute(0,2,1,3) # exchange the mid two dims. [index, frame, car id, features] -> [index, car id, frame, features]
        data = data.reshape([data.shape[0], -1, data.shape[-1]])
        # print(data.shape)
        return data

if __name__ == "__main__":
    vec = vectors_car_road('./maps/DR_CHN_Merging_ZS.osm', 'data/40framespersegtracks_000.npy', 30, 10)
    print('len:', vec.__len__())
    for i in range(vec.__len__()):
        x, lable = vec.__getitem__(i)
        print(x.shape)
        print(lable.shape)
        last_tckid = x[0,-1]
        counter = 1
        for j in range(x.shape[0]):
            if last_tckid != x[j][-1]:
                last_tckid = x[j][-1]
                counter += 1
        if counter != 115:
            print(counter)
        # break
