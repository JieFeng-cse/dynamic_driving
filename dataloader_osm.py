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
        # print(self.data_set[1001,:, 5, 1:3])

        self.data_set, self.R, self.iR = self.RotateAndGetR_agents(self.data_set, train_frame) # change global coordinate to local coordinate
        # print(torch.matmul(self.R, self.iR))
        # print(self.data_set[1001,:, 5, 1:3])
        self.osm_data = torch.from_numpy(main_vector(osm_path))
        
        # print(self.data_set.shape)
        self.all_data_set = self.get_osm_style(self.data_set[:,:self.train_frame,:,:].clone(), train_frame, predict_frame)
        self.all_data_set = self.all_data_set.double()
        # print(self.all_data_set.shape)
        self.osm_data = self.osm_data.reshape([-1, 14]).unsqueeze(0).double()
        self.osm_data = self.osm_data.expand([self.all_data_set.shape[0], self.osm_data.shape[1], self.osm_data.shape[2]]).clone()
        # print(self.osm_data[0,0:2,0:4])
        self.osm_data = self.Rotate_osm(self.osm_data, self.R)

        # print(self.osm_data[0,0:2,0:4])
        # print(self.all_data_set.shape, self.osm_data.shape)
        # print(self.osm_data.shape)
        self.all_data_set = torch.cat([self.all_data_set, self.osm_data], dim=1)
        # print(self.all_data_set.shape)
    def __getitem__(self, index):
        X = self.all_data_set[index]
        label_list = []
        last_agent_loc_list = []
        for j in range(self.train_frame, self.train_frame+self.predict_frame):    
            for i in range(self.data_set[index][j].shape[0]):
                if self.data_set[index][j][i][-1] == 1 :
                    Lable = torch.stack([self.data_set[index][j][i][1],self.data_set[index][j][i][2]]).unsqueeze(0)
                    label_list.append(Lable)
                    # break
        for j in range(0, self.train_frame):    
            for i in range(self.data_set[index][j].shape[0]):
                if self.data_set[index][j][i][-1] == 1 :
                    last_agent_loc = torch.stack([self.data_set[index][j][i][1],self.data_set[index][j][i][2]]).unsqueeze(0)
                    last_agent_loc_list.append(last_agent_loc)
                    # break
        Lable = torch.cat(label_list, dim=0)
        last_agent_locs = torch.cat(last_agent_loc_list, dim=0)
        iR = self.iR[index]
        return (X, Lable, last_agent_locs, iR)
    def __len__(self):
        return self.data_set.shape[0]
    def RotateAndGetR_agents(self, data, train_frame):
        all_xy = data[:,:,:,1:3].clone()
        all_xy = torch.cat([all_xy, torch.ones_like(data[:,:,:,:1])], dim=-1)
        # print(all_xy.shape)
        all_agent_x = data[:,train_frame-1, 0, 1].clone()
        all_agent_y = data[:,train_frame-1, 0, 2].clone()
        all_agent_psi = data[:,train_frame-1, 0, -2].clone()
        R, iR = self.matrixR(all_agent_psi, all_agent_x, all_agent_y)
        Rout = R.clone()
        R = R.unsqueeze(1).unsqueeze(1)
        assert R.shape[-2] == 3 and R.shape[-1] == 3, 'error, R is not a transform matrix.'
        R = R.repeat(1, all_xy.shape[1], all_xy.shape[2], 1, 1)
        # print(R.shape, all_xy.shape)
        all_xy = torch.matmul(R, all_xy.unsqueeze(-1)).squeeze(-1)
        all_xy = all_xy[:,:,:,:2]
        data[:,:,:,1:3] = all_xy
        # print(all_xy.shape)
        return data, Rout, iR
    def Rotate_osm(self, data, R):
        R = R.unsqueeze(1)
        R = R.repeat(1,data.shape[1],1,1)

        all_starts = data[:,:,:2].clone()
        all_ends = data[:,:,2:4].clone()
        all_starts = torch.cat([all_starts, torch.ones_like(data[:,:,:1])], dim=-1)
        all_ends = torch.cat([all_ends, torch.ones_like(data[:,:,:1])], dim=-1)
        # print(R.shape)
        # print(all_starts.shape)
        all_starts = torch.matmul(R, all_starts.unsqueeze(-1)).squeeze(-1)
        all_ends = torch.matmul(R, all_ends.unsqueeze(-1)).squeeze(-1)

        all_starts = all_starts[:,:,:2]
        all_ends = all_ends[:,:,:2]

        data[:,:,:2] = all_starts
        data[:,:,2:4] = all_ends

        return data

    def matrixR(self, theta, x, y):
        cos = torch.cos(theta).clone()
        sin = torch.sin(theta).clone()
        RC1 = torch.stack([cos, -sin, torch.zeros_like(theta)], dim=1) # theta: [all num, ] -> [all num , 3]
        RC2 = torch.stack([sin, cos, torch.zeros_like(theta)], dim=1)
        RC3 = torch.stack([-x*cos-y*sin, x*sin-y*cos, torch.ones_like(theta)], dim=1)
        R = torch.stack([RC1, RC2, RC3], dim = 2) # theta: [all num , 3] -> [all num, 3, 3]
        # print(R.shape)
        iRC1 = torch.stack([cos, sin, torch.zeros_like(theta)], dim=1) # theta: [all num, ] -> [all num , 3]
        iRC2 = torch.stack([-sin, cos, torch.zeros_like(theta)], dim=1)
        iRC3 = torch.stack([x, y, torch.ones_like(theta)], dim=1)
        iR = torch.stack([iRC1, iRC2, iRC3], dim = 2) # theta: [all num , 3] -> [all num, 3, 3]

        return R, iR

    def get_osm_style(self, data, train_frame, predict_frame):
        data_copy = data.clone().detach()
        bu = torch.zeros_like(self.data_set[:,:train_frame,:,:4])
        data = torch.cat([data, bu], dim=3)
        # print(data_copy.shape)
        data[:,:,:,0] = data_copy[:,:,:,1]
        data[:,:,:,1] = data_copy[:,:,:,1] + data_copy[:,:,:,3]*0.1 # 0.001 scale, 0.1s
        data[:,:,:,2] = data_copy[:,:,:,2]
        data[:,:,:,3] = data_copy[:,:,:,2] + data_copy[:,:,:,4]*0.1

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
    # for i in range(vec.__len__()):
    #     x, lable = vec.__getitem__(i)
    #     print(x.shape)
    #     print(lable.shape)
    #     last_tckid = x[0,-1]
    #     counter = 1
    #     for j in range(x.shape[0]):
    #         if last_tckid != x[j][-1]:
    #             last_tckid = x[j][-1]
    #             counter += 1
    #     if counter != 115:
    #         print(counter)
    #     # break