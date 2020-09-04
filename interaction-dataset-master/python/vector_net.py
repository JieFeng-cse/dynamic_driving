import torch
from torch import nn
import torch.nn.functional as F
from sub_graph import SubGraph
from global_graph import Attention
import time

class VectorNet(nn.Module):

    r"""
    Vector network. 
    """

    # def __init__(self, len, pNumber):
    def __init__(self, feature_length):
        r"""
        Construct a VectorNet.
        :param feature_length: length of each vector v ([ds,de,a,j]).
        """
        super(VectorNet, self).__init__()
        layersNumber = 3
        # self.subGraphs = clones(SubGraph(layersNumber=3, feature_length=feature_length), 3)
        self.subGraphs = SubGraph(layersNumber=layersNumber, feature_length=feature_length)
        # self.pLen = feature_length
        self.pLen = feature_length * (2 ** layersNumber)
        self.globalGraph = Attention(C=self.pLen)
    def forward(self, data):
        r"""

        :param data: the input data of network. Each coordinate of key position is centered by
              predicted agent, and the first input feature vector is like [id,0,0,...,0], 'id'
              means the index of predicted agent, and other vectors are sorted by corresponding
              polyline index.

              For each batch, it looks like:
                [[id,0,...,0],
                 [a11,a12,...,a1k,a1_id],
                 [a21,a22,...,a2k,a2_id],
                 ...
                 [an1,an2,...,ank,an_id]]
              satisfied a(i)_id <= a(i+1)_id

              shape: data.shape = [batch size, vNumber, feature_length]
        :return: output
        """
        batch_size, vNum, feature_num = data.shape[0], data.shape[1], data.shape[2]
        
        pids = data[0,:,-1].clone() # get all polylines' ids
        with torch.no_grad():
            data[:,:,-1] = torch.zeros_like(data[:,:,-1]) # erase the track id info.
        
        last_pid = pids[0]
        intervals = [0]
        for i in range(vNum):
            if last_pid != pids[i]:
                last_pid = pids[i]
                intervals.append(i)
        intervals.append(vNum)
        # print(intervals)
        # print(len(intervals))
        mini_result_list = []
        # flag = 0
        for i in range(len(intervals)-1):
            # if flag:
            #     mini_batch = data[:, intervals[i]:intervals[i+1], :]
            #     batch_features = torch.cat([batch_features,self.subGraphs(mini_batch).unsqueeze(1)],dim=1)
            # else:
            #     flag = 1
            #     mini_batch = data[:, intervals[i]:intervals[i+1], :]
            #     batch_features = self.subGraphs(mini_batch).unsqueeze(1)
            mini_batch = data[:, intervals[i]:intervals[i+1], :]
            mini_result_list.append(self.subGraphs(mini_batch).unsqueeze(1))
        batch_features = torch.cat(mini_result_list,dim=1)
        # print(batch_features.shape)

        # batch_feature_list = []
        # # print(data.shape)
        # st_time = time.time()
        # for batch_id in range(batch_size):
        #     one_data = data[batch_id]
        #     last_trk_id = one_data[0][-1]
        #     last_frame = one_data[0].unsqueeze(0).unsqueeze(0)
        #     # print(last_frame.shape)
        #     sub_graph_result_list = []
        #     for vec_num in range(one_data.shape[0]):
        #         if last_trk_id != one_data[vec_num,-1]:
        #             # print(last_frame.shape)
        #             sub_graph_result_list.append(self.subGraphs(last_frame))
        #             last_trk_id = one_data[vec_num,-1]
        #             last_frame = one_data[vec_num].unsqueeze(0).unsqueeze(0)
        #         else:
        #             last_frame = torch.cat([one_data[vec_num].unsqueeze(0).unsqueeze(0), last_frame], dim=1)
        #     sub_graph_result_list.append(self.subGraphs(last_frame)) # to [poly_num, encoded features]

        #     encoded_features = torch.cat(sub_graph_result_list, dim=0).unsqueeze(0)
        #     batch_feature_list.append(encoded_features)
        # print('main vecnet time cost:', time.time()-st_time)
        # batch_features = torch.cat(batch_feature_list, dim=0) #[batchsize, poly_num, encoded features]
        # print(batch_features.shape)
        result_features = self.globalGraph(batch_features)

        return result_features



        '''
        origin code
        '''
        # data = data.permute(1, 0, 2)  # [vNumber, batch size, len]
        # # id = data[0, :, 0].long()
        # pID = data[:, 0, -1].long()
        # data[:, :, -1] = 0

        # batchSize, len = data.shape[1], data.shape[2]
        # P = torch.zeros(batchSize, 0, self.pLen).to(device)

        # j = 1
        # for i in range(1, data.shape[0]):
        #     if i + 1 == data.shape[0] or \
        #             pID[i] != pID[i + 1]:
        #         tmp = torch.zeros(batchSize, 0, len).to(device)
        #         while j <= i:
        #             t = data[j]  # [batch size, len]
        #             t.unsqueeze_(1)  # [batch size, 1, len]
        #             tmp = torch.cat((tmp, t), dim=1)
        #             j += 1

        #         # tmp's shape is [batch size, pvNumber, Len]
        #         # subGraphId = int(data[i, 0, len - 1].item())
        #         # print(tmp.shape)
        #         p = self.subGraphs(tmp)  # [batch size, pLen]
        #         p.unsqueeze_(1)  # [batch size, 1, pLen]
        #         P = torch.cat((P, p), dim=1)
        #         # print('2 VectorNet',i,j, 'subGraphId =',subGraphId)

        # # P's shape is [batch size, pNumber, pLen]
        # # P = F.normalize(P, dim=2)
        # feature = self.globalGraph(P)  # [batch size, pLen]
        # # print(feature.device)
        # # print(feature.shape)
        # # raise NotImplementedError
        # return feature
class VectorNetWithPredicting(nn.Module):

    r"""
      A class for packaging the VectorNet and future trajectory prediction module.
      The future trajectory prediction module uses MLP without ReLu(because we
    hope the coordinate of trajectory can be negative).
    """

    def __init__(self, feature_length, timeStampNumber):
        r"""
        Construct a VectorNet with predicting.
        :param feature_length: same as VectorNet.
        :param timeStampNumber: the length of time stamp for predicting the future trajectory.
        """
        super(VectorNetWithPredicting, self).__init__()
        self.vectorNet = VectorNet(feature_length=feature_length)
        self.timeStamp = timeStampNumber
        self.hidden_size = 64
        self.car_feature = 14
        self.trajDecoder =nn.Sequential(nn.Linear(self.vectorNet.pLen + self.car_feature, self.hidden_size),
                                    nn.LayerNorm(self.hidden_size),
                                    nn.ReLU(True),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.LayerNorm(self.hidden_size),
                                    nn.ReLU(True),
                                    nn.Linear(self.hidden_size, timeStampNumber * 2))
        
         #MLP.MLP(inputSize=self.vectorNet.pLen,outputSize=timeStampNumber * 2,noReLU=False)


    def forward(self, x):
        r"""

        :param x: the same as VectorNet.
        :return: Future trajectory vector with length timeStampNumber*2, the form is (x1,y1,x2,y2,...).
        """
        agent_his_traj = x[:,19,:].clone()
        agent_his_traj[:,-1] = 0.0

        x = self.vectorNet(x)
        x = torch.cat([x, agent_his_traj], dim=1)
        x = self.trajDecoder(x)
        x = x.reshape([x.shape[0], self.timeStamp, 2])
        # for i in range(1, x.shape[1]):
        #     x[:,i,:] = x[:, i-1,:] + x[:,i,:]
        return x


class VectorNetAndTargetPredicting(nn.Module):
    def __init__(self, feature_length):
        super(VectorNetAndTargetPredicting, self).__init__()
        self.vectornet = VectorNet(feature_length=feature_length)
        self.car_feature = 14
        self.hidden_size = 150
        self.targetPred = nn.Sequential(nn.Linear(self.vectornet.pLen + self.car_feature, self.hidden_size),
                                        nn.LayerNorm(self.hidden_size),
                                        nn.ReLU(True),
                                        nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.LayerNorm(self.hidden_size),
                                        nn.ReLU(True),
                                        nn.Linear(self.hidden_size, 2)) # output (x,y)
    def forward(self, x):
        agent_his_traj = x[:,19,:].clone()
        agent_his_traj[:,-1] = 0.0
        x = self.vectornet(x)
        x = torch.cat([x, agent_his_traj], dim=1)
        x = self.targetPred(x)

        return x