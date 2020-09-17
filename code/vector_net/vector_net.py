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
    def __init__(self, feature_length, device='cuda:0'):
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
        self.device = device
    def forward(self, data, osm, osm_interval):
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
        nCar = 10
        # osm map
        osm_subGraph_list = []
        for i in range(len(osm_interval)-1):
            osm_subGraph_list.append(self.subGraphs(osm[:, osm_interval[i]:osm_interval[i+1], :]).unsqueeze(1))
        osm_nodes = torch.cat(osm_subGraph_list, dim=1) #[batch, road_poly_num, features]
        # print(osm_nodes.shape)
        # Suppose we have 10 cars in each frame, then n = 10.
        all_batch_nodes = [] #[batch, nCar, node_feature]
        global_graphs = [] #[batch, features]
        for i in range(len(data)): # num of batch
            data1 = data[i]
            each_batch_nodes = [] # in different batch: different len
            for j in range(len(data1)): # num of trajs(cars that appear), traj0 = agent
                sub_node = self.subGraphs(data1[j].to(self.device).double().unsqueeze(0))
                each_batch_nodes.append(sub_node)
                # break
            # if len(each_batch_nodes) > nCar:
            #     each_batch_nodes = each_batch_nodes[:nCar]
            # else:
            #     to_add = nCar - len(each_batch_nodes)
            #     for _ in range(to_add): # add zero tensors to list if < nCar =10
            #         each_batch_nodes.append(torch.zeros_like(each_batch_nodes[0]).to(self.device).double())
            # all_batch_nodes.append(torch.cat(each_batch_nodes, dim=0)) # [nCar, node_feature]
            each_batch_nodes = torch.cat(each_batch_nodes, dim=0) #[numCar, node_features]
            each_batch_nodes_with_osm = torch.cat([each_batch_nodes, osm_nodes[i,:,:]], dim=0) #[numCar+numRoad, features]
            global_graphs.append(self.globalGraph(each_batch_nodes_with_osm.unsqueeze(0)))
        global_graphs = torch.cat(global_graphs,dim=0)
    
        # all_batch_nodes = torch.stack(all_batch_nodes)
        # all_batch_nodes_with_osm = torch.cat([all_batch_nodes, osm_nodes], dim=1) #[batch, nCar+nPolyRoad, features]
        
            
        
        # batch_size, vNum, feature_num = data.shape[0], data.shape[1], data.shape[2]
        
        
        # pids = data[0,:,-1].clone() # get all polylines' ids
        # with torch.no_grad():
        #     data[:,:,-1] = torch.zeros_like(data[:,:,-1]) # erase the track id info.
        
        # last_pid = pids[0]
        # intervals = [0]
        # for i in range(vNum):
        #     if last_pid != pids[i]:
        #         last_pid = pids[i]
        #         intervals.append(i)
        # intervals.append(vNum)
      
        # mini_result_list = []

        # for i in range(len(intervals)-1):
        #     mini_batch = data[:, intervals[i]:intervals[i+1], :]
        #     mini_result_list.append(self.subGraphs(mini_batch).unsqueeze(1))
        # batch_features = torch.cat(mini_result_list,dim=1)

        # result_features = self.globalGraph(batch_features)

        # result_features = self.globalGraph(all_batch_nodes_with_osm)

        return global_graphs



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

    def __init__(self, feature_length, timeStampNumber, device='cuda:0'):
        r"""
        Construct a VectorNet with predicting.
        :param feature_length: same as VectorNet.
        :param timeStampNumber: the length of time stamp for predicting the future trajectory.
        """
        super(VectorNetWithPredicting, self).__init__()
        self.device = device
        self.vectorNet = VectorNet(feature_length=feature_length)
        self.timeStamp = timeStampNumber
        self.hidden_size = 64
        self.car_feature = self.vectorNet.pLen #14
        self.trajDecoder =nn.Sequential(nn.Linear(self.vectorNet.pLen + self.car_feature, self.hidden_size),
                                    nn.LayerNorm(self.hidden_size),
                                    nn.ReLU(True),
                                    nn.Linear(self.hidden_size, self.hidden_size),
                                    nn.LayerNorm(self.hidden_size),
                                    nn.ReLU(True),
                                    nn.Linear(self.hidden_size, timeStampNumber * 2))
        self.subGraph_agent = SubGraph(layersNumber=3, feature_length=feature_length)
         #MLP.MLP(inputSize=self.vectorNet.pLen,outputSize=timeStampNumber * 2,noReLU=False)


    def forward(self, x, osm, osm_interval):
        r"""

        :param x: the same as VectorNet.
        :return: Future trajectory vector with length timeStampNumber*2, the form is (x1,y1,x2,y2,...).
        """
        # agent_his_traj = x[:,9,:].clone() # the number indicates last known frame
        # agent_his_traj[:,-1] = 0.0
        agent_his_batches = []
        for i in range(len(x)):
            agent_his_batches.append(x[i][0][:])
        agent_his_traj = torch.stack(agent_his_batches).to(self.device).double()
        agent_his_traj = self.subGraph_agent(agent_his_traj)
        # print(agent_his_traj.shape)
        # agent_his_traj = 

        x = self.vectorNet(x, osm, osm_interval)
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