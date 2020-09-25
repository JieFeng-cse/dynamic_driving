import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model import proj_mat
from vector_net import VectorNet, VectorNetWithPredicting
from model import rnn_encoder,rnn_decoder,rnn_model,dynamic_model

def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'tanh')
        nn.init.constant_(m.bias, 0.001)
class MLP_COS(nn.Module):
    def __init__(self):
        super(MLP_COS, self).__init__()
        self.linear1 = nn.Linear(256+2, 256)
        # self.linear2 = nn.Linear(256, 256)
        # self.linear3 = nn.Linear(256, 256)
        # self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 2)
        
        self.apply(weights_init)
        
    def forward(self, x, t, v0):
        x = torch.cat([x, t], dim=1)
        x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        #x = F.leaky_relu(x)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        # x = self.linear2(x)
        #x = F.leaky_relu(x)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        # x = self.linear3(x)
        # #x = F.leaky_relu(x)
        # x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        # x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        return x
    
class Model_COS(nn.Module):
    def __init__(self,args,device):
        super(Model_COS, self).__init__()
        self.frame_n = args.frame_n
        self.frame_gap = args.frame_gap
        self.device = device
        self.vectornet = VectorNet(14)
        self.vectornet_pre = VectorNetWithPredicting(feature_length=14, timeStampNumber=30)
        self.mlp = MLP_COS()
        self.proj = nn.Linear(112+14,256)
        self.norm = nn.LayerNorm(256)
        self.rnn_encoder = rnn_encoder().double()
        self.rnn_decoder = rnn_decoder().double()
        self.rnn_model = rnn_model(
            self.rnn_encoder, self.rnn_decoder, self.device).double()
        self.alpha = nn.Parameter(torch.rand([1]))
        self.beta = nn.Parameter(torch.rand([1]))
        self.situ_pro = nn.Linear(26,2)
    
    def forward(self, x, t, v0, state):
        # print(x[0]._version)
        # agent_his_traj = x[0][:, 9, :].clone()
        # agent_his_traj[:, -1] = 0.0
        batch_size, vNum, feature_num = x[0].shape[0], x[0].shape[1], x[0].shape[2]
        
        pids = x[0][0,:,-1].clone() # get all polylines' ids
        
        last_pid = pids[0]
        intervals = [0]
        for i in range(vNum):
            if last_pid != pids[i]:
                last_pid = pids[i]
                intervals.append(i)
        intervals.append(vNum)
        pos = []
        vel = []
        psi_l = []
        for i in range(len(intervals)-1):
            if torch.sum(x[0][:, intervals[i], 4]) == x[0].shape[0]:
                pos.append(x[0][:, intervals[i+1]-1, 0:2])
                vel.append(x[0][:, intervals[i+1]-1, 6:8])
                psi_l.append(x[0][:, intervals[i+1]-1, 11])
            else:
                break
        pos = torch.stack(pos,dim=1)
        psi_l = torch.stack(psi_l,dim=1)
        vel = torch.stack(vel,dim=1)
        psi_l = torch.abs(psi_l-psi_l[:,0].unsqueeze(1).repeat(1,psi_l.shape[1])) #the difference of psi
        dir_att = torch.zeros_like(psi_l)
        dir_att[(psi_l<2.7)] = 1.0
        dir_att[(psi_l>3.5)] = 1.0

        dis = torch.sqrt((pos - pos[:,0,:].unsqueeze(1).repeat(1,pos.shape[1],1)).pow(2).sum(dim=-1))
        dis_score = torch.exp(-dis)
        dis_score[dis_score<math.exp(-30)] = 0
        dis_score[dir_att == 0] = 0

        vel_dis = torch.sqrt((vel - vel[:,0,:].unsqueeze(1).repeat(1,vel.shape[1],1)).pow(2).sum(dim=-1))
        status_confirm = torch.cat([dis_score,vel_dis],dim=1)
        # print(status_confirm.shape)
        two_pra = self.situ_pro(status_confirm)
        two_pra = torch.softmax(two_pra,dim=-1)

        encodered_agent_features = []  # seq_len * batch_size * node_dim
        gt_agent_features = []
        
        for i in range(self.frame_n):
            if i < self.frame_gap:
                agent_feature = x[1][i, :, :]
                encodered_agent_features.append(agent_feature.clone())
                if i == self.frame_gap - 1:
                    gt_agent_features.append(agent_feature.clone())
            else:
                gt_feature = x[1][i, :, :]
                gt_agent_features.append(gt_feature.clone())
        encodered_agent_features = torch.stack(
            encodered_agent_features).to(self.device)
        gt_agent_features = torch.stack(gt_agent_features).to(
            self.device)  # seq_len * batch_size * node_dim
        # print(encodered_agent_features.shape)
        # print(gt_agent_features.shape)
        predictions = self.rnn_model(
            encodered_agent_features, gt_agent_features, state)
        predictions = predictions.permute(1,0,2)
        # x = self.vectornet(x[0]) #batch * 112
        # x = torch.cat([x, agent_his_traj], dim=1)
        vec_pre = self.vectornet_pre(x[0])
        vec_pre = vec_pre
        #seq_len * batch * 2
        # out, hidden, cell = self.rnn(x[1])
        # out = out.permute(1,0,2)
        # out = out.reshape(out.shape[0],-1)
        # print(out.shape)
        # x = torch.cat([xx,hidden[0],hidden[1],cell[0],cell[1],out],dim=1)
        # print(x.shape)

        # x = self.proj(x)
        # x = self.norm(x)
        # x = torch.tanh(x)

        # x = torch.zeros([t.shape[0],1]).to('cuda:0').double()
        #x = x.view(-1, 512)
        # trace = []
        # for i in range(t.shape[1]):
        #     pos = self.mlp(x, t[:,i].unsqueeze(-1).clone(), v0.unsqueeze(-1).clone())
        #     trace.append(pos)
        
        # trace = torch.stack(trace).double()
        alpha = two_pra[:,0].unsqueeze(-1).unsqueeze(-1).repeat(1,vec_pre.shape[1],vec_pre.shape[2])
        beta = two_pra[:,1].unsqueeze(-1).unsqueeze(-1).repeat(1,vec_pre.shape[1],vec_pre.shape[2])
        final_trace =  alpha * vec_pre + beta * predictions
        return final_trace