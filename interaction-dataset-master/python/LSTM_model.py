import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from model import *
from layer import DeGINConv, DenseGCNConv


class graph_attention(nn.Module):
    def __init__(self, Max_node_num, device, feature_dim=10):
        """
        Build the graph adjecency matrix based on some rules
        use the rules (attention) to determine the difference of importance of different nodes

        args:
        Max_node_num is the max number of cars involved in this scenario, the number is supposed to be the upper bound and then the matrix 
        size can be stable;
        device is the device we will need, cuda or cpu
        feature dim is the dim of input features
        """
        super(graph_attention, self).__init__()
        self.Max_node_num = Max_node_num
        self.feature_dim = feature_dim
        self.device = device
        self.dis_thre = 30
        self.direction_att = nn.Sequential(nn.Linear(2, 4), nn.ReLU(True),
                                           nn.Softmax(dim=-1), nn.Linear(4, 1))
        self.feat_proj = nn.Linear(10, 1)
        self.vel_proj = nn.Linear(1, 1)

    def calc_pairwise_distance(self, X, Y):
        """
        computes pairwise distance between each element
        Args: 
            X: [B,N,D]
            Y: [B,M,D]
        Returns:
            dist: [B,N,M] matrix of euclidean distances
        """
        B = X.shape[0]

        rx = X.pow(2).sum(dim=2).reshape((B, -1, 1)).to(self.device)
        ry = Y.pow(2).sum(dim=2).reshape((B, -1, 1)).to(self.device)

        dist = rx-2.0*X.matmul(Y.transpose(1, 2))+ry.transpose(1, 2)

        return torch.sqrt(dist + 1e-8)

    def cal_dis_metrix(self, X):
        """
        computes adjecency matrix, distance mask
        """
        dis_metrix = np.zeros(
            (X.shape[0], self.Max_node_num, self.Max_node_num))
        dis_metrix = torch.tensor(dis_metrix).to(self.device)
        pos = X
        dis_metrix = self.calc_pairwise_distance(pos, pos)
        return dis_metrix

    def forward(self, X):
        # X: B * n * dim (specific time)
        dir_metrixs = []
        feat_metrixs = []
        dir_col = X[:, :, 8].clone().unsqueeze(-1)
        # 8 is the index of psi, B* n * 1
        feat_col = torch.cat(
            [X[:, :, 0].unsqueeze(-1), X[:, :, 5:8], X[:, :, 9].unsqueeze(-1)], dim=2)
        #batch * n * 5
        for i in range(X.shape[1]):
            tmp_dir = dir_col[:, i].unsqueeze(1).repeat([1, X.shape[1], 1])
            tmp_feat = feat_col[:, i].unsqueeze(1).repeat([1, X.shape[1], 1])
            #B * n * dim
            con_dir = torch.cat([dir_col, tmp_dir], dim=2)
            dir_metrixs.append(con_dir)
            con_feat = torch.cat([feat_col, tmp_feat], dim=2)
            feat_metrixs.append(con_feat)
        dir_metrixs = torch.stack(dir_metrixs, dim=2)
        feat_metrixs = torch.stack(feat_metrixs, dim=2)  # B*n*n*10
        # dir_metrixs -> B * n * n * 2, direction pairs
        dir_scores = self.direction_att(dir_metrixs).squeeze()  # B * n * n * 1
        feat_scores = self.feat_proj(feat_metrixs).squeeze()
        dis_metrix = self.cal_dis_metrix(X[:, :, 1:3].clone())
        vel_metrix = self.cal_dis_metrix(X[:, :, 3:5].clone()).unsqueeze(-1)
        vel_scores = self.vel_proj(vel_metrix).squeeze()

        # dis_metrix[dis_metrix==0] = 1.0 # assign one to digonal
        relation_graph = 30 / (dis_metrix + 0.0001)

        relation_graph = relation_graph * dir_scores
        relation_graph = relation_graph * feat_scores
        relation_graph = relation_graph * vel_scores

        dis_mask = (dis_metrix > 30)
        relation_graph[dis_mask] = -float('inf')
        relation_graph = torch.softmax(relation_graph, dim=2)
        # print(dir_metrixs[0][0])
        # print(dis_metrix[0][0])
        # print(relation_graph[0][0])
        deg = (relation_graph != 0).sum(dim=2)-1
        return deg, relation_graph


class new_model(nn.Module):
    """
    This is a model incorperate gnn and lstm
    """

    def __init__(self, args, device, dropout=0.3, node_dim=10, Max_node_num=13):
        super(new_model, self).__init__()
        self.dropout = dropout
        self.frame_n = args.frame_n
        self.frame_gap = args.frame_gap
        self.Max_node_num = Max_node_num
        self.device = device
        self.rnn_num = 4
        if args.normalization:
            self.layer_norm = nn.LayerNorm([self.Max_node_num, 16])
        self.gc = graph_attention(Max_node_num, device)
        self.num_graph = args.num_graph
        self.gcn = DenseGCNConv(4, 16)
        self.agent_encoder = rnn_encoder()
        self.agent_decoder = rnn_decoder()
        self.hidden = 16
        self.input = 16
        self.agent_rnn = rnn_model(
            self.agent_encoder, self.agent_decoder, self.device)
        self.encoders = torch.nn.ModuleList([rnn_encoder(
            input_dim=self.input, hid_dim=self.hidden) for i in range(self.rnn_num)])
        self.decoders = torch.nn.ModuleList([rnn_decoder(
            output_dim=self.hidden, hid_dim=self.hidden) for i in range(self.rnn_num)])
        self.rnns = torch.nn.ModuleList([rnn_model(self.encoders[i], self.decoders[i], self.device, input_dim=self.input, hid_dim=16)
                                         for i in range(self.rnn_num)])
        self.proj = nn.Linear(8, 2)
        self.proj2 = nn.Linear(4,2)

    def forward(self, X, state):
        psi = X[:, 0, :, 8]
        transed_X = []
        for i in range(X.shape[2]):
            R = proj_mat(psi[:, i], self.device, X[:, 0, i,
                                                   1].clone(), X[:, 0, i, 2].clone())
            if i == 0:
                R_agent = R
            pos_transed = torch.matmul(R.unsqueeze_(1).repeat(1, 40, 1, 1), torch.cat([X[:, :, i, 1:3].unsqueeze(
                2), torch.ones([X.shape[0], X.shape[1], 1, 1]).to(self.device).double()], dim=3).transpose(2, 3))
            pos_transed = pos_transed.squeeze()[:, :, 0:2]
            vel = X[:, :, i, 3:5].clone()
            psi = psi[:, i].unsqueeze(-1).repeat([1, 40])
            vel_transed = vel.clone()
            vel_transed[:, :, 0] = torch.mul(vel[:, :, 0], torch.cos(
                psi)) + torch.mul(vel[:, :, 1], torch.sin(psi))
            vel_transed[:, :, 1] = torch.mul(-vel[:, :, 0], torch.sin(
                psi)) + torch.mul(vel[:, :, 1], torch.cos(psi))
            transed_X.append(torch.cat([pos_transed, vel_transed], dim=2))
        transed_X = torch.stack(transed_X, dim=2)
        #batchszie * seq_len * node_num * dim(4)
        encodered_agent_features = []  # seq_len * batch_size * node_dim
        gt_agent_features = []
        encodered_f1 = []
        encodered_f2 = []
        encodered_f3 = []
        encodered_f4 = []

        gt_f1 = []
        gt_f2 = []
        gt_f3 = []
        gt_f4 = []
        for i in range(self.frame_n):
            if i < self.frame_gap:
                encodered_agent_features.append(transed_X[:, i, 0].clone())
                deg, encoded_adj = self.gc(
                    X[:, i])[0], F.relu(self.gc(X[:, i])[1])
                #deg: batch*n
                x1 = F.relu(self.gcn(transed_X[:, i].clone(), encoded_adj))
                x1 = self.layer_norm(x1)
                x1 = F.tanh(x1)
                # x1: batch * n * hidden size
                idx = (deg >= 4)
                ze = torch.zeros([x1.shape[2]]).double().to(self.device)
                tmp_x = x1.clone()
                tmp_x[idx] = ze
                tmp_x = tmp_x.sum(
                    dim=1)/(x1.shape[1]-idx.sum(dim=1)+0.001).unsqueeze(-1).repeat([1, x1.shape[2]])
                encodered_f1.append(tmp_x.clone())

                idx = (deg < 4)
                tmp_x = x1.clone()
                tmp_x[idx] = ze
                idx_sum = idx.sum(dim=1)
                idx = (deg >= 7)
                tmp_x[idx] = ze
                idx_sum += idx.sum(dim=1)
                tmp_x = tmp_x.sum(
                    dim=1)/(x1.shape[1]-idx_sum+0.001).unsqueeze(-1).repeat([1, x1.shape[2]])
                encodered_f2.append(tmp_x.clone())

                idx = (deg < 7)
                tmp_x = x1.clone()
                tmp_x[idx] = ze
                idx_sum = idx.sum(dim=1)
                idx = (deg >= 10)
                tmp_x[idx] = ze
                idx_sum += idx.sum(dim=1)
                tmp_x = tmp_x.sum(
                    dim=1)/(x1.shape[1]-idx_sum+0.001).unsqueeze(-1).repeat([1, x1.shape[2]])
                encodered_f3.append(tmp_x.clone())

                idx = (deg < 10)
                tmp_x = x1.clone()
                tmp_x[idx] = ze
                tmp_x = tmp_x.sum(
                    dim=1)/(x1.shape[1]-idx.sum(dim=1)+0.001).unsqueeze(-1).repeat([1, x1.shape[2]])
                encodered_f4.append(tmp_x.clone())
                if i == self.frame_gap - 1:
                    gt_agent_features.append(transed_X[:, i, 0].clone())
                    gt_f1.append(encodered_f1[-1])
                    gt_f2.append(encodered_f2[-1])
                    gt_f3.append(encodered_f3[-1])
                    gt_f4.append(encodered_f4[-1])
            else:
                gt_agent_features.append(transed_X[:, i, 0].clone())
                deg, encoded_adj = self.gc(
                    X[:, i])[0], F.relu(self.gc(X[:, i])[1])
                #deg: batch*n
                x1 = F.relu(self.gcn(transed_X[:, i].clone(), encoded_adj))
                x1 = self.layer_norm(x1)
                x1 = F.tanh(x1)
                # x1: batch * n * hidden size
                idx = (deg >= 4)
                ze = torch.zeros([x1.shape[2]]).double().to(self.device)
                tmp_x = x1.clone()
                tmp_x[idx] = ze
                tmp_x = tmp_x.sum(
                    dim=1)/(x1.shape[1]-idx.sum(dim=1)+0.001).unsqueeze(-1).repeat([1, x1.shape[2]])
                gt_f1.append(tmp_x.clone())

                idx = (deg < 4)
                tmp_x = x1.clone()
                tmp_x[idx] = ze
                idx_sum = idx.sum(dim=1)
                idx = (deg >= 7)
                tmp_x[idx] = ze
                idx_sum += idx.sum(dim=1)
                tmp_x = tmp_x.sum(
                    dim=1)/(x1.shape[1]-idx_sum+0.001).unsqueeze(-1).repeat([1, x1.shape[2]])
                gt_f2.append(tmp_x.clone())

                idx = (deg < 7)
                tmp_x = x1.clone()
                tmp_x[idx] = ze
                idx_sum = idx.sum(dim=1)
                idx = (deg >= 10)
                tmp_x[idx] = ze
                idx_sum += idx.sum(dim=1)
                tmp_x = tmp_x.sum(
                    dim=1)/(x1.shape[1]-idx_sum+0.001).unsqueeze(-1).repeat([1, x1.shape[2]])
                gt_f3.append(tmp_x.clone())

                idx = (deg < 10)
                tmp_x = x1.clone()
                tmp_x[idx] = ze
                tmp_x = tmp_x.sum(
                    dim=1)/(x1.shape[1]-idx.sum(dim=1)+0.001).unsqueeze(-1).repeat([1, x1.shape[2]])
                gt_f4.append(tmp_x.clone())
        encodered_agent_features = torch.stack(
            encodered_agent_features).to(self.device)
        gt_agent_features = torch.stack(gt_agent_features).to(
            self.device)  # seq_len * batch_size * node_dim
        encodered_f1 = torch.stack(encodered_f1).to(self.device)
        encodered_f2 = torch.stack(encodered_f2).to(self.device)
        encodered_f3 = torch.stack(encodered_f3).to(self.device)
        encodered_f4 = torch.stack(encodered_f4).to(self.device)
        gt_f1 = torch.stack(gt_f1).to(self.device)
        gt_f2 = torch.stack(gt_f2).to(self.device)
        gt_f3 = torch.stack(gt_f3).to(self.device)
        gt_f4 = torch.stack(gt_f4).to(self.device)
        #en: seq_len(src) * batch_size * dim
        #gt: seq_len(trg) * batch_size * dim
        pre_agent = self.agent_rnn(encodered_agent_features, gt_agent_features, state)
        pre_f1 = self.rnns[0](encodered_f1, gt_f1, state)
        pre_f2 = self.rnns[1](encodered_f2, gt_f2, state)
        pre_f3 = self.rnns[2](encodered_f3, gt_f3, state)
        pre_f4 = self.rnns[3](encodered_f4, gt_f4, state)

        predictions = torch.cat([pre_f1, pre_f2, pre_f3, pre_f4], dim=-1)
        predictions = F.tanh(self.proj(predictions))
        predictions = self.proj2(torch.cat([predictions,pre_agent],dim=-1))
        return R_agent, predictions
