import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from layer import DeGINConv, DenseGCNConv
# from MTlayer import *
import pickle
import os
import scipy.sparse as sp
from scipy.sparse import linalg
from torch.autograd import Variable
import random


def proj_mat(theta, device, X, Y):
    X = X.unsqueeze_(-1).unsqueeze_(-1)
    Y = Y.unsqueeze_(-1).unsqueeze_(-1)
    R11 = torch.cos(theta).unsqueeze_(-1).unsqueeze_(-1)
    R12 = torch.sin(theta).unsqueeze_(-1).unsqueeze_(-1)
    R21 = -torch.sin(theta).unsqueeze_(-1).unsqueeze_(-1)
    R22 = torch.cos(theta).unsqueeze_(-1).unsqueeze_(-1)

    R1 = torch.cat([R11, R12, -R11*X-R12*Y], dim=2)
    R2 = torch.cat([R21, R22, -R21*X-R22*Y], dim=2)
    R3 = torch.zeros([X.shape[0], 1, 3]).to(device).double()
    R3[:, :, 2] = 1
    R = torch.cat([R1, R2, R3], dim=1)
    return R


def proj_mat_inverse(R, device):
    R_inversed = R.clone().transpose(2, 1).to(device)
    R_inversed[:, 2, :] = torch.tensor([0.0, 0.0, 1.0])
    tmp = torch.matmul(-R_inversed[:, 0:2, 0:2],
                       R.clone()[:, 0:2, 2].unsqueeze(-1))
    R_inversed[:, 0:2, 2] = tmp[:, :, 0]
    return R_inversed


class rnn_encoder(nn.Module):
    def __init__(self, input_dim=4, hid_dim=4, n_layers=2, n_dirs=2, dropout=0):
        super(rnn_encoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_dirs = n_dirs
        self.ln = nn.Linear(input_dim, input_dim)
        self.rnn = nn.LSTM(input_dim, hid_dim, n_layers, dropout=dropout,
                           bidirectional=True)
        for i in range(len(self.rnn.all_weights)):
            nn.init.orthogonal_(self.rnn.all_weights[i][0])

    def forward(self, x):
        x = self.ln(x)
        outputs, (hidden, cell) = self.rnn(x)
        hidden = torch.sum(hidden.view(
            self.n_layers, self.n_dirs, -1, self.hid_dim), dim=1)
        cell = torch.sum(
            cell.view(self.n_layers, self.n_dirs, -1, self.hid_dim), dim=1)
        outputs = torch.sum(outputs.view(
            10, -1, self.n_dirs, self.hid_dim), dim=2)  # 10 is seq len
        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        return outputs, hidden, cell


class RNN_Attention(nn.Module):
    def __init__(self, hidden_size):
        super(RNN_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        hidden = torch.mean(hidden, dim=0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class rnn_decoder(nn.Module):
    def __init__(self, output_dim=4, hid_dim=4, n_layers=2, dropout=0):
        super(rnn_decoder, self).__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.input_dim = 4
        self.attention = RNN_Attention(self.hid_dim)
        self.rnn = nn.LSTM(output_dim, hid_dim, n_layers, dropout=dropout)
        for i in range(len(self.rnn.all_weights)):
            nn.init.orthogonal_(self.rnn.all_weights[i][0])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)  # input = [1, batch size]
        # attn_weights = self.attention(hidden,encoder_outputs)
        # context = attn_weights.bmm(encoder_outputs.transpose(0,1))
        # context = context.transpose(0,1) #(1,b,n)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        prediction = F.relu(self.fc_out(output.squeeze(0)))
        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class rnn_model(nn.Module):
    def __init__(self, encoder, decoder, device, input_dim=4, hid_dim=4, output_dim=2):
        super(rnn_model, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fc_in = nn.Linear(self.input_dim, self.input_dim)
        self.linear_list = nn.ModuleList(
            [nn.Linear(self.input_dim, self.input_dim) for i in range(30)])
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, state, teacher_forcing_ratio=0.5):
        if state == 'eva':
            teacher_forcing_ratio = 0
        elif state == 'train':
            teacher_forcing_ratio = 0.5
        # src = [src len, batch size, feature_dim]
        # trg = [trg len, batch size, feature_dim]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        # print("target : {}, {}".format(type(trg), trg.shape))
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len - 1, batch_size,
                              trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_outputs, hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0]
        input = F.relu(self.fc_in(input))
        # print(input.shape)
        for t in range(trg_len - 1):
            output, hidden, cell = self.decoder(
                input, hidden, cell, enc_outputs)
            outputs[t] = output.squeeze()
            teacher_force = random.random() < teacher_forcing_ratio
            input = F.relu(self.linear_list[t](
                trg[t+1])) if teacher_force else output
            # input = output
        outputs = outputs.double()
        # (frame_n - frame_gap) * batch_size * 2
        predictions = self.fc_out(outputs)
        # return predictions.permute(1,0,2).reshape(-1, self.output_dim)
        # print(predictions.shape)
        return predictions


class dynamic_model(nn.Module):
    def __init__(self, args, device, dropout=0.3, node_dim=10):
        super(dynamic_model, self).__init__()
        self.rnn_layer_num = 2
        self.rnn_hid_dim = 8
        self.frame_n = args.frame_n
        self.num_nodes = args.node_num
        self.dropout = dropout
        self.node_dim = node_dim
        self.device = device
        self.gcn_type = args.gcn_type
        self.num_graph = args.num_graph
        self.frame_gap = args.frame_gap
        if args.normalization:
            self.layer_norm_list = nn.ModuleList(
                [nn.LayerNorm([self.num_nodes, 4]) for i in range(self.frame_gap)])
        self.gc_list = torch.nn.ModuleList([similarity_graph_constructor(
            self.num_nodes, 8, 4, self.device) for i in range(self.frame_gap)])
        self.gcn1_list = torch.nn.ModuleList(
            [DenseGCNConv(4, 16) for i in range(self.frame_gap)])
        self.gcn2_list = torch.nn.ModuleList(
            [DenseGCNConv(16, 4) for i in range(self.frame_gap)])
        self.rnn_encoder = rnn_encoder().double()
        self.rnn_decoder = rnn_decoder().double()
        self.rnn_model = rnn_model(
            self.rnn_encoder, self.rnn_decoder, self.device).double()
        # self.no_gcn = True
        self.proj1 = torch.nn.ModuleList(
            [nn.Linear(4, 16) for i in range(self.frame_gap)])
        self.proj2 = torch.nn.ModuleList(
            [nn.Linear(16, 4) for i in range(self.frame_gap)])

    def forward(self, X, state):  # X: batchsize * seq_len * node_num * dim
        encoder_frames = []
        gt_frames = []
        X = torch.tensor(X, dtype=torch.double).to(self.device)
        psi = torch.atan2(X[:, 0, 0, 3], X[:, 0, 0, 2])
        R = proj_mat(psi, self.device,
                     X[:, 0, 0, 0].clone(), X[:, 0, 0, 1].clone())
        pos_transed = torch.matmul(R.unsqueeze_(1).repeat(1, 40, 1, 1), torch.cat([X[:, :, 0, 0:2].unsqueeze(
            2), torch.ones([X.shape[0], X.shape[1], 1, 1]).to(self.device).double()], dim=3).transpose(2, 3))
        pos_transed = pos_transed.squeeze()
        vel = X[:, :, 0, 2:].clone()
        # print(pos_transed.shape)
        # print(psi.shape)
        psi = psi.unsqueeze(-1).repeat([1, 40])
        # print(psi.shape)
        # print(vel.shape)
        vel_transed = vel.clone()
        vel_transed[:, :, 0] = torch.mul(vel[:, :, 0], torch.cos(
            psi)) + torch.mul(vel[:, :, 1], torch.sin(psi))
        vel_transed[:, :, 1] = torch.mul(-vel[:, :, 0], torch.sin(psi)) + \
            torch.mul(vel[:, :, 1], torch.cos(psi))
        XX = torch.cat([pos_transed[:, :, 0:2], vel_transed], dim=2)
        # print(XX.shape)
        # print("pos",pos_transed.shape)
        # print("R",R.repeat(1,40,1,1).shape,R.repeat(1,40,1,1)[0][0])

        for i in range(self.frame_n):
            if i < self.frame_gap:
                encoder_frames.append(XX[:, i, :].clone())
            else:
                gt_frames.append(XX[:, i, :].clone())
        gt_frames = torch.stack(gt_frames).type(torch.double).to(self.device)
        encoder_frames = torch.stack(encoder_frames).type(
            torch.double).to(self.device)
        # print("encoder_frames: ", encoder_frames.shape)
        encodered_agent_features = []  # seq_len * batch_size * node_dim
        gt_agent_features = []
        # if self.no_gcn:
        #     for i in range(self.frame_n):
        #         if i < self.frame_gap:
        #             agent_feature = F.relu(self.proj[i](encoder_frames[i,:,0,:]))
        #             encodered_agent_features.append(agent_feature.clone())
        #             if i == self.frame_gap - 1:
        #                 gt_agent_features.append(agent_f  eature.clone())
        #         else:
        #             j = round(torch.rand(1).item()*(self.frame_gap - 1))
        #             gt_feature = F.relu(self.proj[j](gt_frames[i-self.frame_gap,:,0,:]))
        #             gt_agent_features.append(agent_feature.clone())
        for i in range(self.frame_n):
            if i < self.frame_gap:
                # encoder_adj = F.relu(self.gc_list[i](encoder_frames[i]))
                # x1 = F.relu(self.gcn1_list[i](encoder_frames[i],encoder_adj))
                # x2 = self.gcn2_list[i](x1, encoder_adj)
                # x2 = self.layer_norm_list[i](x2)
                # x2 = F.relu(x2)
                # agent_feature = F.relu(self.proj2[i](self.proj1[i](encoder_frames[i,:,0,:])))
                agent_feature = encoder_frames[i, :, :]
                encodered_agent_features.append(agent_feature.clone())
                if i == self.frame_gap - 1:
                    gt_agent_features.append(agent_feature.clone())
            else:
                j = round(torch.rand(1).item()*(self.frame_gap - 1))
                # gt_adj = F.relu(self.gc_list[j](gt_frames[i-self.frame_gap]))
                # x1 = F.relu(self.gcn1_list[j](gt_frames[i-self.frame_gap],gt_adj))
                # x2 = self.gcn2_list[j](x1, gt_adj)
                # x2 = self.layer_norm_list[j](x2)
                # x2 = F.relu(x2)
                # gt_feature = F.relu(self.proj2[j](self.proj1[j](gt_frames[i-self.frame_gap,:,0,:])))
                gt_feature = gt_frames[i-self.frame_gap, :, :]
                gt_agent_features.append(gt_feature.clone())
        encodered_agent_features = torch.stack(
            encodered_agent_features).to(self.device)
        gt_agent_features = torch.stack(gt_agent_features).to(
            self.device)  # seq_len * batch_size * node_dim
        # print(encodered_agent_features.shape)
        # print(gt_agent_features.shape)
        predictions = self.rnn_model(
            encodered_agent_features, gt_agent_features, state)
        return R, predictions


class similarity_graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, feature_dim=4, static_feat=1):
        super(similarity_graph_constructor, self).__init__()
        # print("icra model")
        self.nnodes = nnodes
        self.feature_dim = feature_dim
        self.device = device
        self.NG = 2
        self.fc_theta = nn.Linear(self.feature_dim, 16)
        self.fc_phi = nn.Linear(self.feature_dim, 16)
        self.only_dis = False

    def cal_dis_metrix(self, X):
        dis_metrix = np.zeros((X.shape[0], self.nnodes, self.nnodes))
        dis_metrix = torch.tensor(dis_metrix).to(self.device)
        pos = X[:, :, 1:3]*1000
        dis_metrix = self.calc_pairwise_distance(pos, pos)
        return dis_metrix, (dis_metrix > 30)

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

        return torch.sqrt(dist)

    def forward(self, X):
        dis_metrix, dis_mask = self.cal_dis_metrix(X)
        if not self.only_dis:
            X = torch.tensor(X, dtype=torch.double, device=self.device)
            feature_theta = self.fc_theta(X)
            feature_phi = self.fc_phi(X)
            relation_graph = torch.matmul(
                feature_theta, feature_phi.transpose(1, 2)) / 4
            relation_graph[dis_mask] = -float('inf')
            relation_graph = torch.softmax(relation_graph, dim=2)
        else:
            relation_graph = dis_metrix
            relation_graph[dis_mask] = -float('inf')
            relation_graph = (30 / (relation_graph + 0.000001))
            relation_graph = torch.softmax(relation_graph, dim=2)
        # assert relation_graph.shape[0] == X.shape[0], "batch size is wrong"
        # assert relation_graph.shape[1] == self.nnodes, "node number is wrong"
        # assert relation_graph.shape[2] == self.nnodes, "not an adjacency matrix"
        return relation_graph


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, feature_dim=10, static_feat=1):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        self.static_feat = static_feat
        if static_feat is not None:
            xd = feature_dim
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim, dim)
            self.lin2 = nn.Linear(dim, dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha

    def forward(self, X):  # feature should be input from here, so that is the real problem
        if self.static_feat is None:
            pass
            # nodevec1 = self.emb1(idx)
            # nodevec2 = self.emb2(idx)
        else:
            nodevec1 = X
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.matmul(nodevec1, nodevec2.transpose(2, 1)) - \
            torch.matmul(nodevec2, nodevec1.transpose(2, 1))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(adj.shape[0], self.nnodes,
                           self.nnodes).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = adj.topk(self.k, 2)
        mask.scatter_(-1, t1, s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, X):
        if self.static_feat is None:
            pass
            # nodevec1 = self.emb1(idx)
            # nodevec2 = self.emb2(idx)
        else:
            nodevec1 = X
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - \
            torch.mm(nodevec2, nodevec1.transpose(1, 0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj


class Model(nn.Module):
    def __init__(self, args, device, dropout=0.3, node_dim=10, use_cuda=True, graph_con='DynamicMts'):
        super(Model, self).__init__()
        self.use_cuda = use_cuda
        self.frame_n = args.frame_n
        self.num_nodes = args.node_num
        self.dropout = dropout
        self.node_dim = node_dim
        self.device = device
        self.fusion = args.fusion
        self.gcn_type = args.gcn_type
        self.num_graph = args.num_graph
        self.max_or_fixed = 'max'
        self.multigraph = args.multigraph
        if args.normalization:
            if not self.multigraph:
                self.layer_norm = nn.ModuleList(
                    [nn.LayerNorm([self.num_nodes, 8]) for i in range(self.frame_n - 1)])
            else:
                self.gcn_layer_norm_list1 = torch.nn.ModuleList(
                    [nn.LayerNorm([self.num_nodes, 8]) for i in range(self.num_graph)])
                self.gcn_layer_norm_list2 = torch.nn.ModuleList(
                    [nn.LayerNorm([self.num_nodes, 8]) for i in range(self.num_graph)])
        if self.fusion:
            if graph_con == 'DynamicMts':
                self.gc1 = graph_constructor(
                    self.num_nodes, 8, 10, self.device)
                self.gc2 = graph_constructor(
                    self.num_nodes, 8, 10, self.device)
            elif graph_con == 'icra':
                if not self.multigraph:
                    self.gc1 = similarity_graph_constructor(
                        self.num_nodes, 8, 10, self.device)
                    self.gc2 = similarity_graph_constructor(
                        self.num_nodes, 8, 10, self.device)
                else:
                    self.gc1_list = torch.nn.ModuleList([similarity_graph_constructor(
                        self.num_nodes, 8, 10, self.device) for i in range(self.num_graph)])
                    self.gc2_list = torch.nn.ModuleList([similarity_graph_constructor(
                        self.num_nodes, 8, 10, self.device) for i in range(self.num_graph)])

            if args.gcn_type == 'gcn':
                if not self.multigraph:
                    self.gcn1 = DenseGCNConv(10, 32)
                    self.gcn2 = DenseGCNConv(32, 16)
                    self.gcn3 = DenseGCNConv(16, 8)
                else:
                    self.gcn1_list = torch.nn.ModuleList(
                        [DenseGCNConv(10, 16) for i in range(self.num_graph)])
                    self.gcn2_list = torch.nn.ModuleList(
                        [DenseGCNConv(16, 8) for i in range(self.num_graph)])

                    # self.gcn21_list = torch.nn.ModuleList([DenseGCNConv(10,16) for i in range(self.num_graph)])
                    # self.gcn22_list = torch.nn.ModuleList([DenseGCNConv(16,8) for i in range(self.num_graph)])

            elif args.gcn_type == 'gin':
                ginnn = nn.Sequential(
                    nn.Linear(10, 16),
                    # nn.ReLU(True),
                    # nn.Linear(args.hid1, args.hid2),
                    nn.ReLU(True),
                    nn.Linear(16, 8),
                    nn.ReLU(True)
                )
                self.gin = DeGINConv(ginnn)
            self.lin1 = nn.Linear(16, 8)
            self.lin2 = nn.Linear(8, 2)
        else:
            if graph_con == 'DynamicMts':
                self.gc = graph_constructor(self.num_nodes, 8, 10, self.device)
            elif graph_con == 'icra':
                self.gc_list = torch.nn.ModuleList([similarity_graph_constructor(
                    self.num_nodes, 8, 10, self.device) for i in range(self.num_graph)])
            self.gcn_list = torch.nn.ModuleList(
                [DenseGCNConv(10, 10) for i in range(self.num_graph)])
            self.gcn_layer_norm_list = torch.nn.ModuleList([nn.LayerNorm(
                [self.num_nodes*(self.frame_n-1), self.node_dim]) for i in range(self.num_graph)])
            self.lin1 = nn.Linear(10, 2)

        # self.criterion = nn.MSELoss(size_average = False).cuda()
    def forward(self, x):
        if self.fusion:
            x_graph_1 = x[:, 0, :, :]
            x_graph_2 = x[:, 1, :, :]
            x_graph_1 = torch.squeeze(x_graph_1)
            x_graph_2 = torch.squeeze(x_graph_2)
            x_graph_1 = torch.tensor(
                x_graph_1, dtype=torch.float32).to(self.device)
            x_graph_2 = torch.tensor(
                x_graph_2, dtype=torch.float32).to(self.device)
            if not self.multigraph:
                adp1 = F.relu(self.gc1(x_graph_1))
                adp2 = F.relu(self.gc2(x_graph_2))
                if self.gcn_type == 'gcn':
                    x11 = self.gcn1(x_graph_1, adp1)
                    # x11 = self.layer_norm[0](x11)
                    x11 = F.relu(x11)
                    x12 = self.gcn2(x11, adp1)
                    # x12 = self.layer_norm[0](x12)
                    x12 = F.relu(x12)
                    x13 = self.gcn3(x12, adp1)
                    x13 = self.layer_norm[0](x13)
                    x13 = F.relu(x13)

                    x21 = self.gcn1(x_graph_2, adp2)
                    # x21 = self.layer_norm[1](x21)
                    x21 = F.relu(x21)
                    x22 = self.gcn2(x21, adp2)
                    # x21 = self.layer_norm[1](x22)
                    x21 = F.relu(x22)
                    x23 = self.gcn3(x22, adp2)
                    x23 = self.layer_norm[1](x23)
                    x23 = F.relu(x23)

                elif self.gcn_type == 'gin':
                    x13 = F.relu(self.gin(x_graph_1, adp1))
                    x13 = self.layer_norm[0](x13)
                    x23 = F.relu(self.gin(x_graph_2, adp2))
                    x23 = self.layer_norm[0](x23)
                # x_final = torch.cat((x23,x13),2) #B*node_num*2feature_dim
                # x_final = x_final.reshape(x_final.shape[0],x_final.shape[1]*x_final.shape[2])
                x_agent_feature = torch.cat((x13[:, 0, :], x23[:, 0, :]), 1)
                assert x_agent_feature.shape[0] == x23.shape[0], "batch size error"
                assert x_agent_feature.shape[1] == 16, "input channel error"

                h1 = self.lin1(x_agent_feature)
                h2 = self.lin2(h1)
                return h2
            else:
                graph1_feature_list = []
                for i in range(self.num_graph):
                    A1 = self.gc1_list[i](x_graph_1)
                    x11 = F.relu(self.gcn1_list[i](x_graph_1, A1))
                    x12 = self.gcn2_list[i](x11, A1)
                    x12 = self.gcn_layer_norm_list1[i](x12)
                    x12 = F.relu(x12)
                    graph1_feature_list.append(x12)
                graph_feature1 = torch.sum(
                    torch.stack(graph1_feature_list), dim=0)
                graph2_feature_list = []
                for i in range(self.num_graph):
                    A2 = self.gc2_list[i](x_graph_2)
                    x21 = F.relu(self.gcn1_list[i](x_graph_2, A2))
                    x22 = self.gcn2_list[i](x21, A2)
                    x22 = self.gcn_layer_norm_list2[i](x22)
                    x22 = F.relu(x22)
                    graph2_feature_list.append(x22)
                graph_feature2 = torch.sum(
                    torch.stack(graph2_feature_list), dim=0)
                x_agent_feature = torch.cat(
                    (graph_feature1[:, 0, :], graph_feature2[:, 0, :]), 1)
                assert x_agent_feature.shape[0] == x_graph_1.shape[0], "batch size error"
                assert x_agent_feature.shape[1] == 16, "input channel error"

                h1 = self.lin1(x_agent_feature)
                h2 = self.lin2(h1)
                return h2

        else:  # the same as cvpr paper, group activity recognition
            batch_size = x.shape[0]
            # to [B, num_car*frame-1, features]
            x_graph = x.reshape([batch_size, -1, self.node_dim]).float()
            # x_graph = torch.sum(x, dim = 1) # lost one dimension, to [B, num_car, features]
            # print(x_graph.shape)
            # x_graph = torch.tensor(x_graph, dtype=torch.float32).to(self.device)
            graph_feature_list = []
            for i in range(self.num_graph):
                A = self.gc_list[i](x_graph)
                x1 = self.gcn_list[i](x_graph, A)
                x1 = self.gcn_layer_norm_list[i](x1)
                x1 = F.relu(x1)
                graph_feature_list.append(x1)
            # 3 parallel graph
            graph_features = torch.sum(torch.stack(graph_feature_list), dim=0)
            graph_features = graph_features.reshape(
                [batch_size, self.frame_n-1, self.num_nodes, self.node_dim])
            if self.max_or_fixed == 'max':
                graph_features_pooled, _ = torch.max(graph_features, dim=2)
            elif self.max_or_fixed == 'fixed':
                graph_features_pooled = graph_features[:, :, 0, :]
            graph_features_pooled_flat = graph_features_pooled.reshape(
                [-1, self.node_dim])
            # x_agent_feature = x1[:,0,:] ###
            est_pos = self.lin1(graph_features_pooled_flat)
            est_pos = est_pos.reshape([batch_size, -1, 2])
            est_pos = torch.mean(est_pos, dim=1).reshape([batch_size, -1])
            return est_pos
