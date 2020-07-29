import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from layer import DeGINConv,DenseGCNConv
# from MTlayer import *
import pickle
import os
import scipy.sparse as sp
from scipy.sparse import linalg
from torch.autograd import Variable
class similarity_graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, feature_dim = 10, static_feat =1):
        super(similarity_graph_constructor, self).__init__()
        print("icra model")
        self.nnodes = nnodes
        self.feature_dim = feature_dim
        self.device = device
        self.NG = 2
        self.fc_theta = nn.Linear(self.feature_dim, 16) 
        self.fc_phi = nn.Linear(self.feature_dim, 16)
        self.only_dis = False
    def cal_dis_metrix(self,X):
        dis_metrix = np.zeros((X.shape[0], self.nnodes, self.nnodes))
        dis_metrix = torch.tensor(dis_metrix).to(self.device)
        pos = X[:,:,1:3]*1000
        dis_metrix = self.calc_pairwise_distance(pos,pos)
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
        B=X.shape[0]
        
        rx=X.pow(2).sum(dim=2).reshape((B,-1,1)).to(self.device)
        ry=Y.pow(2).sum(dim=2).reshape((B,-1,1)).to(self.device)
        
        dist=rx-2.0*X.matmul(Y.transpose(1,2))+ry.transpose(1,2)
        
        return torch.sqrt(dist)
    def forward(self,X):
        dis_metrix, dis_mask = self.cal_dis_metrix(X)
        if not self.only_dis:
            feature_theta = self.fc_theta(X)
            feature_phi = self.fc_phi(X)
            relation_graph = torch.matmul(feature_theta,feature_phi.transpose(1,2)) / 4
            relation_graph[dis_mask]=-float('inf')
            relation_graph = torch.softmax(relation_graph,dim=2) 
        else:
            relation_graph = dis_metrix
            relation_graph[dis_mask] = -float('inf')
            relation_graph = (30 / (relation_graph + 0.000001))
            relation_graph = torch.softmax(relation_graph,dim=2)
        assert relation_graph.shape[0] == X.shape[0], "batch size is wrong"
        assert relation_graph.shape[1] == self.nnodes, "node number is wrong"
        assert relation_graph.shape[2] == self.nnodes, "not an adjacency matrix"
        return relation_graph       




class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, feature_dim = 10, static_feat =1):
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
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha

    def forward(self, X): #feature should be input from here, so that is the real problem
        if self.static_feat is None:
            pass
            # nodevec1 = self.emb1(idx)
            # nodevec2 = self.emb2(idx)
        else:
            nodevec1 = X
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.matmul(nodevec1, nodevec2.transpose(2,1))-torch.matmul(nodevec2, nodevec1.transpose(2,1))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(adj.shape[0], self.nnodes, self.nnodes).to(self.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,2)
        mask.scatter_(-1,t1,s1.fill_(1))
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

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class Model(nn.Module):
    def __init__(self,args,device,dropout = 0.3, node_dim = 10, use_cuda = True, graph_con = 'DynamicMts'):
        super(Model,self).__init__()
        self.use_cuda = use_cuda
        self.frame_n = args.frame_n
        self.num_nodes = args.node_num
        self.dropout = dropout
        self.node_dim = node_dim
        self.device = device
        self.fusion = args.fusion
        self.gcn_type = args.gcn_type
        print(self.gcn_type)
        if args.normalization:
            self.layer_norm = nn.ModuleList([ nn.LayerNorm([self.num_nodes, 8]) for i in range(self.frame_n - 1) ])
        if self.fusion:
            if graph_con == 'DynamicMts': 
                self.gc1 = graph_constructor(self.num_nodes,8,10,self.device)
                self.gc2 = graph_constructor(self.num_nodes,8,10,self.device)
            elif graph_con == 'icra':
                self.gc1 = similarity_graph_constructor(self.num_nodes,8,10,self.device)
                self.gc2 = similarity_graph_constructor(self.num_nodes,8,10,self.device)
            if args.gcn_type == 'gcn':
                self.gcn1 = DenseGCNConv(10,32)
                self.gcn2 = DenseGCNConv(32,16)
                self.gcn3 = DenseGCNConv(16,8)
            elif args.gcn_type == 'gin':
                ginnn = nn.Sequential(
                    nn.Linear(10,16),
                    # nn.ReLU(True),
                    # nn.Linear(args.hid1, args.hid2),
                    nn.ReLU(True),
                    nn.Linear(16,8),
                    nn.ReLU(True)
                )
                self.gin = DeGINConv(ginnn)
            self.lin1 = nn.Linear(16,8)
            self.lin2 = nn.Linear(8,2)
        else:
            if graph_con == 'DynamicMts':
                self.gc = graph_constructor(self.num_nodes,8,10,self.device)
            elif graph_con == 'icra':
                self.gc = similarity_graph_constructor(self.num_nodes,8,10,self.device)
            self.gcn1 = DenseGCNConv(10,8)
            self.lin1 = nn.Linear(8,4)
            self.lin2 = nn.Linear(4,2)

        # self.criterion = nn.MSELoss(size_average = False).cuda()
    def forward(self,x):
        if self.fusion:
            x_graph_1 = x[:,0,:,:]
            x_graph_2 = x[:,1,:,:]
            x_graph_1 = torch.squeeze(x_graph_1)
            x_graph_2 = torch.squeeze(x_graph_2)
            x_graph_1 = torch.tensor(x_graph_1, dtype=torch.float32).to(self.device)
            x_graph_2 = torch.tensor(x_graph_2, dtype=torch.float32).to(self.device)
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
            x_agent_feature = torch.cat((x13[:,0,:],x23[:,0,:]),1)
            assert x_agent_feature.shape[0] == x23.shape[0], "batch size error"
            assert x_agent_feature.shape[1] == 16, "input channel error"

            h1 = self.lin1(x_agent_feature)
            h2 = self.lin2(h1)
            return h2
        else:
            x_graph = torch.sum(x, dim = 1)
            x_graph = torch.tensor(x_graph, dtype=torch.float32).to(self.device)
            A = self.gc(x_graph)

            x1 = self.gcn1(x_graph, A)

            x_agent_feature = x1[:,0,:]
            h1 = self.lin1(x_agent_feature)
            h2 = self.lin2(h1)
            return h2




