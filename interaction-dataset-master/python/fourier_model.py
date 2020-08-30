import torch
import torch.nn as nn
import torch.nn.functional as F
from model import proj_mat
from vector_net import VectorNet
from model import rnn_encoder


def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a = 0, mode = 'fan_in', nonlinearity = 'tanh')
        nn.init.constant_(m.bias, 0.001)
class MLP_COS(nn.Module):
    def __init__(self, rate=1.0):
        super(MLP_COS, self).__init__()
        self.rate = rate
        self.linear1 = nn.Linear(256+2, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        # self.linear4 = nn.Linear(512, 256)
        self.linear5 = nn.Linear(256, 2)
        
        self.apply(weights_init)
        
    def forward(self, x, t, v0):
        x = torch.cat([x, t], dim=1)
        x = torch.cat([x, v0], dim=1)
        x = self.linear1(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear3(x)
        #x = F.leaky_relu(x)
        x = torch.tanh(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        
        # x = self.linear4(x)
        #x = F.leaky_relu(x)
        x = torch.cos(self.rate*x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear5(x)
        return x
    
class Model_COS(nn.Module):
    def __init__(self,rate=1.0):
        super(Model_COS, self).__init__()
        self.vectornet = VectorNet(14)
        self.mlp = MLP_COS(rate)
        self.proj = nn.Linear(140,256)
        self.norm = nn.LayerNorm(256)
        self.rnn = rnn_encoder(hid_dim=2)
    
    def forward(self, x, t, v0):
        xx = self.vectornet(x[0]) #batch * 112
        out, hidden, cell = self.rnn(x[1])
        out = out.permute(1,0,2)
        out = out.reshape(out.shape[0],-1)
        # print(out.shape)
        x = torch.cat([xx,hidden[0],hidden[1],cell[0],cell[1],out],dim=1)
        # print(x.shape)
        x = self.proj(x)
        x = self.norm(x)
        x = F.tanh(x)
        # x = torch.zeros([t.shape[0],1]).to('cuda:0').double()
        #x = x.view(-1, 512)
        trace = []
        for i in range(t.shape[1]):
            pos = self.mlp(x, t[:,i].unsqueeze(-1).clone(), v0.unsqueeze(-1).clone())
            trace.append(pos)
        trace = torch.stack(trace).double().permute(1,0,2)
        # print(trace.shape)
        return trace