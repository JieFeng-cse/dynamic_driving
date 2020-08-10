import argparse
import math
import os
import time
import datetime
import torch
import torch.nn as nn
from model import Model, dynamic_model
from dataloader import TemData, RNNData
import numpy as np
import importlib
import sys
import Optim
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as Data
def evaluate(validation_loader,model,criterion,batch_size,device):
    model.eval()
    total_loss = 0
    n_smaples = 0
    i = 0
    with torch.no_grad():
        for Xs,Lables in validation_loader:
            Xs = Xs[:,:,:,1:5]
            Xs = Xs.to(device)
            # print(Xs.shape)
            Lables = Lables.to(device)
            output = (model(Xs)).type(torch.double)
            Lables = (Lables[:,0,:]).type(torch.double)
            output = output.reshape(-1,2)
            Lables = Lables.reshape(-1,2)
            if i%200 == 0:
                # print('prediction: ' + str(output[0]) + ' label: ' + str(Lables[0]))
                print('ground truth:',Lables[0], 'prediction:', output[0])
            val_loss = criterion(output, Lables)
            val_loss.type(torch.double)

            total_loss += val_loss.item()
            n_smaples += output.shape[0]
            i += 1
            torch.cuda.empty_cache()
    return total_loss / n_smaples

def adjust_lr(optimizer,new_lr):
    print("Changing learning rate to ",new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
def train(dataset_loader,model,criterion,optim, batch_size, device):
    model.train()
    total_loss = 0
    n_smaples = 0
    for Xs, Lables in dataset_loader:
        # model.zero_grad()
        Xs = Xs[:,:,:,1:5]
        Xs = torch.tensor(Xs, dtype=torch.double).to(device)
        Lables = torch.tensor(Lables, dtype=torch.double).to(device)
        Lables = Lables[:,0,:]
        output = (model(Xs)).type(torch.double).to(device)
        Lables = (Lables).type(torch.double).to(device)
        output = output.reshape(-1,2)
        Lables = Lables.reshape(-1,2)
        train_loss = criterion(output, Lables)
        train_loss.type(torch.double)
        optim.optimizer.zero_grad()
        train_loss.backward()

        total_loss += train_loss.item()
        n_smaples += output.shape[0]
        grad_norm = optim.step()
        # torch.cuda.empty_cache()
    return total_loss / n_smaples

parser = argparse.ArgumentParser(description='PyTorch traj forecasting')
parser.add_argument('--epochs', type=int, default=3000,help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',help='batch size')
parser.add_argument('--val_batch_size', type=int, default=512, metavar='N',help='batch size')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--mode', type=str, default='train',help='train or test')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',help='report interval')

parser.add_argument('--save', type=str,  default='/home/jonathon/Documents/new_project/interaction-dataset-master/model_logs/',help='path to save the final model')
parser.add_argument('--cuda', type=str, default=True) 
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--gc', type=str, default='icra',help='type of graph constructor') #'DynamicMts'
parser.add_argument('--gcn_type', type=str, default='gcn',help='type of graph convolution')
parser.add_argument('--fusion', type=bool, default=True,help='fuse information from different frames before or after gc')
parser.add_argument('--multigraph', type=bool, default=True,help='multipul graphs')
parser.add_argument('--shuffle', type=bool, default=True,help='shuffle')
parser.add_argument('--num_graph',type=int,default=3,help='num_graph')

parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--clip',type=int,default=5,help='clip')
parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--frame_n', type=int, default=40,help='every case have n frames')
parser.add_argument('--frame_gap', type=int, default=10,help='frame gap between two agents')
parser.add_argument('--normalization', type=bool, default=True, help='whether to do batch normalization')
parser.add_argument('--RNN', type=bool, default=True, help='RNN model')

parser.add_argument('--device',type=str,default='cuda:0',help='')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--node_dim',type=int,default=10,help='dim of nodes')
parser.add_argument('--node_num',type=int,default=13,help='num of nodes')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')
args = parser.parse_args()

args.fusion = True
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
if not args.RNN:
    npy_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/track_0_feature.npy'
    val_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/track_1_feature.npy'

    feature_map = TemData(npy_path,args.frame_n,args.device)
    val_map = TemData(val_path,args.frame_n,args.device)
else:
    npy_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_CHN_Merging_ZS/40framespersegtracks_001.npy'
    val_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_CHN_Merging_ZS/40framespersegtracks_000.npy'
    feature_map = RNNData(args.frame_n, args.frame_gap,npy_path,args.device)
    val_map = RNNData(args.frame_n, args.frame_gap,val_path,args.device)

if args.gpu != None:
    device = torch.device(args.device)
else:
    device = torch.device('cpu')


if not args.RNN:
    model = Model(args,device,graph_con=args.gc)
    model.double()
else:
    model = dynamic_model(args, device)
    model.double()
if args.cuda:
    model.cuda()

criterion = nn.MSELoss(size_average = False).cuda()
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip,lr_decay=args.weight_decay
)
dataset_loader = Data.DataLoader(dataset=feature_map,
                                                    batch_size=args.batch_size,
                                                    shuffle=args.shuffle, num_workers= 8)
val_loader = Data.DataLoader(dataset=val_map,
                                            batch_size=args.val_batch_size,
                                            shuffle=args.shuffle, num_workers= 8)
ttime = str(datetime.datetime.now()).replace(' ','-')
save_model = args.save+ttime
best_val = 1000
new_lr = args.lr
try:
    if args.mode == 'train':
        print('begin training')
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            if epoch %800 == 0:
                new_lr = new_lr / 2
                adjust_lr(optim.optimizer, new_lr)
            train_loss = train(dataset_loader,model,criterion,optim,args.batch_size,device)
            val_loss = evaluate(val_loader,model,criterion,args.batch_size,device)
            print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.9f} | val_loss {:5.9f} |'.format(epoch,(time.time()-epoch_start_time),math.sqrt(train_loss*1000000),math.sqrt(val_loss*1000000)))
            if val_loss < best_val:
                p = os.path.join(args.save,args.gc +'best_model.pt')
                torch.save(model, p)
                best_val = val_loss
                print(best_val)
    elif args.mode == 'test':
        print('begin testing')
        p = os.path.join(args.save,args.gc +'best_model.pt')
        test_model = torch.load(p)
        epoch_start_time = time.time()
        test_loss = evaluate(val_loader,test_model,criterion,args.batch_size)
        print('| time: {:5.2f}s | test_loss {:5.9f} |'.format((time.time()-epoch_start_time),math.sqrt(test_loss*1000000)))

except KeyboardInterrupt:
    print('-' * 89)
    if args.mode == 'train':
        print('Exiting from training early')
    elif args.mode == 'test':
        print('Exiting from testing early')
