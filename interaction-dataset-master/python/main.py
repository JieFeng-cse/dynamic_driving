import argparse
import math
import os
import time
import datetime
import torch
import torch.nn as nn
from model import Model, dynamic_model, proj_mat, proj_mat_inverse
from dataloader import TemData, RNNData
from torch.autograd import grad
import numpy as np
import importlib
import sys
import Optim
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from extract_osm import *
from sub_graph import *
from dataloader_osm import vectors_car_road
from fourier_model import Model_COS

import torch.utils.data as Data
def evaluate(validation_loader,model,criterion,batch_size,device, epoch_id):
    model.eval()
    total_loss = 0
    n_samples = 0
    i = 0
    with torch.no_grad():
        for Xs,Labels in validation_loader:
            Xs = Xs[:,:,:,1:5]
            Xs = Xs.to(device)
            # print(Xs.shape)
            Labels = Labels.to(device)
            Labels = (Labels[:,:,:]).type(torch.double)
            label_tmp = Labels.clone() #batch_size*seq_len*4
            
            R, output = model(Xs)
            R = R.type(torch.double).to(device)
            output = output.type(torch.double).to(device)
            out_tmp = output.permute(1,0,2).clone()

            Labels_transed = torch.matmul(R.repeat(1,30,1,1),
                torch.cat([Labels[:,:,0:2].unsqueeze(2),torch.ones([Labels.shape[0],Labels.shape[1],1,1]).to(device).double()],dim=3).transpose(2,3))
            Labels_transed = Labels_transed.squeeze()
            Labels = Labels_transed[:,:,0:2]
            
            Labels = Labels.permute(1,0,2).contiguous()
            # print(Labels.shape)
            output = output.reshape(-1,2)
            Labels = Labels.reshape(-1,2)
            if i%200 == 0:
                print('ground truth:',Labels[0], 'prediction:', output[0])
                # L = np.mean((Labels.cpu().numpy() - output.cpu().numpy())**2)
                # print(L, criterion(output, Labels))
                R_inversed = proj_mat_inverse(R.squeeze(),device)
                traj_world = torch.matmul(R_inversed.unsqueeze(1).repeat(1,30,1,1),torch.cat([out_tmp[:,:,0:2].unsqueeze(2),torch.ones([out_tmp.shape[0],out_tmp.shape[1],1,1]).to(device).double()],dim=3).transpose(2,3))
                traj_world = traj_world.squeeze()[:,:,0:2]
                main_drawer('/home/jonathon/Documents/new_project/interaction-dataset-master/maps/DR_CHN_Merging_ZS.osm',label_tmp[0].cpu().detach().numpy(),traj_world[0].cpu().detach().numpy(),epoch_id)


            val_loss = criterion(output, Labels)
            # val_loss = ((output - Labels)**2).mean()
            val_loss.type(torch.double)

            total_loss += val_loss.item()
            n_samples += output.shape[0]/30
            i += 1
            torch.cuda.empty_cache()
    return total_loss / i

def adjust_lr(optimizer,new_lr):
    print("Changing learning rate to ",new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
def train(dataset_loader,model,criterion,optim, batch_size, device):
    model.train()
    total_loss = 0
    n_samples = 0
    i = 0
    for Xs, Labels in dataset_loader:
        # model.zero_grad()
        Xs = Xs[:,:,:,1:5]
        Xs = torch.tensor(Xs, dtype=torch.double).to(device).requires_grad_(True)

        Labels = torch.tensor(Labels, dtype=torch.double).to(device).requires_grad_(True)
        R, output = model(Xs)
        R = R.type(torch.double).to(device)
        output = output.type(torch.double).to(device)
        output = output.reshape(-1,2).contiguous()
        Labels_transed = torch.matmul(R.repeat(1,30,1,1),
            torch.cat([Labels[:,:,0:2].unsqueeze(2),torch.ones([Labels.shape[0],Labels.shape[1],1,1]).to(device).double()],dim=3).transpose(2,3))
        # print(Labels_transed.shape)
        Labels_transed = Labels_transed.squeeze()
        Labels = Labels_transed[:,:,0:2]
        
        Labels = Labels.permute(1,0,2).contiguous()
        # print(Labels.shape)
        Labels = (Labels).type(torch.double).to(device)
        Labels = Labels.reshape(-1,2).contiguous()

        train_loss = criterion(output, Labels)
        train_loss.type(torch.double)
        optim.optimizer.zero_grad()
        train_loss.backward()
        i += 1
        total_loss += train_loss.item()
        n_samples += output.shape[0]/30
        grad_norm = optim.step()
        # torch.cuda.empty_cache()
    return total_loss / i

def train_fourier(dataset_loader,model,criterion,optim, batch_size, device, epoch_id):
    model.train()
    total_loss_xy = 0
    total_loss_vxy = 0
    n_samples = 0
    i = 0
    sequence_tracer = True
    for Xs, Labels in dataset_loader:
        Xs = torch.tensor(Xs, dtype=torch.double).to(device).requires_grad_(True)
        Labels = torch.tensor(Labels, dtype=torch.double).to(device).requires_grad_(True)
        t = torch.arange(start = 0, end = 40, step = 1).unsqueeze(0).repeat(Labels.shape[0],1)
        t = t.double().to(device)
        t.requires_grad = True
        psi = Labels[:,0,4].double().to(device)
        v0 = torch.mul(Labels[:,0,2],torch.cos(psi)) + torch.mul(Labels[:,0,3],torch.sin(psi))
        # output = model(Xs,t,v0)
        
        R = proj_mat(psi,device,Labels[:,0,0].clone(),Labels[:,0,1].clone())
        R_inversed = proj_mat_inverse(R,device)
        pos_transed = torch.matmul(R.unsqueeze(1).repeat(1,40,1,1),torch.cat([Labels[:,:,0:2].unsqueeze(2),torch.ones([Labels.shape[0],Labels.shape[1],1,1]).to(device).double()],dim=3).transpose(2,3))
        pos_transed = pos_transed.squeeze()[:,:,0:2]
        
        vel = Labels[:,:,2:4]
        psi = psi.unsqueeze(-1).repeat([1,40])

        vel_transed = vel.clone()
        vel_transed[:,:,0] = torch.mul(vel[:,:,0],torch.cos(psi)) + torch.mul(vel[:,:,1],torch.sin(psi))
        vel_transed[:,:,1] = torch.mul(-vel[:,:,0],torch.sin(psi)) + torch.mul(vel[:,:,1],torch.cos(psi))

        #just for experiment
        X = torch.cat([pos_transed[:,0:10],vel_transed[:,0:10]],dim = 2).permute(1,0,2).contiguous()
        XX = [Xs,X]
        output = model(XX,t,v0)  #output: batchsize*seq_len*2
        traj_world = torch.matmul(R_inversed.unsqueeze(1).repeat(1,40,1,1),torch.cat([(output).unsqueeze(2),torch.ones([output.shape[0],output.shape[1],1,1]).to(device).double()],dim=3).transpose(2,3))
        traj_world = traj_world.squeeze()[:,:,0:2]
        # print(traj_world.shape)

        Label_world = torch.matmul(R_inversed.unsqueeze(1).repeat(1,40,1,1),torch.cat([(pos_transed).unsqueeze(2),torch.ones([pos_transed.shape[0],pos_transed.shape[1],1,1]).to(device).double()],dim=3).transpose(2,3))
        Label_world = Label_world.squeeze()[:,:,0:2]
        
        if i%100 == 0 and i !=0:

            print("prediction: ",output[0][0])
            print("pos: ",pos_transed[0][0])
            main_drawer('/home/jonathon/Documents/new_project/interaction-dataset-master/maps/DR_CHN_Merging_ZS.osm',Label_world[0].cpu().detach().numpy(),traj_world[0].cpu().detach().numpy(),epoch_id)
        # print(X.shape)
        #just for experiment
        # print(vel_transed.shape)
        vx = grad(output[:,:,0].sum(), t, create_graph=True)[0].unsqueeze(-1)
        vy = grad(output[:,:,1].sum(), t, create_graph=True)[0].unsqueeze(-1)
        output_vxy = (15.0)*torch.cat([vx, vy], dim=2)
        
        output = output.reshape(-1,2).contiguous()
        output_vxy =output_vxy.reshape(-1,2).contiguous()
        pos_transed = pos_transed.reshape(-1,2).contiguous()
        vel_transed = vel_transed.reshape(-1,2).contiguous()


        loss_xy = criterion(output,pos_transed)
        loss_vxy = criterion(output_vxy,vel_transed)

        loss = loss_xy + loss_vxy*0.5
        loss.type(torch.double)
        optim.optimizer.zero_grad()
        loss.backward()
        i += 1
        total_loss_xy += loss_xy.item()  #dis
        total_loss_vxy += loss_vxy.item()
        n_samples += output.shape[0]/40
        grad_norm = optim.step()
        # torch.cuda.empty_cache()
    # print('projbda: ', vel_transed[0, 0, :])
    # print('p: ', output[0, :], pos_transed[0, :])
    # print('v: ', output_vxy[0, :], vel_transed[0, :])
    return total_loss_xy / i, total_loss_vxy/i

        

parser = argparse.ArgumentParser(description='PyTorch traj forecasting')
parser.add_argument('--epochs', type=int, default=200,help='upper epoch limit')
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
parser.add_argument('--model', type=str, default='rnn', help='RNN model')

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
if args.model == 'icra':
    npy_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/track_0_feature.npy'
    val_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/track_1_feature.npy'

    feature_map = TemData(npy_path,args.frame_n,args.device)
    val_map = TemData(val_path,args.frame_n,args.device)
elif args.model == 'rnn':
    npy_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_CHN_Merging_ZS/40framespersegtracks_001.npy'
    val_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_CHN_Merging_ZS/40framespersegtracks_000.npy'
    feature_map = RNNData(args.frame_n, args.frame_gap,npy_path,args.device)
    val_map = RNNData(args.frame_n, args.frame_gap,val_path,args.device)
elif args.model == 'fourier':
    npy_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_CHN_Merging_ZS/40framespersegtracks_001.npy'
    val_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/DR_CHN_Merging_ZS/40framespersegtracks_000.npy'

    feature_map = vectors_car_road('/home/jonathon/Documents/new_project/interaction-dataset-master/maps/DR_CHN_Merging_ZS.osm', npy_path, 10,30)
    val_map = vectors_car_road('/home/jonathon/Documents/new_project/interaction-dataset-master/maps/DR_CHN_Merging_ZS.osm', val_path, 10,30)
if args.gpu != None:
    device = torch.device(args.device)
else:
    device = torch.device('cpu')


if args.model == 'icra':
    model = Model(args,device,graph_con=args.gc)
    model.double()
elif args.model == 'rnn':
    model = dynamic_model(args, device)
    model.double()
elif args.model == 'fourier':
    model = Model_COS()
    model.double()
else:
    print('no modle named:', args.model)
    exit()
if args.cuda:
    model.cuda()

#criterion = nn.MSELoss(size_average = False).cuda()
criterion = nn.MSELoss().cuda()
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
val_loss = best_val
new_lr = args.lr
# for i in model.state_dict():
#     print(i)
# lr_scheduler = ReduceLROnPlateau(optim,  mode='min', factor=0.5, patience=20, verbose=True)

try:
    if args.mode == 'train':
        print('begin training')
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            if epoch %800 == 0:
                new_lr = new_lr / 2
                adjust_lr(optim.optimizer, new_lr)
            if args.model == 'rnn':
                train_loss = train(dataset_loader,model,criterion,optim,args.batch_size,device)
                # val_loss = 0
                val_loss = evaluate(val_loader,model,criterion,args.batch_size,device, str(epoch)+args.model)
                print('| end of epoch {:3d} | time: {:5.2f}s | xy_loss {:5.9f} | val_loss {:5.9f} |'.format(epoch,(time.time()-epoch_start_time),math.sqrt(train_loss),math.sqrt(val_loss)))
            elif args.model == 'fourier':
                xy_loss,v_loss = train_fourier(dataset_loader,model,criterion,optim,args.batch_size,device,str(epoch)+args.model)
                print('| end of epoch {:3d} | time: {:5.2f}s | xy_loss {:5.9f} | vel_loss {:5.9f} |'.format(epoch,(time.time()-epoch_start_time),math.sqrt(xy_loss),math.sqrt(v_loss)))
            # val_loss = evaluate(val_loader,model,criterion,args.batch_size,device)
            #*1000000
            if val_loss < best_val:
                p = os.path.join(args.save,args.gc +'best_model.pt')
                torch.save(model, p)
                best_val = val_loss
                print(best_val)
            # for name, param in model.named_parameters():
            #     if param.grad != None:
            #         print('Name: ', name, 'if_grad: ', param.requires_grad)
            #         print('grad_value: ', param.grad)
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
