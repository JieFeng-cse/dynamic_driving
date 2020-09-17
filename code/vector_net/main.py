from extract_osm import *
from sub_graph import *
from dataloader_osm import vectors_car_road, CarTrajAndMap, collect_func
from vector_net import VectorNetWithPredicting, VectorNetAndTargetPredicting
import numpy as np
import time
from extract_osm import main_drawer
from tqdm import tqdm
import os

import torch
from torch import nn
# torch.multiprocessing.set_sharing_strategy('file_system')
# sub_g = SubGraph(feature_length=14, layersNumber=3)
# train_data = torch.utils.data.DataLoader(vectors_car_road('./maps/DR_CHN_Merging_ZS.osm','data/40framespersegtracks_000.npy',30,10),
#                                         batch_size=32, shuffle=True, num_workers=4)
# for X, labels in train_data:
#     print(X.shape)
#     print(labels.shape)
#     y = sub_g(X)
#     print(y.shape)
#     break

train_pth = '/home/jonathon/Documents/new_project/interaction-dataset-master/vec_dir/DR_USA_Intersection_EP1/40framesperseg_000.pickle'
val_pth = '/home/jonathon/Documents/new_project/interaction-dataset-master/vec_dir/DR_USA_Intersection_EP1/40framesperseg_001.pickle'
map_pth = '/home/jonathon/Documents/new_project/interaction-dataset-master/maps/DR_USA_Intersection_EP1.osm'
def train(train_set, model, criterion, optim, device):
    model.train()
    total_loss = 0
    n_smaples = 0
    st_time = time.time()
    counter = 0
    for X, labels, osm, osm_interval,_ in tqdm(train_set):
        counter += 1
        # print('train processing:', counter)
        # X = X.to(device).double()
        labels = labels.to(device)
        labels = labels.type(torch.double)
        osm = osm.to(device).double()
        # labels = labels[:,-1,:] ##
        # last_loc = last_loc.to(device).type(torch.double)
        # print(last_loc[:,-1,:])
        optim.zero_grad()
        output = model(X, osm, osm_interval).type(torch.double)
        # local coordinate: None
        # output = (last_loc[:, -1, :]).unsqueeze(1) + output # predict series
        # output = last_loc[:, -1, :] + output # predict target
        loss = criterion(output, labels)
        loss.backward()
        optim.step()
        with torch.no_grad():
            output = output.reshape(-1, 2).contiguous()
            labels = labels.reshape(-1, 2).contiguous()
            total_loss += torch.norm((output - labels),2, dim=1).mean()
        # total_loss += loss.item() * X.shape[0]
        n_smaples += len(X)
    return total_loss / counter


def eval(val_set, model, criterion, device, epoch, need_draw=False):
    model.eval()
    total_loss = 0
    n_smaples = 0
    i = 0
    with torch.no_grad():
        for X, labels, osm, osm_interval, iR in tqdm(val_set):
            # X = X.to(device).double()
            iR = iR.to(device).double()
            labels = labels.to(device)
            labels = labels.type(torch.double)
            osm = osm.to(device).double()
            # labels = labels[:,-1,:] ##
            # last_loc = last_loc.to(device).type(torch.double)
            output = model(X, osm, osm_interval).type(torch.double)
            # local coordinate: None
            # output = (last_loc[:, -1, :]).unsqueeze(1) + output # predict series
            # output = last_loc[:, -1, :] + output # predict target
            if i == 0:
                if True:
                    truth = Local2Global(iR[0], labels[0])
                    pred = Local2Global(iR[0], output[0])
                    if not os.path.exists('./results/' + os.path.split(map_pth)[1].split('.')[0]):
                        os.makedirs('./results/' + os.path.split(map_pth)[1].split('.')[0])
                        print("new folder")
                    main_drawer(map_pth, truth.cpu().numpy(), \
                        pred.cpu().numpy(), './results/' + os.path.split(map_pth)[1].split('.')[0] +'/epoch%d.png' % (epoch))
            # val_loss = criterion(output, labels)
            output = output.reshape(-1, 2).contiguous()
            labels = labels.reshape(-1, 2).contiguous()
            total_loss += torch.norm((output - labels),2, dim=1).mean()
            # total_loss += val_loss.item() * X.shape[0]
            n_smaples += len(X)
            i += 1
            # print('val processing:', i)
    return total_loss / i

def Local2Global(iR, locs):
    pNum = locs.shape[0]
    
    iR = iR.unsqueeze(0)
    iR = iR.repeat(pNum, 1, 1)

    locs = torch.cat([locs, torch.ones_like(locs[:,:1])], dim = -1)
    loc_global = torch.matmul(iR, locs.unsqueeze(-1)).squeeze(-1)
    loc_global = loc_global[:,:2]

    return loc_global
def main(epochs, train_frame, pred_frame):
    device = torch.device('cuda:0')
    model = VectorNetWithPredicting(feature_length=14, timeStampNumber=pred_frame)
    # model = VectorNetAndTargetPredicting(feature_length=14)
    model = model.to(device=device).double()

    # train_set = torch.utils.data.DataLoader(
    #     vectors_car_road('./maps/DR_CHN_Merging_ZS.osm',
    #                      'data/40framespersegtracks_000.npy', train_frame, pred_frame),
    #     batch_size=128, shuffle=True, num_workers=6)
    train_set = torch.utils.data.DataLoader(
        CarTrajAndMap(map_pth, train_pth, train_frame, pred_frame),
        batch_size=16, shuffle=True, num_workers=0, collate_fn=collect_func)
    # val_set = torch.utils.data.DataLoader(
    #     vectors_car_road('./maps/DR_CHN_Merging_ZS.osm',
    #                      'data/40framespersegtracks_001.npy', train_frame, pred_frame),
    #     batch_size=512, shuffle=True, num_workers=6)
    val_set = torch.utils.data.DataLoader(
        CarTrajAndMap(map_pth, val_pth, train_frame, pred_frame),
        batch_size=512, shuffle=True, num_workers=0, collate_fn=collect_func)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=30,gamma=0.5,last_epoch=-1)
    for epoch in range(epochs):
        if epoch > 4 and epoch % 5 == 0:
            need_draw = True
        else:
            need_draw = False
        epoch_start_time = time.time()
        train_loss = train(train_set, model=model,
                           criterion=criterion, optim=optim, device=device)
        val_loss = eval(val_set, model, criterion, device,
                        epoch, need_draw=need_draw)
        # val_loss = 0.0
        scheduler.step()
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.9f} | val_loss {:5.9f} |'.format(
            epoch, (time.time()-epoch_start_time), train_loss, val_loss))


if __name__ == "__main__":
    # mode = 'target'
    main(epochs=150, train_frame=10, pred_frame=30)
