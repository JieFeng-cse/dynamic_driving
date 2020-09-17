from extract_osm import *
from sub_graph import *
from dataloader_osm import vectors_car_road
from vector_net import VectorNetWithPredicting, VectorNetAndTargetPredicting
import numpy as np
import time
from extract_osm import main_drawer

import torch
from torch import nn

# sub_g = SubGraph(feature_length=14, layersNumber=3)
# train_data = torch.utils.data.DataLoader(vectors_car_road('./maps/DR_CHN_Merging_ZS.osm','data/40framespersegtracks_000.npy',30,10),
#                                         batch_size=32, shuffle=True, num_workers=4)
# for X, labels in train_data:
#     print(X.shape)
#     print(labels.shape)
#     y = sub_g(X)
#     print(y.shape)
#     break


def train(train_set, model, criterion, optim, device):
    model.train()
    total_loss = 0
    n_smaples = 0
    st_time = time.time()
    for X, labels, last_loc, _ in train_set:
        X = X.to(device).double()
        labels = labels.to(device)
        labels = labels.type(torch.double)
        # labels = labels[:,-1,:] ##
        last_loc = last_loc.to(device).type(torch.double)
        optim.zero_grad()
        output = model(X).type(torch.double)
        
        output = (last_loc[:, -1, :]).unsqueeze(1) + output # predict series
        # output = last_loc[:, -1, :] + output # predict target
        loss = criterion(output, labels)
        loss.backward()
        optim.step()
        with torch.no_grad():
            total_loss += torch.nn.MSELoss()(output, labels).item() * X.shape[0]
        # total_loss += loss.item() * X.shape[0]
        n_smaples += X.shape[0]
    return total_loss / n_smaples


def eval(val_set, model, criterion, device, epoch, need_draw=False):
    model.eval()
    total_loss = 0
    n_smaples = 0
    i = 0
    with torch.no_grad():
        for X, labels, last_loc, iR in val_set:
            X = X.to(device).double()
            iR = iR.to(device).double()
            labels = labels.to(device)
            labels = labels.type(torch.double)
            # labels = labels[:,-1,:] ##
            last_loc = last_loc.to(device).type(torch.double)
            output = model(X).type(torch.double)
            output = (last_loc[:, -1, :]).unsqueeze(1) + output # predict series
            # output = last_loc[:, -1, :] + output # predict target
            if (i+1) % 15 == 0:
                print('ground truth:', labels[0], 'prediction:', output[0])
                if True:
                    truth = Local2Global(iR[0], labels[0])
                    pred = Local2Global(iR[0], output[0])
                    main_drawer('./maps/DR_CHN_Merging_ZS.osm', truth.cpu().numpy(
                    ), pred.cpu().numpy(), './results/CHN_Merging_epoch%d.png' % (epoch))
            # val_loss = criterion(output, labels)
            total_loss += torch.nn.MSELoss()(output, labels).item() * X.shape[0]
            # total_loss += val_loss.item() * X.shape[0]
            n_smaples += X.shape[0]
            i += 1
    return total_loss / n_smaples

def Local2Global(iR, locs):
    pNum = locs.shape[0]
    
    iR = iR.unsqueeze(0)
    iR = iR.repeat(pNum, 1, 1)

    locs = torch.cat([locs, torch.ones_like(locs[:,:1])], dim = -1)
    loc_global = torch.matmul(iR, locs.unsqueeze(-1)).squeeze(-1)
    loc_global = loc_global[:,:2]

    return loc_global
def main(epochs):
    device = torch.device('cuda:0')
    model = VectorNetWithPredicting(feature_length=14, timeStampNumber=20)
    # model = VectorNetAndTargetPredicting(feature_length=14)
    model = model.to(device=device).double()

    train_set = torch.utils.data.DataLoader(
        vectors_car_road('./maps/DR_CHN_Merging_ZS.osm',
                         'data/40framespersegtracks_000.npy', 20, 20),
        batch_size=128, shuffle=True, num_workers=6)
    val_set = torch.utils.data.DataLoader(
        vectors_car_road('./maps/DR_CHN_Merging_ZS.osm',
                         'data/40framespersegtracks_001.npy', 20, 20),
        batch_size=512, shuffle=True, num_workers=6)

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.SmoothL1Loss()
    optim = torch.optim.Adam(model.parameters())
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
        print('| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.9f} | val_loss {:5.9f} |'.format(
            epoch, (time.time()-epoch_start_time), math.sqrt(train_loss), math.sqrt(val_loss)))


if __name__ == "__main__":
    # mode = 'target'
    main(epochs=100)
