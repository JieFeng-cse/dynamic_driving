import numpy as np
import torch
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible

class TemData(Data.Dataset):
    "just for now"
    def __init__(self, data_path, frame_n, device):
        self.data_set = torch.tensor(np.load(data_path))
        self.frame_n = frame_n
        self.device = device
        # print(self.data_set.shape)
        assert len(self.data_set.shape) == 4, "something wrong with the feature map"
    def __getitem__(self, index):
        X = torch.unsqueeze(self.data_set[index][0], 0)
        for i in range(1, self.frame_n - 1):
            tmp = torch.unsqueeze(self.data_set[index][i], 0)
            X = torch.cat((X,tmp),0)

        for i in range(self.data_set[index][self.frame_n-1].shape[0]):
            if self.data_set[index][self.frame_n-1][i][-1] == 1:
                Lable = torch.stack([self.data_set[index][self.frame_n-1][i][1],self.data_set[index][self.frame_n-1][i][2]])
        return (X, Lable)
    def __len__(self):
        return self.data_set.shape[0]

if __name__ == "__main__":
    npy_path = '/home/jonathon/Documents/new_project/interaction-dataset-master/data/track_0_feature.npy'
    frame_n = 3
    device = torch.device('cuda:0')
    feature_map = TemData(npy_path,frame_n,device)
    dataset_loader = Data.DataLoader(dataset=feature_map,
                                                    batch_size=32,
                                                    shuffle=True)
    for Xs, Lables in dataset_loader:
        print(Lables)

