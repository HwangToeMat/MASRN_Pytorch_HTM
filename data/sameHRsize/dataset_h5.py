import torch.utils.data as data
import torch
import h5py

class Read_dataset_h5(data.Dataset):
    def __init__(self, file_path):
        super(Read_dataset_h5, self).__init__()
        hf = h5py.File(file_path)
        self.input = hf.get('input')
        self.label_x2 = hf.get('label_x2')
        self.label_x3 = hf.get('label_x3')
        self.label_x4 = hf.get('label_x4')

    def __getitem__(self, index):
        return torch.from_numpy(self.input[index,:,:,:]).float(), torch.from_numpy(self.label[index,:,:,:]).float(), torch.from_numpy(self.label[index,:,:,:]).float(), torch.from_numpy(self.label[index,:,:,:]).float()

    def __len__(self):
        return self.input.shape[0]
