import torch.utils.data as data
import torch
import h5py

class Read_dataset_h5(data.Dataset):
    def __init__(self, file_path):
        super(Read_dataset_h5, self).__init__()
        hf = h5py.File(file_path)
        self.input_x2 = hf.get('input_x2')
        self.input_x3 = hf.get('input_x3')
        self.input_x4 = hf.get('input_x4')
        self.label_x2 = hf.get('label_x2')
        self.label_x3 = hf.get('label_x3')
        self.label_x4 = hf.get('label_x4')

    def __getitem__(self, index):
        i_x2 = torch.from_numpy(self.input_x2[index,:,:,:]).float()
        i_x3 = torch.from_numpy(self.input_x3[index,:,:,:]).float()
        i_x4 = torch.from_numpy(self.input_x4[index,:,:,:]).float()
        l_x2 = torch.from_numpy(self.label_x2[index,:,:,:]).float()
        l_x3 = torch.from_numpy(self.label_x3[index,:,:,:]).float()
        l_x4 = torch.from_numpy(self.label_x4[index,:,:,:]).float()
        return i_x2, i_x3, i_x4, l_x2, l_x3, l_x4

    def __len__(self):
        return self.input_x2.shape[0]
