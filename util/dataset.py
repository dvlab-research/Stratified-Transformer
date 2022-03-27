import os
import h5py
import numpy as np

from torch.utils.data import Dataset


def make_dataset(split='train', data_root=None, data_list=None):
    if not os.path.isfile(data_list):
        raise (RuntimeError("Point list file do not exist: " + data_list + "\n"))
    point_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    for line in list_read:
        point_list.append(os.path.join(data_root, line.strip()))
    return point_list


class PointData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None, num_point=None, random_index=False):
        assert split in ['train', 'val', 'test']
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform
        self.num_point = num_point
        self.random_index = random_index

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_path = self.data_list[index]
        f = h5py.File(data_path, 'r')
        data = f['data'][:]
        if self.split is 'test':
            label = 255  # place holder
        else:
            label = f['label'][:]
        f.close()
        if self.num_point is None:
            self.num_point = data.shape[0]
        idxs = np.arange(data.shape[0])
        if self.random_index:
            np.random.shuffle(idxs)
        idxs = idxs[0:self.num_point]
        data = data[idxs, :]
        if label.size != 1:  # seg data
            label = label[idxs]
        if self.transform is not None:
            data, label = self.transform(data, label)
        return data, label


if __name__ == '__main__':
    data_root = 'dataset/modelnet40'
    data_list = 'dataset/modelnet40/list/val.txt'
    point_data = PointData('train', data_root, data_list)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
