import os
import numpy as np
import SharedArray as SA

import torch
from torch.utils.data import Dataset

from util.voxelize import voxelize
from util.data_util import sa_create, collate_fn
from util.data_util import data_prepare_scannet as data_prepare
import glob

class Scannetv2(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()

        self.split = split
        self.data_root = data_root
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.shuffle_index = shuffle_index
        self.loop = loop

        if split == "train" or split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        else:
            raise ValueError("no such split: {}".format(split))
            
        print("voxel_size: ", voxel_size)
        print("Totally {} samples in {} set.".format(len(self.data_list), split))

    def __getitem__(self, idx):
        # data_idx = self.data_idx[idx % len(self.data_idx)]

        # data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        data = torch.load(data_path)

        coord, feat = data[0], data[1]
        if self.split != 'test':
            label = data[2]

        coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return coord, feat, label

    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_list) * self.loop


if __name__ == '__main__':
    data_root = '/home/share/Dataset/s3dis'
    test_area, voxel_size, voxel_max = 5, 0.04, 80000

    point_data = S3DIS(split='train', data_root=data_root, test_area=test_area, voxel_size=voxel_size, voxel_max=voxel_max)
    print('point data size:', point_data.__len__())
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    for idx in range(1):
        end = time.time()
        voxel_num = []
        for i, (coord, feat, label, offset) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            print('tag', coord.shape, feat.shape, label.shape, offset.shape, torch.unique(label))
            voxel_num.append(label.shape[0])
            end = time.time()
    print(np.sort(np.array(voxel_num)))
