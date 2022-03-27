import numpy as np
import random
import SharedArray as SA

import torch

from util.voxelize import voxelize


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def collate_fn_limit_mix3d(batch, max_batch_points, logger, p):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    k = 0
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coord])
        s_now = sum([x.shape[0] for x in coord[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    if random.random() <= p:
        coord_mix3d = []
        feat_mix3d = []
        label_mix3d = []
        offset_mix3d = []
        for i in range(0, k, 2):
            if i == k-1:
                coord_mix3d_i = coord[i]
                feat_mix3d_i = feat[i]
                label_mix3d_i = label[i]
                offset_mix3d.append(offset[i])
            else:
                coord_mix3d_i = torch.cat([coord[i], coord[i+1]], 0)
                feat_mix3d_i = torch.cat([feat[i], feat[i+1]], 0)
                label_mix3d_i = torch.cat([label[i], label[i+1]], 0)
                offset_mix3d.append(offset[i+1])
            coord_mix3d.append(coord_mix3d_i)
            feat_mix3d.append(feat_mix3d_i)
            label_mix3d.append(label_mix3d_i)
        return torch.cat(coord_mix3d), torch.cat(feat_mix3d), torch.cat(label_mix3d), torch.IntTensor(offset_mix3d)

    return torch.cat(coord[:k]), torch.cat(feat[:k]), torch.cat(label[:k]), torch.IntTensor(offset[:k])
    # return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)
    # return torch.cat(coord_mix3d), torch.cat(feat_mix3d), torch.cat(label_mix3d), torch.IntTensor(offset_mix3d)


def collate_fn_limit(batch, max_batch_points, logger):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    k = 0
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        if count > max_batch_points:
            break
        k += 1
        offset.append(count)

    if logger is not None and k < len(batch):
        s = sum([x.shape[0] for x in coord])
        s_now = sum([x.shape[0] for x in coord[:k]])
        logger.warning("batch_size shortened from {} to {}, points from {} to {}".format(len(batch), k, s, s_now))

    return torch.cat(coord[:k]), torch.cat(feat[:k]), torch.cat(label[:k]), torch.IntTensor(offset[:k])
    # return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)

def collate_fn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    # print("coord:", len(coord))
    for item in coord:
        # print("item shape:",item.shape)
        count += item.shape[0]
        offset.append(count)
    return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset)


def area_crop(coord, area_rate, split='train'):
    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= coord_min; coord_max -= coord_min
    x_max, y_max = coord_max[0:2]
    x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
    if split == 'train' or split == 'trainval':
        x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
    else:
        x_s, y_s = (x_max - x_size) / 2, (y_max - y_size) / 2
    x_e, y_e = x_s + x_size, y_s + y_size
    crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
    return crop_idx


def load_kitti_data(data_path):
    data = np.fromfile(data_path, dtype=np.float32)
    data = data.reshape((-1, 4))  # xyz+remission
    return data


def load_kitti_label(label_path, remap_lut):
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape(-1)
    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half
    assert ((sem_label + (inst_label << 16) == label).all())
    sem_label = remap_lut[sem_label]
    return sem_label.astype(np.int32)


def data_prepare(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
    coord -= (coord_min + coord_max) / 2.0
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v101(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_scannet(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        # coord, feat, label = transform(coord, feat, label)
        coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat)
    label = torch.LongTensor(label)
    return coord, feat, label

def data_prepare_v102(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    while voxel_max and label.shape[0] > voxel_max * 1.1:
        area_rate = voxel_max / float(label.shape[0])
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        x_max, y_max = coord_max[0:2]
        x_size, y_size = np.sqrt(area_rate) * x_max, np.sqrt(area_rate) * y_max
        if split == 'train':
            x_s, y_s = random.uniform(0, x_max - x_size), random.uniform(0, y_max - y_size)
        else:
            x_s, y_s = 0, 0
        x_e, y_e = x_s + x_size, y_s + y_size
        crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
        if crop_idx.shape[0] < voxel_max // 8: continue
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]

    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v103(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        xy_area = 7
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(xy_area)
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[1] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[1] * (y_area + 1) / float(xy_area)
            crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
            if crop_idx.shape[0] > 0:
                init_idx = crop_idx[np.random.randint(crop_idx.shape[0])] if 'train' in split else label.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
                coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v104(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
        coord -= coord_min; coord_max -= coord_min
        xy_area = 10
        while True:
            x_area, y_area = np.random.randint(xy_area), np.random.randint(xy_area)
            x_s, y_s = coord_max[0] * x_area / float(xy_area), coord_max[1] * y_area / float(xy_area)
            x_e, y_e = coord_max[0] * (x_area + 1) / float(xy_area), coord_max[1] * (y_area + 1) / float(xy_area)
            crop_idx = np.where((coord[:, 0] >= x_s) & (coord[:, 0] <= x_e) & (coord[:, 1] >= y_s) & (coord[:, 1] <= y_e))[0]
            if crop_idx.shape[0] > 0:
                init_idx = crop_idx[np.random.randint(crop_idx.shape[0])] if 'train' in split else label.shape[0] // 2
                crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
                coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
                break
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord -= coord_min
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label


def data_prepare_v105(coord, feat, label, split='train', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False):
    if transform:
        coord, feat, label = transform(coord, feat, label)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
    if voxel_max and label.shape[0] > voxel_max:
        init_idx = np.random.randint(label.shape[0]) if 'train' in split else label.shape[0] // 2
        crop_idx = np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        coord, feat, label = coord[crop_idx], feat[crop_idx], label[crop_idx]
    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord[:, 0:2] -= coord_min[0:2]
    coord = torch.FloatTensor(coord)
    feat = torch.FloatTensor(feat) / 255.
    label = torch.LongTensor(label)
    return coord, feat, label
