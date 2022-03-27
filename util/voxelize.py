import numpy as np
from collections import Sequence
import torch
from torch_geometric.nn import voxel_grid

def grid_sample(pos, batch_index, size, start=None, return_p2v=True):
    # pos: float [N, 3]
    # batch_szie: long int
    # size: float [3, ]
    # start: float [3, ] / None

    # print("pos.shape: {}, batch.shape: {}".format(pos.shape, batch.shape))
    # print("size: ", size)

    # batch [N, ]
    batch = torch.zeros(pos.shape[0])
    for i in range (1, len(batch_index)):
        batch[batch_index[i-1]:batch_index[i]] = i
        
    cluster = voxel_grid(pos, batch, size, start=start) #[N, ]

    if return_p2v == False:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster

    unique, cluster, counts = torch.unique(cluster, sorted=True, return_inverse=True, return_counts=True)

    # print("unique.shape: {}, cluster.shape: {}, counts.shape: {}".format(unique.shape, cluster.shape, counts.shape))

    # input()

    # obtain p2v_map
    n = unique.shape[0]
    k = counts.max().item()
    p2v_map = cluster.new_zeros(n, k) #[n, k]
    mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1) #[n, k]
    p2v_map[mask] = torch.argsort(cluster)
    # max_point
    max_point = 48
    if k > max_point:
        counts = torch.where(counts > max_point, max_point, counts)
        p2v_map = p2v_map[:,0:max_point]

    return cluster, p2v_map, counts

def fnv_hash_vec(arr):
    """
    FNV64-1A
    """
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
    for j in range(arr.shape[1]):
        hashed_arr *= np.uint64(1099511628211)
        hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr


def ravel_hash_vec(arr):
    """
    Ravel the coordinates after subtracting the min coordinates.
    """
    assert arr.ndim == 2
    arr = arr.copy()
    arr -= arr.min(0)
    arr = arr.astype(np.uint64, copy=False)
    arr_max = arr.max(0).astype(np.uint64) + 1

    keys = np.zeros(arr.shape[0], dtype=np.uint64)
    # Fortran style indexing
    for j in range(arr.shape[1] - 1):
        keys += arr[:, j]
        keys *= arr_max[j + 1]
    keys += arr[:, -1]
    return keys


def voxelize(coord, voxel_size=0.05, hash_type='fnv', mode=0):
    discrete_coord = np.floor(coord / np.array(voxel_size))
    if hash_type == 'ravel':
        key = ravel_hash_vec(discrete_coord)
    else:
        key = fnv_hash_vec(discrete_coord)

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, count = np.unique(key_sort, return_counts=True)
    if mode == 0:  # train mode
        idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
        idx_unique = idx_sort[idx_select]
        return idx_unique
    else:  # val mode
        return idx_sort, count

    '''
    #_, idx = np.unique(key, return_index=True)
    #return idx

    idx_sort = np.argsort(key)
    key_sort = key[idx_sort]
    _, idx_start, count = np.unique(key_sort, return_counts=True, return_index=True)
    idx_list = np.split(idx_sort, idx_start[1:])
    return idx_list
    '''
