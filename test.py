import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from util import config, transform
from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs
from util.voxelize import voxelize
import torch_points_kernels as tp
import torch.nn.functional as F

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Classification / Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointweb.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointweb.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    # get model
    if args.arch == 'stratified_transformer':
        
        from model.stratified_transformer import Stratified

        args.patch_size = args.grid_size * args.patch_size
        args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
            args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    elif args.arch == 'swin3d_transformer':
        
        from model.swin3d_transformer import Swin

        args.patch_size = args.grid_size * args.patch_size
        args.window_sizes = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
        args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
        args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

        model = Swin(args.depths, args.channels, args.num_heads, \
            args.window_sizes, args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
            rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, \
            concat_xyz=args.concat_xyz, num_classes=args.classes, \
            ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

    else:
        raise Exception('architecture {} not supported yet'.format(args.arch))
    
    model = model.cuda()

    #model = torch.nn.DataParallel(model.cuda())
    logger.info(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    names = [line.rstrip('\n') for line in open(args.names_path)]
    if os.path.isfile(args.model_path):
        logger.info("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name.replace("item", "stem")] = v
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))


    # transform
    test_transform_set = []
    test_transform_set.append(None) # for None aug
    test_transform_set.append(None) # for permutate

    # aug 90
    logger.info("augmentation roate")
    logger.info("rotate_angle: {}".format(90))
    test_transform = transform.RandomRotate(rotate_angle=90, along_z=args.get('rotate_along_z', True))
    test_transform_set.append(test_transform)
    
    # aug 180
    logger.info("augmentation roate")
    logger.info("rotate_angle: {}".format(180))
    test_transform = transform.RandomRotate(rotate_angle=180, along_z=args.get('rotate_along_z', True))
    test_transform_set.append(test_transform)
    
    # aug 270
    logger.info("augmentation roate")
    logger.info("rotate_angle: {}".format(270))
    test_transform = transform.RandomRotate(rotate_angle=270, along_z=args.get('rotate_along_z', True))
    test_transform_set.append(test_transform)
    
    if args.data_name == 's3dis':
        
        # shift +0.2
        test_transform = transform.RandomShift_test(shift_range=0.2)
        test_transform_set.append(test_transform)

        # shift -0.2
        test_transform = transform.RandomShift_test(shift_range=-0.2)
        test_transform_set.append(test_transform)

    test(model, criterion, names, test_transform_set)


def data_prepare():
    if args.data_name == 's3dis':
        data_list = sorted(os.listdir(args.data_root))
        data_list = [item[:-4] for item in data_list if 'Area_{}'.format(args.test_area) in item]
    elif args.data_name == 'scannetv2':
        data_list = sorted(os.listdir(args.data_root_val))
        data_list = [item[:-4] for item in data_list if '.pth' in item]
        # data_list = sorted(glob.glob(os.path.join(args.data_root_val, "*.pth")))
    else:
        raise Exception('dataset not supported yet'.format(args.data_name))
    print("Totally {} samples in val set.".format(len(data_list)))
    return data_list


def data_load(data_name, transform):

    if args.data_name == 's3dis':
        data_path = os.path.join(args.data_root, data_name + '.npy')
        data = np.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[:, :3], data[:, 3:6], data[:, 6]
    elif args.data_name == 'scannetv2':
        data_path = os.path.join(args.data_root_val, data_name + '.pth')
        data = torch.load(data_path)  # xyzrgbl, N*7
        coord, feat, label = data[0], data[1], data[2]
        # print("type(coord): {}".format(type(coord)))

    if transform:
        coord, feat = transform(coord, feat)

    idx_data = []
    if args.voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, args.voxel_size, mode=1)
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]
            idx_data.append(idx_part)
    else:
        idx_data.append(np.arange(label.shape[0]))
    return coord, feat, label, idx_data


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    if args.data_name == 's3dis':
        feat = feat / 255.
    return coord, feat


def test(model, criterion, names, test_transform_set):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    args.batch_size_test = 5 
    # args.voxel_max = None
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    data_list = data_prepare()
    for idx, item in enumerate(data_list):
        end = time.time()
        pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
        label_save_path = os.path.join(args.save_folder, '{}_{}_label.npy'.format(item, args.epoch))
        
        if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
            logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
            pred, label = np.load(pred_save_path), np.load(label_save_path)
        else:
            # ensemble output
            pred_all = 0
            for aug_id in range(len(test_transform_set)):
                test_transform = test_transform_set[aug_id]
                    
                if os.path.isfile(pred_save_path) and os.path.isfile(label_save_path):
                    logger.info('{}/{}: {}, loaded pred and label.'.format(idx + 1, len(data_list), item))
                    pred, label = np.load(pred_save_path), np.load(label_save_path)
                else:
                    coord, feat, label, idx_data = data_load(item, test_transform)
                    pred = torch.zeros((label.size, args.classes)).cuda()
                    idx_size = len(idx_data)
                    idx_list, coord_list, feat_list, offset_list  = [], [], [], []
                    for i in range(idx_size):
                        logger.info('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
                        idx_part = idx_data[i]
                        coord_part, feat_part = coord[idx_part], feat[idx_part]
                        if args.voxel_max and coord_part.shape[0] > args.voxel_max:
                            coord_p, idx_uni, cnt = np.random.rand(coord_part.shape[0]) * 1e-3, np.array([]), 0
                            while idx_uni.size != idx_part.shape[0]:
                                init_idx = np.argmin(coord_p)
                                dist = np.sum(np.power(coord_part - coord_part[init_idx], 2), 1)
                                idx_crop = np.argsort(dist)[:args.voxel_max]
                                coord_sub, feat_sub, idx_sub = coord_part[idx_crop], feat_part[idx_crop], idx_part[idx_crop]
                                dist = dist[idx_crop]
                                delta = np.square(1 - dist / np.max(dist))
                                coord_p[idx_crop] += delta
                                coord_sub, feat_sub = input_normalize(coord_sub, feat_sub)
                                idx_list.append(idx_sub), coord_list.append(coord_sub), feat_list.append(feat_sub), offset_list.append(idx_sub.size)
                                idx_uni = np.unique(np.concatenate((idx_uni, idx_sub)))
                                # cnt += 1; logger.info('cnt={}, idx_sub/idx={}/{}'.format(cnt, idx_uni.size, idx_part.shape[0]))
                        else:
                            coord_part, feat_part = input_normalize(coord_part, feat_part)
                            idx_list.append(idx_part), coord_list.append(coord_part), feat_list.append(feat_part), offset_list.append(idx_part.size)
                    batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
                    for i in range(batch_num):
                        s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
                        idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                        idx_part = np.concatenate(idx_part)
                        coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                        feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
                        offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                        with torch.no_grad():
                            
                            offset_ = offset_part.clone()
                            offset_[1:] = offset_[1:] - offset_[:-1]
                            batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long().cuda(non_blocking=True)

                            sigma = 1.0
                            radius = 2.5 * args.grid_size * sigma
                            neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord_part, coord_part, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
                            neighbor_idx = neighbor_idx.cuda(non_blocking=True)

                            if args.concat_xyz:
                                feat_part = torch.cat([feat_part, coord_part], 1)

                            pred_part = model(feat_part, coord_part, offset_part, batch, neighbor_idx)
                            pred_part = F.softmax(pred_part, -1) # Add softmax

                        torch.cuda.empty_cache()
                        pred[idx_part, :] += pred_part
                        logger.info('Test: {}/{}, {}/{}, {}/{}, {}/{}'.format(aug_id+1, len(test_transform_set), idx + 1, len(data_list), e_i, len(idx_list), args.voxel_max, idx_part.shape[0]))
                pred = pred / (pred.sum(-1)[:, None]+1e-8)
                pred_all += pred
            pred = pred_all / len(test_transform_set)
            loss = criterion(pred, torch.LongTensor(label).cuda(non_blocking=True))  # for reference
            pred = pred.max(1)[1].data.cpu().numpy()

        # calculation 1: add per room predictions
        intersection, union, target = intersectionAndUnion(pred, label, args.classes, args.ignore_label)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        batch_time.update(time.time() - end)
        logger.info('Test: [{}/{}]-{} '
                    'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Accuracy {accuracy:.4f}.'.format(idx + 1, len(data_list), label.size, batch_time=batch_time, accuracy=accuracy))
        pred_save.append(pred); label_save.append(label)
        if not os.path.isfile(pred_save_path):
            np.save(pred_save_path, pred)
            
        if not os.path.isfile(label_save_path):
            np.save(label_save_path, label)

    if not os.path.exists(os.path.join(args.save_folder, "pred.pickle")):
        with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
            pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    if not os.path.exists(os.path.join(args.save_folder, "label.pickle")):
        with open(os.path.join(args.save_folder, "label.pickle"), 'wb') as handle:
            pickle.dump({'label': label_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), args.classes, args.ignore_label)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    logger.info('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(args.classes):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}, name: {}.'.format(i, iou_class[i], accuracy_class[i], names[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')


if __name__ == '__main__':
    main()
