# Stratified Transformer for 3D Point Cloud Segmentation
*Xin Lai<sup>\*</sup>, Jianhui Liu<sup>\*</sup>, Li Jiang, Liwei Wang, Hengshuang Zhao, Shu Liu, Xiaojuan Qi, Jiaya Jia*

This is the official PyTorch implementation of our paper [**Stratified Transformer for 3D Point Cloud Segmentation**](https://arxiv.org/pdf/2203.14508.pdf) that has been accepted to CVPR 2022. [\[arXiv\]](https://arxiv.org/pdf/2203.14508.pdf) [\[CVF\]](https://openaccess.thecvf.com/content/CVPR2022/papers/Lai_Stratified_Transformer_for_3D_Point_Cloud_Segmentation_CVPR_2022_paper.pdf)

<div align="center">
  <img src="figs/fig.jpg"/>
</div>

# Highlight 
1. Our method (*Stratified Transformer*) achieves the state-of-the-art performance on 3D point cloud semantic segmentation on both S3DIS and ScanNetv2 datasets. **It is the first time for a point-based method to outperform the voxel-based ones**, such as SparseConvNet and MinkowskiNet;
2. *Stratified Transformer* is point-based, and constructed by Transformer with standard multi-head self-attention, enjoying large receptive field, robust generalization ability as well as competitive performance;
3. This repository develops a memory-efficient implementation to combat the issue of **variant-length tokens** with several CUDA kernels, avoiding unnecessary momery occupation of vacant tokens. We also use shared memory for further acceleration.

# Get Started

## Environment

1. Install dependencies

```
pip install -r requirements.txt
```

If you have any problem with the above command, you can also install them by

```
pip install torch_sparse==0.6.12
pip install torch_points3d==1.3.0
pip install tensorboard timm termcolor tensorboardX
```

2. Compile pointops

Make sure you have installed `gcc` and `cuda`, and `nvcc` can work (Note that if you install cuda by conda, it won't provide nvcc and you should install cuda manually.). Then, compile and install pointops2 as follows. (We have tested on gcc==7.5.0 and cuda==10.1)
```
cd lib/pointops2
python3 setup.py install
```

## Datasets Preparation

### S3DIS
Please refer to https://github.com/yanx27/Pointnet_Pointnet2_pytorch for S3DIS preprocessing. Then modify the `data_root` entry in the .yaml configuration file.

### ScanNetv2
Please refer to https://github.com/dvlab-research/PointGroup for the ScanNetv2 preprocessing. Then change the `data_root` entry in the .yaml configuration file accordingly.

## Training

### S3DIS
- Stratified Transformer
```
python3 train.py --config config/s3dis/s3dis_stratified_transformer.yaml
```

- 3DSwin Transformer (The vanilla version shown in our paper)
```
python3 train.py --config config/s3dis/s3dis_swin3d_transformer.yaml
```

### ScanNetv2
- Stratified Transformer
```
python3 train.py --config config/scannetv2/scannetv2_stratified_transformer.yaml
```

- 3DSwin Transformer (The vanilla version shown in our paper)
```
python3 train.py --config config/scannetv2/scannetv2_swin3d_transformer.yaml
```

Note: It is normal to see the the results on S3DIS fluctuate between -0.5\% and +0.5\% mIoU maybe because the size of S3DIS is relatively small, while the results on ScanNetv2 are relatively stable.

## Testing
For testing, first change the `model_path`, `save_folder` and `data_root_val` (if applicable) accordingly. Then, run the following command. 
```
python3 test.py --config [YOUR_CONFIG_PATH]
```

## Pre-trained Models

For your convenience, you can download the pre-trained models and training/testing logs from [Here](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155154502_link_cuhk_edu_hk/EihXWr_HEnJIvR_M0_YRbSgBV-6VEIhmbOA9TMyCmKH35Q?e=hLAPNi).


# Citation
If you find this project useful, please consider citing:

```
@inproceedings{lai2022stratified,
  title={Stratified Transformer for 3D Point Cloud Segmentation},
  author={Lai, Xin and Liu, Jianhui and Jiang, Li and Wang, Liwei and Zhao, Hengshuang and Liu, Shu and Qi, Xiaojuan and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8500--8509},
  year={2022}
}
```
