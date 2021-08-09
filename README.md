
# pointnet_plus_plus_paddlepaddle

**Paper:** PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
## 一、简介
![prediction example](https://github.com/charlesq34/pointnet2/blob/master/doc/teaser.jpg)
PointNet++与PointNet相比网络可以更好的提取局部特征。网络使用空间距离（metric space distances），使用PointNet对点集局部区域进行特征迭代提取，使其能够学到局部尺度越来越大的特征。基于自适应密度的特征提取方法，解决了点集分布不均匀的问题。
[PointNet++](https://arxiv.org/abs/1706.02413)
## 二、复现精度
| 指标 | 原论文 | 复现精度 |
| --- | --- | --- |
| top-1 Acc | 90.7 | 92.0 |

## 三、数据集
使用的数据集为：[ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip)。

## 四、环境依赖

- 硬件：GPU、CPU

- 框架：
  - PaddlePaddle >= 2.0.0
  - tqdm

## 五、快速开始
### Data Preparation
Download [alignment ModelNet](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and put it in `./dataset/modelnet40_normal_resampled/`

### Train
```
python train_modelnet.py --process_data
```

### Test
```
python test_modelnet.py --log_dir path_to_model
```

## 六、代码结构与详细说明

### 6.1 代码结构

```
|—— README.md
|—— provider.py    # 点云数据增强
|—— ModelNetDataset.py # 数据集定义及加载
|── train_modelnet.py       # 训练网络
|── test_modelnet.py     # 测试网络
|—— models        # 模型文件定义
```
### 6.2 参数说明

可以在 `train_modelnet.py` 中设置训练与评估相关参数，具体如下：

|  参数   | 默认值  | 说明 | 其他 |
|  ----  |  ----  |  ----  |  ----  |
| batch_size  | 24 | batch_size 大小 ||
| epoch  | 200, 可选 | epoch次数 ||
| batch_size  | 32, 可选 | batch_size 大小 ||
| learning_rate | 0.001, 可选 | 初始学习率 ||
| num_point | 1024, 可选 | 采样的点的个数 ||
| decay_rate | 1e-4, 可选 | weight decay ||
| use_normals | False, 可选 | normalize 点 ||
| use_uniform_sample | False, 可选 | 均匀采样 ||
| process_data | False, 可选 | 是否预处理数据，如果没有下载预处理的数据需要为true ||

**Reference Implementation:**
* [TensorFlow (Official)](https://github.com/charlesq34/pointnet2)
* [PyTorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)

