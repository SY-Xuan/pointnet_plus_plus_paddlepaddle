# 复现总结与心得
## 问题
复现主要参考的是[PyTorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)的pytorch实现，pytorch的大部分api可以在paddlepaddle中找到对应，最困难的地方在于，paddlepaddle没法办法进行二维的索引，对应原实现中的多个部分
```python
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
```

```python
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
```

```python
def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
```

```python
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points
```

这里所有的二维索引都没有办法使用，包括经常使用的mask方法

这里进行了一些妥协，将需要二维索引的地方进行拉直，从而可以将二维索引变为n个一维索引，但是这里肯定对速度有所损失，暂时没有想到好的办法
```python
farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
batch_indices = torch.arange(B, dtype=torch.long).to(device)
for i in range(npoint):
    centroids[:, i] = farthest
    centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
```

```python
farthest = paddle.randint(0, N, (B, ), dtype="int64")
for i in range(npoint):
    centroids[:, i] = farthest
    # centroid = xyz[batch_indices, farthest, :].reshape((B, 1, 3))
    centroid = paddle.zeros((B, 1, 3), dtype="float32")
    for j in range(3):
        centroid[:,:,j] = xyz[:,:,j].index_sample(farthest.reshape((-1, 1))).reshape((B, 1))
```

对于mask的地方，可以直接使用数值运算的方法达到mask的目的
```python
mask = dist < distance
distance[mask] = dist[mask]
farthest = torch.max(distance, -1)[1]
```

```python
mask = dist < distance
mask = mask.astype("int64")
mask_index = paddle.nonzero(mask)

if mask_index.size > 0:
    distance = distance * (1 - mask.astype("float32")) + dist * mask.astype("float32")
```

## 总结
目前的paddlepaddle可以支持大多数pytorch的API，但是其对于多维索引的支持不足，非常影响使用体验，而多维索引又是一个在日常的研究以及工程中，非常常规的功能，这里需要改进。