from __future__ import print_function
import paddle
from paddle.nn import Conv1D, ReLU, BatchNorm1D, Linear
import paddle.nn.functional as F
from time import time
import numpy as np
from models.pointnet_plus_plus_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class Pointnet2_cls_msg(paddle.nn.Layer):
    def __init__(self,num_class,normal_channel=True):
        super(Pointnet2_cls_msg, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = Linear(1024, 512)
        self.bn1 = BatchNorm1D(512)
        self.drop1 = paddle.nn.Dropout(p=0.4)
        self.fc2 = Linear(512, 256)
        self.bn2 = BatchNorm1D(256)
        self.drop2 = paddle.nn.Dropout(0.5)
        self.fc3 = Linear(256, num_class)
        self.log_softmax = paddle.nn.LogSoftmax(axis=-1)


    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.reshape((B, 1024))
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = self.log_softmax(x)

        return x,l3_points

