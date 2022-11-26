import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.utils import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class PointNetPP(nn.Module):
    def __init__(self, num_parts, use_normals=False):
        super(PointNetPP, self).__init__()
        if use_normals:
            additional_channel = 3  # x, y, z, Nx, Ny, Nz
        else:
            additional_channel = 0  # x, y ,z
        self.use_normals = use_normals
        self.num_parts = num_parts
        # set abstraction multi-scale grouping
        self.sa1 = PointNetSetAbstractionMsg(512, [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128,  [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        # basic set abstraction
        self.sa3 = PointNetSetAbstraction(n_point=None, n_sample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        # feature propagation
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, self.num_parts, 1)

    def forward(self, xyz, cls_label):
        """
        called in train.py::train() : seg_pred, trans_feat = model(points, onehot(cls, num_part))
        called in train.py::eval()
        :param xyz: tensor(n_points, 3+3*use_normals) float32
        :param cls_label: tensor(num_classes, num_part) with tensor[i, y[i]] = 1, 0 otherwise.
        :return:
            x:
        """

        # Set Abstraction layers
        batch, chl, n_points = xyz.shape

        if self.use_normals:
            l0_points = xyz  # == xyz[:, :6 , :]
            l0_xyz = xyz[:, :3, :]
        else:
            l0_points = xyz  # do not distinguish points and xyz
            l0_xyz = xyz

        cls_label_one_hot = cls_label.view(batch, 16, 1).repeat(1, 1, n_points)

        # set abstraction + unit PointNet
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)

        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))  # batch * chl = 128 * n_points
        x = self.drop1(feat)
        x = self.conv2(x)  # batch * (chl = num_classes) * n_points
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)  # batch * n_points *  (chl = num_classes)
        # print("x", x.shape)
        return x  # , l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):  # trans_feat
        """
        calculate loss
        :param pred:
        :param target:
        :param trans_feat:
        :return:
        """
        total_loss = F.nll_loss(pred, target)
        return total_loss
