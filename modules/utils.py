"""
Utility functions & Basic Network Blocs
"""

import torch
import torch.nn as nn
import logging
import torch.nn.functional as F


def onehot(y, num_classes):
    """
    turn y:labels into one-hot encoding
    :param y: tensor(n, 1) labels
    :param num_classes: int m
    :return: tensor(n, m) with tensor[i, y[i]] = 1, 0 otherwise.
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]  # magic transformation !
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)


def log_string(logger, str):
    """
    print training messages and write to log
    :param str: log message
    """
    logger.info(str)
    print(str)


def fps(xyz, n_point):
    """
    Farthest point (down)sampling : down sample the pointcloud to n_point points.
    :param xyz: point cloud, torch.tensor(
    :param n_point: target number of samples
    :return: sampled points [batch * n_point]
    """
    if xyz.shape[1] < n_point:
        print("Warning - Upsampling : Number of points < target number.")

    batch, N, chl = xyz.shape

    # initilisation ---------------
    samples_index = torch.zeros(batch, n_point, dtype=torch.long).to(xyz.device)
    distance = torch.inf * torch.ones(batch, N).to(xyz.device)
    # sample one index between [0,N] in each cloud : batch_size times
    farthest_index = torch.randint(0, N, (batch,), dtype=torch.long).to(xyz.device)
    batch_index = torch.arange(batch, dtype=torch.long).to(xyz.device)

    # interation --------------
    # samples_index : batch * N
    # distance : batch * N
    # farthest_index : batch,
    # batch_index : [0, ..., batch] static, for indexing
    for i in range(n_point):
        samples_index[:, i] = farthest_index  # batch,
        sampled_xyz = xyz[batch_index, farthest_index, :].unsqueeze(1)  # .view(batch, 1, 3)
        dist = torch.sum((xyz - sampled_xyz) ** 2, dim=2)
        distance[dist < distance] = dist[dist < distance]
        farthest_index = torch.max(distance, dim=-1)[1]  # values/indices
    return samples_index


def euclidean_distance(src, dst):
    """
    Calculate the Euclidean Distance Matrix between 2 clouds
    :param src: source (Batch, n_points, Channels)
    :param dst: target (Batch, Mpoints, Channels )
    :return: distance_matrix : batch * N * M
    """
    assert src.shape[0] == dst.shape[0], "Clouds should have the same Batch Size"
    # batch_size = src.shape[0]
    # N, M = src.shape[1], dst.shape[1]
    # euclidean
    src = src[:, :, 0:3]
    dst = dst[:, :, 0:3]

    cross_terms = torch.matmul(src, dst.permute(0, 2, 1))
    n_squares = torch.sum(src ** 2, dim=2, keepdim=True)  #
    m_squares = torch.sum(dst ** 2, dim=2, keepdim=True)
    return -2 * cross_terms + n_squares + m_squares.permute(0, 2, 1)


def knn(k, xyz, new_xyz):
    """
    compute knn of the sampled points (centroids)
    :param k: k of knn
    :param xyz: all points [batch, N, 3]
    :param new_xyz: centroids [batch, N1, 3] N1 < N
    :return: [batch, N1, k] N1 groups of point indices.
    """
    dist_matrix = euclidean_distance(new_xyz, xyz)
    knn_result = dist_matrix.topk(k, largest=False)
    return knn_result.indices  # batch * N1 * k


def sample_and_group(n_point, n_sample, xyz, features, return_fps=False):
    """
    Down sample and group points
    Input:
        n_point: (int)
        n_sample: (int)
        xyz: input points position data, [batch, N, 3]
        features: input points data, [batch, N, D]
    Return:
        new_xyz: sampled points position data, [batch, n_point, n_sample, 3]
        new_points: sampled points data, [batch, n_point, n_sample, 3+D]
    """
    batch, nb, chl = xyz.shape

    # down sampling -> new_xyz
    fps_idx = fps(xyz, n_point)
    new_xyz = get_points_from_index(xyz, fps_idx)  # [batch, n_point, chl]
    # grouping and normalization
    idx = knn(n_sample, xyz, new_xyz)  # [batch, n_point, k=n_sample]
    grouped_xyz = get_points_from_index(xyz, idx)  # [batch, n_point, k=n_sample, chl]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(batch, n_point, 1, chl)  # very important !!! see article

    if features is not None:
        grouped_features = get_points_from_index(features, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)  # [batch, n_point, n_sample, chl+D]
    else:
        new_points = grouped_xyz_norm

    if return_fps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, features):
    """
    Input:
        xyz: input points position data, [batch, n_points, chl=3]
        features: input features, [batch, n_points, feat_dim]
    Return:
        new_xyz: sampled points position data, [batch, n_points, chl=3]
        new_points: sampled points data, [batch, 1, n_points, chl + feat_dim]
    """
    device = xyz.device
    batch, n_points, chl = xyz.shape
    # print("sample and groupe all shape:", xyz.shape, features.shape)

    new_xyz = torch.zeros(batch, 1, chl).to(device)
    grouped_xyz = xyz.view(batch, 1, n_points, chl)

    if features is not None:
        new_points = torch.cat([grouped_xyz, features.view(batch, 1, n_points, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def get_points_from_index(points, idx):
    """
    the gather_points function in the official tensorflow implementation
    Input:
        points: input points data, [batch, n_points, chl]
        idx: sample index data, [batch, n_sample]
    Return:
        new_points:, indexed points data, [batch, n_sample, chl]
    """
    device = points.device
    batch = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, n_point, n_sample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.n_point = n_point
        self.n_sample = n_sample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [batch, chl, N]
            points: input points data, [batch, D, N]
        Return:
            new_xyz: sampled points position data, [batch, chl, S]
            new_points_concat: sample points feature data, [batch, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.n_point, self.n_sample, xyz, points)
        # new_xyz: sampled points position data, [batch, n_point, chl]
        # new_points: sampled points data, [batch, n_point, n_sample, chl+D]
        new_points = new_points.permute(0, 3, 2, 1)  # [batch, chl+D, n_sample,n_point]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [batch, chl, N]
            xyz2: sampled input points position data, [batch, chl, S]
            points1: input points data, [batch, D, N]
            points2: input points data, [batch, D, S]
        Return:
            new_points: upsampled points data, [batch, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        batch, n_points_1, chl = xyz1.shape
        _, n_points_2, _ = xyz2.shape

        if n_points_2 == 1:  # or <=1
            # if the point set 2 has only one single point (suppose not empty)
            # then simply duplicate them !
            interpolated_points = points2.repeat(1, n_points_1, 1)
        else:
            # if the point set 2 has more than one point
            # interpolate point attributes as weighted sum (by euclidean distance) of neighbors
            dists = euclidean_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [batch, n_points_1, 3]

            # avoid divided by 0 ERROR
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(get_points_from_index(points2, idx) * weight.view(batch, n_points_1, 3, 1),
                                            dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            # merge points
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class PointNetSetAbstractionMsg(nn.Module):
    # MSG stands for Multi-scale grouping(MSG)
    # knn version implemented instead of ball query
    def __init__(self, n_point, n_sample_list, in_channel, mlp_list):
        """
        Set Abstraction Layer with Multi-Scale Grouping
        Args:
            n_point: number of input points to be sampled
            n_sample_list: list of #Sampling Points for knn e.g. [32, 64, 128]
            in_channel: input #channels
            mlp_list:  list of consecutive dense layers (specifying the chls)
        """
        super(PointNetSetAbstractionMsg, self).__init__()
        self.n_point = n_point
        self.n_sample_list = n_sample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        # initialize conv & bn blocks from mlp list
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [batch, chl, N]
            points: input points data, [batch, D, N]
        Return:
            new_xyz: sampled points position data, [batch, chl, S]
            new_points_concat: sample points feature data, [batch, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        batch, N, chl = xyz.shape

        # sample n_point points from the input point set
        S = self.n_point
        index = fps(xyz, S)
        new_xyz = get_points_from_index(xyz, index)
        # print(new_xyz.shape, xyz.shape, index.shape, S, self.n_sample_list)

        new_points_list = []  # to store output feature vectors
        # perform knn in different scales and pass the points through MLPs to get embeddings
        for i, K in enumerate(self.n_sample_list):
            group_idx = knn(K, xyz, new_xyz)
            grouped_xyz = get_points_from_index(xyz, group_idx)
            grouped_xyz -= new_xyz.view(batch, S, 1, chl)
            if points is not None:
                grouped_points = get_points_from_index(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [batch, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [batch, D', S]
            new_points_list.append(new_points)

        # feature concatenation
        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetSetAbstractionMrg(nn.Module):
    # Mrg stands for Multi-resolution grouping(MRG)
    # knn version implemented instead of ball query
    def __init__(self, n_point, n_sample, in_channel_previous, in_channel, mlps, convs):
        super(PointNetSetAbstractionMrg, self).__init__()
        self.n_point = n_point
        self.n_sample = n_sample
        self.n_sample_list = [n_sample]
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel_previous
        for out_channel in mlps:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        convs_ = nn.ModuleList()
        bns_ = nn.ModuleList()
        last_channel = in_channel + 3
        for out_channel in convs:
            convs_.append(nn.Conv2d(last_channel, out_channel, 1))
            bns_.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.conv_blocks.append(convs_)
        self.bn_blocks.append(bns_)

    def forward(self, previous_xyz, previous_points, xyz, points):
        """
        Input:
            xyz: input points position data, [batch, chl, N]
            points: input points data, [batch, D, N]
        Return:
            new_xyz: sampled points position data, [batch, chl, S]
            new_points_concat: sample points feature data, [batch, D', S]
        """
        # Ssg on previous xyz
        previous_xyz = previous_xyz.permute(0, 2, 1)
        if previous_points is not None:
            previous_points = previous_points.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        new_previous_xyz, new_previous_points = sample_and_group(self.n_point, self.n_sample, previous_xyz,
                                                                 previous_points)
        # new_xyz: sampled points position data, [batch, n_point, chl]
        # new_points: sampled points data, [batch, n_point, n_sample, chl+D]
        new_previous_points = new_previous_points.permute(0, 3, 2, 1)  # [batch, chl+D, n_sample,n_point]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_previous_points = F.relu(bn(conv(new_previous_points)))

        new_previous_points = torch.max(new_previous_points, 2)[0]
        new_previous_xyz = new_previous_xyz.permute(0, 2, 1)

        # single Msg on current xyz
        batch, chl, N = new_previous_xyz.shape
        xyz = xyz.permute(0, 2, 1)
        new_previous_xyz = new_previous_xyz.permute(0, 2, 1)
        # print(new_previous_xyz.shape) # 16 128 3
        new_points_list = [new_previous_points]  # merge with previous
        for i, K in enumerate(self.n_sample_list):
            # print("xyz shape : ", xyz.shape, new_previous_xyz.shape, K)
            group_idx = knn(K, xyz, new_previous_xyz)
            # print("grouped_idx: ", group_idx.shape)
            grouped_xyz = get_points_from_index(xyz, group_idx)
            # print("grouped_xyz: ", grouped_xyz.shape)
            grouped_xyz -= new_previous_xyz.view(batch, N, 1, chl)
            if points is not None:
                grouped_points = get_points_from_index(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz
            # print("grouped points: ", grouped_points.shape)

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [batch, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points = F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [batch, D', S]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1)
        new_previous_xyz = new_previous_xyz.permute(0, 2, 1)
        return new_previous_xyz, new_points_concat
