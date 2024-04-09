# Helper functions for calculating loss

# Code based on https://github.com/futscdav/strotss

import time
import numpy as np
import torch
import torch.nn.functional as F


def reflectance_loss(texture, mask):
    """
    minimize texture variance (mse), albedo regularization to ensure an uniform skin albedo
    Parameters:
        texture       --torch.tensor, (B, N, 3)
        mask          --torch.tensor, (N), 1 or 0

    """
    mask = mask.reshape([1, mask.shape[0], 1])
    texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
    loss = torch.sum(((texture - texture_mean) * mask)**2) / (texture.shape[0] * torch.sum(mask))
    return loss


def perceptual_loss(id_featureA, id_featureB):
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
    # assert torch.sum((cosine_d > 1).float()) == 0
    return torch.sum(1 - cosine_d) / cosine_d.shape[0]


def pairwise_distances_cos(x, y):
    # x ** 2 为x与x之间个元素之间相乘，sum(1)在行方向上求和，若X为[2048, 2179]，则(x ** 2).sum(1)后的shape为[1, 2048]，view之后为[2048, 1]
    # x_norm的每个元素和每个超列的各元素平方之和再开方，形状为[2048, 1]
    x_norm = torch.sqrt((x ** 2).sum(1).view(-1, 1))
    # y_t为y的转置，如果y的形状为[2048, 2179]，则转置后的矩阵为[2179, 2048]
    y_t = torch.transpose(y, 0, 1)
    # x_norm的每个元素和每个超列的各元素平方之和再开方，形状为[1, 2048]
    y_norm = torch.sqrt((y ** 2).sum(1).view(1, -1))
    # torch.mm为矩阵乘法，例如[1, 2]的矩阵乘以[2, 3]的矩阵结果为[1, 3]的矩阵，dist形状为[2048, 2048]
    dist = 1. - torch.mm(x, y_t) / x_norm / y_norm
    return dist


def pairwise_distances_sq_l2(x, y):
    # (X - Y) * (X - Y) = X^2 + Y^2 - 2X * Y
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 1e-5, 1e5) / x.size(1)


def distmat(x, y, cos_d=True):
    if cos_d:
        M = pairwise_distances_cos(x, y)
    else:
        M = torch.sqrt(pairwise_distances_sq_l2(x, y))
    return M


def rgb_to_yuv(rgb):
    C = torch.Tensor([[0.577350, 0.577350, 0.577350],
                      [-0.577350, 0.788675, -0.211325],
                      [-0.577350, -0.211325, 0.788675]]).to(rgb.device)
    yuv = torch.mm(C, rgb)
    return yuv


def content_loss_new(content_feat, stylized_feat):
    """
    :param content_feat: 从VGG16的深层中提取出来的内容图像的特征图
    :param stylized_feat: 从VGG16的深层中提取出来的待优化图像的特征图
    :return: content loss（与STROTSS不同，这里的内容损失是最初的神经风格迁移算法的版本，即直接用VGG网络的深层特征表示图像的结构特征
    """
    L2loss = torch.nn.MSELoss()
    content_loss = L2loss(content_feat[-2], stylized_feat[-2])
    content_loss += L2loss(content_feat[-1], stylized_feat[-1])
    
    return content_loss


def content_loss(feat_result, feat_content):
    """
    :param feat_result: shape: [1, 2179, 2048, 1]，2048为超列的个数，2179为超列的长度
    :param feat_content: [1, 2179, 2048, 1]，2048为超列的个数，2179为超列的长度
    :return: content loss
    """
    d = feat_result.size(1)
    # X: [2048, 2179], Y: [2048, 2179]，X和Y为超列矩阵，每行为一个超列
    X = feat_result.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
    Y = feat_content.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    Mx = distmat(X, X)
    Mx = Mx / Mx.sum(0, keepdim=True)

    My = distmat(Y, Y)
    My = My / My.sum(0, keepdim=True)

    d = torch.abs(Mx - My).mean() * X.shape[0]

    return d


def remd_loss(X, Y, cos_d=True):
    """
    :param X: shape: [1, 2179, 2048, 1]，2048为超列的个数，2179为超列的长度
    :param Y: shape: [1, 2179, 5000, 1]，5000为超列的个数，2179为超列的长度
    :param cos_d:
    :return: style structure loss
    """
    d = X.shape[1]
    if d == 3:
        X = rgb_to_yuv(X.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
        Y = rgb_to_yuv(Y.transpose(0, 1).contiguous().view(d, -1)).transpose(0, 1)
    else:
        # X: [2048, 2179], Y: [5000, 2179]，X和Y为超列矩阵，每行为一个超列
        X = X.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)
        Y = Y.transpose(0, 1).contiguous().view(d, -1).transpose(0, 1)

    # CX_M: [2048, 5000](这里是以计算Lr为例)
    CX_M = distmat(X, Y, cos_d=cos_d)

    # m1: [2048]，即输出图像中的每个超列与风格图像中所有超列的最小余弦距离
    # m2: [5000]，即风格图像中的每个超列与输出图像中所有超列的最小余弦距离
    m1, m1_inds = CX_M.min(1)
    m2, m2_inds = CX_M.min(0)

    remd = torch.max(m1.mean(), m2.mean())
    # remd = m1.mean()

    flag = 'equal'
    if m1.mean() > m2.mean():
        flag = 'm1'
    elif m1.mean() < m2.mean():
        flag = 'm2'

    return remd, flag


def moment_loss(X, Y, moments=[1, 2]):
    loss = 0.
    X = X.squeeze().t()
    Y = Y.squeeze().t()

    mu_x = torch.mean(X, 0, keepdim=True)
    mu_y = torch.mean(Y, 0, keepdim=True)
    mu_d = torch.abs(mu_x - mu_y).mean()

    if 1 in moments:
        loss = loss + mu_d

    if 2 in moments:
        X_c = X - mu_x
        Y_c = Y - mu_y
        X_cov = torch.mm(X_c.t(), X_c) / (X.shape[0] - 1)
        Y_cov = torch.mm(Y_c.t(), Y_c) / (Y.shape[0] - 1)
        D_cov = torch.abs(X_cov - Y_cov).mean()
        loss = loss + D_cov

    return loss


def TV(x):
    ell = torch.pow(torch.abs(x[:, :, 1:, :] - x[:, :, 0:-1, :]), 2).mean()
    ell += torch.pow(torch.abs(x[:, :, :, 1:] - x[:, :, :, 0:-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:, :, 1:, 1:] - x[:, :, :-1, :-1]), 2).mean()
    ell += torch.pow(torch.abs(x[:, :, 1:, :-1] - x[:, :, :-1, 1:]), 2).mean()
    ell /= 4.
    return ell
