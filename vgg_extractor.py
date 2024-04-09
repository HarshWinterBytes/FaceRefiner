# Extract features from a pretrained VGG16

# Code from https://github.com/futscdav/strotss

from pickle import FALSE
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt


class VGG16_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_layers = models.vgg16(pretrained=True).features

        for param in self.parameters():
            param.requires_grad = False
        self.capture_layers = [1, 3, 6, 8, 11, 13, 15, 22, 29]

    def forward_base(self, x, add_noise=False):
        feat = [x]
        # if add_noise:
        #     batch, _, height, width = x.shape
        #     noise = x.new_empty(batch, 1, height, width).normal_()
        #     feat[0] = feat[0] + noise
        for i in range(len(self.vgg_layers)):
            x = self.vgg_layers[i](x)
            if i in self.capture_layers:
                if add_noise and i >= 15:
                    batch, _, height, width = x.shape
                    noise = x.new_empty(batch, 1, height, width).normal_()
                    x = x + noise
                feat.append(x)
        return feat

    def forward(self, x, add_noise=False):
        x = (x + 1.) / 2.
        x = x - (torch.Tensor([0.485, 0.456, 0.406]).to(x.device).view(1, -1, 1, 1))
        x = x / (torch.Tensor([0.229, 0.224, 0.225]).to(x.device).view(1, -1, 1, 1))
        feat = self.forward_base(x, add_noise)
        return feat

    def forward_samples_hypercolumn(self, X, samps=100, add_noise=False):
        """
        提取风格图像的特征
        X为风格图像
        samps为超列采样点的数量
        """
        feat = self.forward(X, add_noise)

        # 生成128 × 128的网格点矩阵（以X的shape为[1, 3, 128, 128]为例）
        xx, xy = np.meshgrid(np.arange(X.shape[2]), np.arange(X.shape[3]))
        # xx: [128 * 128] -> [16384, ] -> [16384, 1]
        xx = np.expand_dims(xx.flatten(), 1)
        xy = np.expand_dims(xy.flatten(), 1)
        # xc: [16384, 2]
        xc = np.concatenate([xx, xy], 1)

        samples = min(samps, xc.shape[0])

        np.random.shuffle(xc)
        # xx: [1000, ], yy: [1000, ]
        xx = xc[:samples, 0]
        yy = xc[:samples, 1]

        # plt.xlim(0, feat[0].size(2))
        # plt.ylim(0, feat[0].size(3))
        # plt.scatter(xx, X.shape[2] - yy, c='r')
        # plt.show()
        # input('wait')

        feat_samples = []
        for i in range(len(feat)):

            layer_feat = feat[i]

            # Hack to detect lower resolution
            if i > 0 and feat[i].size(2) < feat[i - 1].size(2):
                xx = xx / 2.0
                yy = yy / 2.0

            xx = np.clip(xx, 0, layer_feat.shape[2] - 1).astype(np.int32)
            yy = np.clip(yy, 0, layer_feat.shape[3] - 1).astype(np.int32)

            # 在每个feature map上随机取1000个点
            features = layer_feat[:, :, xx[range(samples)], yy[range(samples)]]
            feat_samples.append(features.clone().detach())

        # feat为选定的VGG网络层提取出来的所有的特征
        feat = torch.cat(feat_samples, 1)
        return feat

    def forward_samples_hypercolumn_by_mask(self, mask, X, samps=100):
        """
        根据mask区域局部提取风格图像的特征
        mask为采样区域
        X为风格图像
        samps为超列采样点的数量
        """
        feat = self.forward(X)
        g = np.array(np.argwhere(mask[0][0] > 0))
        samples = min(samps, g.shape[1])
        indices = random.sample(range(0, g.shape[1]), samples)
        # indexes = np.array([random.randint(0, g.shape[0] - 1) for _ in range(samples)])
        # xx = g[indices, 0]
        # yy = g[indices, 1]

        xx = g[0, indices]
        yy = g[1, indices]

        # plt.xlim(0, feat[0].size(2))
        # plt.ylim(0, feat[0].size(3))
        # plt.scatter(xx, mask.shape[3] - yy, c='r')
        # plt.show()
        # input('wait')

        feat_samples = []
        for i in range(len(feat)):

            layer_feat = feat[i]

            # Hack to detect lower resolution
            if i > 0 and feat[i].size(2) < feat[i - 1].size(2):
                xx = xx / 2.0
                yy = yy / 2.0

            xx = np.clip(xx, 0, layer_feat.shape[2] - 1).astype(np.int32)
            yy = np.clip(yy, 0, layer_feat.shape[3] - 1).astype(np.int32)

            # 在每个feature map上随机取1000个点
            features = layer_feat[:, :, xx[range(samples)], yy[range(samples)]]
            feat_samples.append(features.clone().detach())

        # feat为选定的VGG网络层提取出来的所有的特征
        feat = torch.cat(feat_samples, 1)
        return feat