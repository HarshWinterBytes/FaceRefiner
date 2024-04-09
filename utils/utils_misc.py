import math
import cv2
import os
import PIL
import numpy as np
from PIL import Image
from scipy.interpolate import NearestNDInterpolator
import torch
import random
import menpo

def tensor_erode(img_tensor):
    kernel_size = img_tensor.shape[1] // 20
    img = convert_image(img_tensor.repeat(3, 1, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    eroding = cv2.morphologyEx(closing, cv2.MORPH_ERODE, kernel)
    _, mask = cv2.threshold(eroding, 127, 255, cv2.THRESH_BINARY)
    mask_pil = Image.fromarray(mask)
    mask_tensor = pil_to_tensor(mask_pil)[:, 0, :, :].unsqueeze(0)

    return mask_tensor
    

def compute_rotation(angles, device):
    """
    给定x, y, z三个轴的旋转量，计算旋转矩阵
    这个函数用于计算每张人脸在三维空间需要调整的角度
    """
    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(device)
    zeros = torch.zeros([batch_size, 1]).to(device)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

    # x轴的旋转矩阵
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])

    # y轴的旋转矩阵
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    # z轴的旋转矩阵
    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    # 将三个坐标轴的旋转矩阵线性组合为最终的旋转矩阵
    rot = rot_z @ rot_y @ rot_x

    return rot.permute(0, 2, 1)


def get_opp_pose(face_vertex, angle, camera_d, device):
    """
    获得与原来人脸相反的姿势的人脸
    """
    face_vertex[..., -1] = camera_d - face_vertex[..., -1]
    rotation = compute_rotation(angle, device)
    rotation_inv = torch.inverse(rotation)
    z_mean = torch.mean(face_vertex[..., 2])
    face_vertex[..., 2] = face_vertex[..., 2] - z_mean
    face_vertex = face_vertex @ rotation_inv
    angle = angle * -1
    rotation = compute_rotation(angle, device)
    face_vertex = face_vertex @ rotation
    face_vertex[..., 2] = face_vertex[..., 2] + z_mean
    face_vertex[..., -1] = camera_d - face_vertex[..., -1]

    return face_vertex


def fill_UV(UV):
    mask = np.sum(UV.pixels, 0) == 0
    xx, yy = np.meshgrid(np.arange(UV.shape[1]), np.arange(UV.shape[0]))
    xym = np.vstack((np.ravel(xx[~mask]), np.ravel(yy[~mask]))).T
    data = UV.pixels[:, ~mask]
    for i in range(3):
        interp = NearestNDInterpolator(xym, data[i])
        result = interp(np.ravel(xx[mask]), np.ravel(yy[mask]))
        UV.pixels[i, mask] = result
    return UV


def im_menpo2PIL(menpo_im):
    return PIL.Image.fromarray((menpo_im.pixels_with_channels_at_back() * 255).astype(np.uint8))


def im_PIL2menpo(pil_im):
    return menpo.image.Image.init_from_channels_at_back(np.array(pil_im).astype(np.float) / 255)


def pil_loader(path, flag=False):
    with open(path, 'rb') as f:
        img = Image.open(f)
        if flag:
            img = fill_UV(np.array(img))
            img = Image.fromarray(img)
        return img.convert('RGB')


def pil_resize_to_tensor(pil, trg_size):
    resized = pil.resize((trg_size, trg_size), Image.BICUBIC)
    return pil_to_tensor(resized)


def pil_to_tensor(pil):
    return (torch.Tensor(np.array(pil).astype(np.float) / 127.5) - 1.0).permute((2, 0, 1)).unsqueeze(0)


################################################################################
# Feature sampling functions https://github.com/futscdav/strotss
################################################################################
def convert_image(x):
    x_out = np.clip(x.permute(1, 2, 0).detach().cpu().numpy(), -1.0, 1.0)
    x_out -= x_out.min()
    x_out /= x_out.max()
    x_out = (x_out * 255).astype(np.uint8)
    return x_out


def sample_indices(feat_content):
    const = 128 ** 2  # 32k or so
    big_size = feat_content.shape[2] * feat_content.shape[3]

    # //为整数除法
    stride_x = int(max(math.floor(math.sqrt(big_size // const)), 1))
    offset_x = np.random.randint(stride_x)
    stride_y = int(max(math.ceil(math.sqrt(big_size // const)), 1))
    offset_y = np.random.randint(stride_y)

    # [offset_x::stride_x] 表示从offset_x开始，以步长为stride_x取值
    xx, xy = np.meshgrid(np.arange(feat_content.shape[2])[offset_x::stride_x],
                         np.arange(feat_content.shape[3])[offset_y::stride_y])

    xx = xx.flatten()
    xy = xy.flatten()
    return xx, xy


def sample_indices_by_mask(mask):
    g = np.array(np.argwhere(mask[0][0] > 0))
    samples = g.shape[1]
    indices = random.sample(range(0, g.shape[1]), samples)
    # xx = g[indices, 0]
    # yy = g[indices, 1]
    xx = g[0, indices]
    yy = g[1, indices]
    return xx, yy


def spatial_feature_extract(feat_result, feat_content, xx, xy):
    l2, l3 = [], []
    device = feat_result[0].device

    # Loop over each extracted layer
    for i in range(len(feat_result)):
        fr = feat_result[i]
        fc = feat_content[i]

        # Hack to detect reduced scale
        if i > 0 and feat_result[i - 1].size(2) > feat_result[i].size(2):
            xx = xx / 2.0
            xy = xy / 2.0

        # Go back to ints and get residual
        xxm = np.floor(xx).astype(np.float32)
        xxr = xx - xxm

        xym = np.floor(xy).astype(np.float32)
        xyr = xy - xym

        # Do bilinear resampling
        w00 = torch.from_numpy((1. - xxr) * (1. - xyr)).float().view(1, 1, -1, 1).to(device)
        w01 = torch.from_numpy((1. - xxr) * xyr).float().view(1, 1, -1, 1).to(device)
        w10 = torch.from_numpy(xxr * (1. - xyr)).float().view(1, 1, -1, 1).to(device)
        w11 = torch.from_numpy(xxr * xyr).float().view(1, 1, -1, 1).to(device)

        xxm = np.clip(xxm.astype(np.int32), 0, fr.size(2) - 1)
        xym = np.clip(xym.astype(np.int32), 0, fr.size(3) - 1)

        s00 = xxm * fr.size(3) + xym
        s01 = xxm * fr.size(3) + np.clip(xym + 1, 0, fr.size(3) - 1)
        s10 = np.clip(xxm + 1, 0, fr.size(2) - 1) * fr.size(3) + xym
        s11 = np.clip(xxm + 1, 0, fr.size(2) - 1) * fr.size(3) + np.clip(xym + 1, 0, fr.size(3) - 1)

        fr = fr.view(1, fr.size(1), fr.size(2) * fr.size(3), 1)
        fr = fr[:, :, s00, :].mul_(w00).add_(fr[:, :, s01, :].mul_(w01)).add_(fr[:, :, s10, :].mul_(w10)).add_(
            fr[:, :, s11, :].mul_(w11))

        fc = fc.view(1, fc.size(1), fc.size(2) * fc.size(3), 1)
        fc = fc[:, :, s00, :].mul_(w00).add_(fc[:, :, s01, :].mul_(w01)).add_(fc[:, :, s10, :].mul_(w10)).add_(
            fc[:, :, s11, :].mul_(w11))

        l2.append(fr)
        l3.append(fc)

    x_st = torch.cat([li.contiguous() for li in l2], 1)
    c_st = torch.cat([li.contiguous() for li in l3], 1)

    return x_st, c_st


def print_and_save_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = opt.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.output_dir, opt.exp_name)
    mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'options.txt')
    try:
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    except PermissionError as error:
        print("permission error {}".format(error))
        pass


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
    

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)