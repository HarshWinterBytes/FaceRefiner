import torch.nn.functional as F


def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode, align_corners=False)


def laplacian(x):  # x - upsample(downsample(x))
    return x - tensor_resample(tensor_resample(x, [max(x.size(2) // 2, 1), max(x.size(3) // 2, 1)]),
                               [x.size(2), x.size(3)])


def create_laplacian_pyramid(image, pyramid_depth):
    laplacian_pyramid = []
    current_image = image
    for i in range(pyramid_depth):
        laplacian_pyramid.append(laplacian(current_image))
        next_size = [max(current_image.size(2) // 2, 1), max(current_image.size(3) // 2, 1)]
        current_image = tensor_resample(current_image, next_size)
    laplacian_pyramid.append(current_image)

    return laplacian_pyramid


def synthetize_image_from_laplacian_pyramid(laplacian_pyramid, add_noise=False):
    current_image = laplacian_pyramid[-1]
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        up_x = laplacian_pyramid[i].size(2)
        up_y = laplacian_pyramid[i].size(3)
        current_image = laplacian_pyramid[i] + tensor_resample(current_image, (up_x, up_y))

    return current_image