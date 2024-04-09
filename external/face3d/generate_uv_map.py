"""
Generate 2d uv maps representing different attributes(colors, depth, image position, etc)
: render attributes to uv space.
"""
import sys
import numpy as np

from . import mesh

sys.path.append('..')


def process_uv(uv_coords, uv_h=512, uv_w=512):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
    return uv_coords


def get_uv_map(colors, triangles, uv_coords, uv_h=377, uv_w=599):
    colors = colors / np.max(colors)
    # uv_h = uv_w = 512
    uv_coords = process_uv(uv_coords, uv_h, uv_w)

    # uv texture map
    uv_texture_map = mesh.render.render_colors(uv_coords, triangles, colors, uv_h, uv_w, c=3)
    uv_texture_map = (uv_texture_map * 255).astype(np.uint8)
    # io.imsave(name, np.squeeze(uv_texture_map))

    return uv_texture_map
