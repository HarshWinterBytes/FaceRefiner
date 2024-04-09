import os
import cv2
import math
import copy
import torch
import numpy as np
import menpo.io as mio
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat, savemat
from external.face3d import generate_uv_map
from utils.utils_misc import fill_UV, im_menpo2PIL, im_PIL2menpo
from menpo.shape.mesh.textured import TexturedTriMesh
from menpo3d.rasterize.cpu import rasterize_barycentric_coordinate_images
from menpo3d.rasterize.base import rasterize_mesh_from_barycentric_coordinate_images


class UV_Helper:
    def __init__(self, face_model='BFM'):
        self.index_exp = loadmat(os.path.join('external/deep3dfacerecon_pytorch/BFM', 'BFM_front_idx.mat'))
        self.index_exp = self.index_exp['idx'].astype(np.int32) - 1
        if face_model == 'BFM':
            self.uv_coords = loadmat(os.path.join('external/deep3dfacerecon_pytorch/BFM', 'BFM_UV.mat'))['UV']
            self.uv_coords = self.uv_coords[self.index_exp]
        elif face_model == 'multi_pie':
            self.uv_coords = mio.import_pickle(os.path.join('external/deep3dfacerecon_pytorch/BFM', 'tcoords_full.pkl'))
            self.uv_coords = self.uv_coords.points[self.index_exp]
        self.uv_coords = np.reshape(self.uv_coords, [-1, 2])
        
        # # uv-gan的UV纹理坐标
        # self.uv_gan_coords = mio.import_pickle(os.path.join('/media/chengxuan/4T/OSTeC', 'models/topology/tcoords_full.pkl'))
        # self.uv_gan_coords = self.uv_gan_coords.points[self.index_exp]
        # self.uv_gan_coords = np.reshape(self.uv_gan_coords, [-1, 2])

        self.uv_shape = [1024, 1024]
        # self.uv_shape = [595, 377]
        uv_mesh = self.uv_coords.copy()[:, ::-1]
        uv_mesh[:, 0] = 1 - uv_mesh[:, 0]
        uv_mesh *= self.uv_shape
        self.uv_mesh = np.concatenate([uv_mesh, uv_mesh[:, 0:1] * 0], 1)
    
    def check_dir(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)

    def save_inter_results(self, opt, img_name, coarse_tex, valid_tex, mask, parsing_mask, raw_mask, recon_dict):
        output_dir = os.path.join(opt.output_dir, opt.exp_name)
        coarse_tex_path = os.path.join(output_dir, 'coarse', img_name)
        valid_tex_path = os.path.join(output_dir, 'valid_tex', img_name)
        mask_path = os.path.join(output_dir, 'mask', img_name)
        angle_path = os.path.join(output_dir, 'angle', img_name.replace('png', 'mat'))
        # parsing_path = os.path.join(opt.save_dir, 'parsing', img_name)
        # raw_mask_path = os.path.join(opt.save_dir, 'raw_mask', img_name)
        self.check_dir(output_dir)
        self.check_dir(os.path.join(output_dir, 'coarse'))
        self.check_dir(os.path.join(output_dir, 'valid_tex'))
        self.check_dir(os.path.join(output_dir, 'mask'))
        self.check_dir(os.path.join(output_dir, 'angle'))
        # self.check_dir(os.path.join(opt.save_dir, 'parsing'))
        # self.check_dir(os.path.join(opt.save_dir, 'raw_mask'))

        cv2.imwrite(coarse_tex_path, cv2.cvtColor(coarse_tex, cv2.COLOR_BGR2RGB))
        cv2.imwrite(valid_tex_path, cv2.cvtColor(valid_tex, cv2.COLOR_BGR2RGB))
        cv2.imwrite(mask_path, mask)
        savemat(angle_path, {'angle': recon_dict['euler_angles']})
        # cv2.imwrite(parsing_path + '.png', parsing_mask)
        # cv2.imwrite(raw_mask_path + '.png', raw_mask)

    def render_uv_image(self, generated, tcoords, recon_dict):
        uv_tmesh = TexturedTriMesh(self.uv_mesh, tcoords, generated, trilist=recon_dict['trilist'])
        bcs = rasterize_barycentric_coordinate_images(uv_tmesh, self.uv_shape)
        img = rasterize_mesh_from_barycentric_coordinate_images(uv_tmesh, *bcs)
        img.pixels = np.clip(img.pixels, 0.0, 1.0)

        return img
    
    def get_coarse_tex(self, recon_dict, uv_h=512, uv_w=512):
        color = np.clip(255. * recon_dict['color'], 0, 255).astype(np.uint8)
        coarse_tex = generate_uv_map.get_uv_map(color, recon_dict['trilist'], copy.deepcopy(self.uv_coords), uv_h=uv_h, uv_w=uv_w)

        return coarse_tex

    def get_visibility(self, tmesh, threshold=0.4):#, pose_angle_deg=[0, 0, 0], cam_dist=-4.5):
        camera_direction = -tmesh.points / np.tile(np.linalg.norm(tmesh.points, axis=1), [3, 1]).T
        view_angle = np.sum(camera_direction * tmesh.vertex_normals(), 1)
        view_angle[view_angle < threshold] = 0

        return np.squeeze(view_angle)
    
    def tensor_rotate(self, img, degree):
        angle = -degree * math.pi / 180
        theta = torch.tensor([
            [math.cos(angle), math.sin(-angle), 0],
            [math.sin(angle), math.cos(angle), 0]
        ], dtype=torch.float)
        grid = F.affine_grid(theta.unsqueeze(0), img.shape).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        output = F.grid_sample(img, grid)

        return output


    def sample_color(self, tex_sampled, img_size, sample_type='nearest'):
        """
        在给定的UV纹理图像上根据UV坐标采样，采样方式有两种：临近采样和双线性采样
        """
        tex_sampled = self.tensor_rotate(tex_sampled.clone().contiguous(), 90)
        if sample_type == 'nearest':
            tex_xy = np.round(self.uv_coords * img_size).astype(np.int32) - 1
            face_color = tex_sampled[:, :, tex_xy[..., 0], tex_xy[..., 1]].permute(0, 2, 1).contiguous()
        elif sample_type == 'bilinear':
            # next 4 pixels
            tex_xy = self.uv_coords * img_size - 1
            ul = tex_sampled[:, :, np.floor(tex_xy[..., 0]).astype(np.int32), np.floor(tex_xy[..., 1]).astype(np.int32)].permute(0, 2, 1).contiguous()
            ur = tex_sampled[:, :, np.floor(tex_xy[..., 0]).astype(np.int32), np.ceil(tex_xy[..., 1]).astype(np.int32)].permute(0, 2, 1).contiguous()
            dl = tex_sampled[:, :, np.ceil(tex_xy[..., 0]).astype(np.int32), np.floor(tex_xy[..., 1]).astype(np.int32)].permute(0, 2, 1).contiguous()
            dr = tex_sampled[:, :, np.ceil(tex_xy[..., 0]).astype(np.int32), np.ceil(tex_xy[..., 1]).astype(np.int32)].permute(0, 2, 1).contiguous()

            yd = tex_xy[..., 0] - np.floor(tex_xy[..., 0])
            xd = tex_xy[..., 1] - np.floor(tex_xy[..., 1])
            face_color = ul * (1 - xd) * (1 - yd) + ur * xd * (1 - yd) + dl * (1 - xd) * yd + dr * xd * yd
        
        return face_color
    
    def data_preprocess(self, opt, im, recon_dict, img_name, parsing_mask=None):
        # 获得粗糙的纹理
        coarse_tex = self.get_coarse_tex(recon_dict, opt.img_size, opt.img_size)

        dense_lms = recon_dict['dense_lms'] / im.shape[::-1]
        dense_lms[:, 1] = 1 - dense_lms[:, 1]
        im_full = fill_UV(im_PIL2menpo(np.array(im_menpo2PIL(im))))

        # 完整的UV图像
        img_uv_src = self.render_uv_image(im_full, dense_lms, recon_dict)
        img_uv_src.pixels = img_uv_src.pixels[0:3]
        img_uv_src = fill_UV(img_uv_src)

        # 获得face_parsing mask在UV空间中的表示
        # if parsing_mask is not None:
        # parsing_mask = cv2.resize(parsing_mask, (1120, 1120))
        # parsing_mask = Image.fromarray(cv2.cvtColor(parsing_mask, cv2.COLOR_BGR2RGB))
        # parsing_mask = im_PIL2menpo(parsing_mask)
        # parsing_mask = self.render_uv_image(parsing_mask, dense_lms, recon_dict)
        # parsing_mask = cv2.cvtColor(np.asarray(im_menpo2PIL(parsing_mask)),cv2.COLOR_RGB2GRAY)

        # 计算人脸的角度，判断是否为侧脸
        _, yaw_angle, _ = recon_dict['euler_angles']
        is_profile = np.abs(yaw_angle* 180 / np.pi) > 25
        visibility_threshold = 0.4
        if is_profile:
            visibility_threshold = 0.6
        tmesh = TexturedTriMesh(recon_dict['vertices'], copy.deepcopy(self.uv_coords), img_uv_src,
                                trilist=recon_dict['trilist'])
        tmesh_rotated = TexturedTriMesh(recon_dict['vertices_rotated_mask'], tmesh.tcoords.points, tmesh.texture,
                                trilist=tmesh.trilist)

        # 计算每个顶点的可见性，根据相机视角和面的夹角来判断，从而得到人脸可见部分的mask
        vertices_vis_ind = self.get_visibility(tmesh_rotated, visibility_threshold)
        face_color = recon_dict['color']
        face_color[vertices_vis_ind == 0] = [0, 0, 0]
        face_color[vertices_vis_ind != 0] = [255, 255, 255]
        mask_img = generate_uv_map.get_uv_map(face_color, recon_dict['trilist'], copy.deepcopy(self.uv_coords))
        # mask_img = cv2.resize(mask_img, (1024, 1024))
        # parsing_mask = cv2.resize(parsing_mask, (1024, 1024))
        mask_img = cv2.resize(mask_img, (595, 377))
        # parsing_mask = cv2.resize(parsing_mask, (595, 377))

        raw_mask = mask_img

        # 合并人脸可见部分的mask和face_parsing的mask
        # mask_img = cv2.bitwise_and(mask_img, mask_img, mask=parsing_mask)

        # 形态学闭操作
        if np.abs(yaw_angle* 180 / np.pi) >= 15:
            kernel_size = (10, 10)
        else:
            kernel_size = (20, 20)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        closing = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
        _, mask = cv2.threshold(closing, 127, 255, cv2.THRESH_BINARY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # texture = cv2.cvtColor(np.array(im_menpo2PIL(img_uv_src)), cv2.COLOR_RGB2BGR)
        texture = np.array(im_menpo2PIL(img_uv_src))
        # texture = cv2.resize(texture, (1024, 1024))
        mask = cv2.resize(mask, (opt.img_size, opt.img_size))
        texture = cv2.resize(texture, (opt.img_size, opt.img_size))
        valid_tex = cv2.bitwise_and(texture, texture, mask=mask)
        if opt.save_inter_results:
            self.save_inter_results(opt, img_name, coarse_tex, valid_tex, mask, parsing_mask, raw_mask, recon_dict)

        results = {
            'coarse_tex': coarse_tex,
            'valid_tex': valid_tex,
            'mask': mask
        }

        return results