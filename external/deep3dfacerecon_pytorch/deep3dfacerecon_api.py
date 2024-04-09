"""This script is the test script for Deep3DFaceRecon_pytorch
"""

import os
from models import create_model
from util.preprocess import align_img
import numpy as np
from util.load_mats import load_lm3d
import torch 
from argparse import Namespace
import inspect

def read_data(im, lm, lm3d_std, to_tensor=True):
    # to RGB
    W,H = im.size
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    im_aligned = im
    _, im_hq, _, _ = align_img(im, lm, lm3d_std, target_size=5*224., rescale_factor=5*102.)
    if to_tensor:
        im = torch.tensor(np.array(im)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
        im_hq = torch.tensor(np.array(im_hq)/255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return im, lm, im_hq, im_aligned


class Deep3dModel:
    def __init__(self, device = 0):
        # opt = TestOptions().parse()  # get test options
        opt = Namespace()
        opt.model = 'facerecon'
        opt.isTrain = False
        opt.name = 'face_recon'
        opt.net_recon = 'resnet50'
        opt.use_last_fc = False
        opt.init_path = 'checkpoints/init_model/resnet50-0676ba61.pth'
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        opt.bfm_folder = '{}/BFM'.format(current_dir)
        opt.checkpoints_dir = '{}/checkpoints'.format(current_dir)
        opt.bfm_model = 'BFM_model_front.mat'
        opt.camera_d = 10.0
        opt.focal = 1015.0
        opt.center = 112.0
        opt.z_near = 5.0
        opt.z_far = 15.0
        opt.epoch = 'latest'
        opt.phase = 'test'
        opt.use_ddp = False

        device = torch.device(device)
        torch.cuda.set_device(device)
        self.model = create_model(opt)
        self.model.setup(opt)
        self.model.device = device
        self.model.parallelize()
        self.model.eval()
        self.lm3d_std = load_lm3d(opt.bfm_folder)

    def recontruct(self, im, lms):

        im_tensor, lm_tensor, im_hq_tensor, im_aligned = read_data(im, lms, self.lm3d_std)
        data = {
            'imgs': im_tensor,
            'lms': lm_tensor
        }
        self.model.set_input(data)  # unpack data from data loader
        with torch.no_grad():
            coeffs = self.model.net_recon(self.model.input_img)
            self.model.facemodel.to(self.model.device)

            coef_dict = self.model.facemodel.split_coeff(coeffs)
            face_shape = self.model.facemodel.compute_shape(coef_dict['id'], coef_dict['exp'])
            rotation = self.model.facemodel.compute_rotation(coef_dict['angle'])

            face_shape_transformed = self.model.facemodel.transform(face_shape, rotation, coef_dict['trans'])
            face_vertex = self.model.facemodel.to_camera(face_shape_transformed)

            face_proj = self.model.facemodel.to_image(face_vertex)
            landmark = self.model.facemodel.get_landmarks(face_proj)

            face_texture = self.model.facemodel.compute_texture(coef_dict['tex'])
            face_norm = self.model.facemodel.compute_norm(face_shape)
            face_norm_roted = face_norm @ rotation
            face_color = self.model.facemodel.compute_color(face_texture, face_norm_roted, coef_dict['gamma'])

            recon_dict = {}
            pred_coeffs = {key:coef_dict[key].cpu().numpy() for key in coef_dict}
            recon_dict['vertices'] = face_shape.cpu().numpy()[0]
            recon_dict['vertices_rotated'] = face_vertex.cpu().numpy()[0] #(self.pred_shape @ rotation + self.pred_coeffs_dict['trans'].unsqueeze(1)).cpu().numpy()[0]
            recon_dict['vertices_rotated'][..., -1] = recon_dict['vertices_rotated'][..., -1]
            recon_dict['vertices_rotated_mask'] = face_vertex.cpu().numpy()[0] #(self.pred_shape @ rotation + self.pred_coeffs_dict['trans'].unsqueeze(1)).cpu().numpy()[0]
            recon_dict['vertices_rotated_mask'][..., -1] = recon_dict['vertices_rotated_mask'][..., -1] * -1
            recon_dict['trilist'] = self.model.facemodel.face_buf.cpu().numpy()
            recon_dict['euler_angles'] = pred_coeffs['angle'][0]
            recon_dict['input'] = im_hq_tensor[0].cpu().numpy()
            recon_dict['dense_lms'] = face_proj.cpu().numpy()[0] * 5
            recon_dict['dense_lms'][:,1] = recon_dict['input'].shape[1] - 5 - recon_dict['dense_lms'][:,1]
            recon_dict['color'] = face_color.cpu().numpy()[0]
            recon_dict['im_aligned'] = im_aligned
            recon_dict['im_hq'] = im_hq_tensor
            recon_dict['landmark'] = landmark

        return recon_dict


