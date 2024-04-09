import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import sys
import time
import cv2
import warnings
import menpo
sys.path.append('external/deep3dfacerecon_pytorch')
sys.path.append('external/face3d')
import numpy as np
from PIL import Image
from external.deep3dfacerecon_pytorch.deep3dfacerecon_api import Deep3dModel
from external.deep3dfacerecon_pytorch.util.nvdiffrast import MeshRenderer
from external.deep3dfacerecon_pytorch.models import networks
from external.deep3dfacerecon_pytorch.util import util
from utils.utils_uv import UV_Helper
from utils.utils_misc import im_menpo2PIL, pil_resize_to_tensor, pil_loader, convert_image, print_and_save_options
from style_transfer_3d import style_transfer_3D
from vgg_extractor import VGG16_Extractor


def main(args):
    source_dir = args.source_dir
    content_dir = args.content_dir
    style_dir = args.style_dir
    mask_dir = args.mask_dir
    device = args.device
    img_size = args.img_size

    fov_256 = 2 * np.arctan(args.center_256 / args.focal_256) * 180 / np.pi
    renderer_256 = MeshRenderer(
        rasterize_fov=fov_256, znear=args.z_near, zfar=args.z_far, rasterize_size=int(2 * args.center_256), use_opengl=True
    )
    fov_512 = 2 * np.arctan(args.center_512 / args.focal_512) * 180 / np.pi
    renderer_512 = MeshRenderer(
        rasterize_fov=fov_512, znear=args.z_near, zfar=args.z_far, rasterize_size=int(2 * args.center_512), use_opengl=True
    )
    renderers = [renderer_256, renderer_512]

    deep3dmodel = Deep3dModel()
    uv_helper = UV_Helper(args.face_model)
    vgg16_extractor = VGG16_Extractor().to(device).eval()
    net_recog = networks.define_net_recog(
                net_recog=args.net_recog, pretrained_path=args.net_recog_path
                ).to(device)
    full_mask_path = os.path.join('./full_masks', 'bfm_full_mask.png' if args.face_model == 'BFM' else 'multi_pie_mask.png')
    full_mask = cv2.resize(cv2.imread(full_mask_path), (img_size, img_size)).astype(np.uint8)
    full_mask[full_mask >= 1] = 1
    full_mask[full_mask < 1] = 0

    im_paths = [os.path.join(source_dir, i) for i in sorted(os.listdir(source_dir)) if i.endswith('png') or i.endswith('jpg')]
    lm_paths = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_paths]
    lm_paths = [os.path.join(i.replace(i.split(os.path.sep)[-1], ''), 'detections', i.split(os.path.sep)[-1]) for i in lm_paths]
    
    for i in range(len(im_paths)):
        img_name = im_paths[i].split(os.path.sep)[-1].replace('jpg', 'png')
        print(i, img_name)
        output_dir = os.path.join(args.output_dir, args.exp_name)
        if args.check_results_exist and os.path.exists(os.path.join(output_dir, str(args.style_transfer_num), img_name)):
            continue
        # img = Image.open(im_paths[i]).convert('RGB')
        img = Image.open(im_paths[i])
        lms = np.loadtxt(lm_paths[i]).astype(np.float32)
        lms = lms.reshape([-1, 2])
        recon_dict = deep3dmodel.recontruct(img, lms)

        if args.use_preprocess:
            print('begin to preprocess input image')
            img = menpo.image.Image(recon_dict['input'])
            inter_results = uv_helper.data_preprocess(args, img, recon_dict, img_name, parsing_mask=None)
            print('image preprocess finished!')

            content_pil = Image.fromarray(inter_results['coarse_tex']).convert('RGB')
            style_pil = Image.fromarray(inter_results['valid_tex']).convert('RGB')
            mask_pil = Image.fromarray(inter_results['mask']).convert('RGB')
        else:
            content_cv2 = uv_helper.get_coarse_tex(recon_dict, img_size, img_size)
            content_pil = Image.fromarray(content_cv2).convert('RGB')
            style_pil = pil_loader(os.path.join(style_dir, img_name)).convert('RGB')
            mask_pil = pil_loader(os.path.join(mask_dir, img_name)).convert('RGB')
        
        if not content_dir is None:
            if not os.path.exists(os.path.join(content_dir, img_name)):
                continue
            content_pil = pil_loader(os.path.join(content_dir, img_name)).convert('RGB')

        style_pil_flip = style_pil.transpose(Image.FLIP_LEFT_RIGHT)
        # raw_im_pil = Image.fromarray(convert_image(recon_dict['im_hq'][0]))
        raw_im_pil = im_menpo2PIL(img)
        content_im_orig = pil_resize_to_tensor(content_pil, img_size).to(device)
        style_im_orig = pil_resize_to_tensor(style_pil, img_size).to(device)
        mask_im_orig = pil_resize_to_tensor(mask_pil, img_size).to(device)
        raw_im_orig = pil_resize_to_tensor(raw_im_pil, img_size).to(device)
        style_flip_orig = pil_resize_to_tensor(style_pil_flip, img_size).to(device)
        datas = {
            'option': args,
            'content_im': content_im_orig,
            'style_im': style_im_orig,
            'style_im_flip': style_flip_orig,
            'mask_im': mask_im_orig,
            'full_mask': full_mask, 
            'raw_im': raw_im_orig,
            'recon_dict': recon_dict,
            'uv_helper': uv_helper,
            'device': device,
            'renderers': renderers,
            'vgg16_extractor': vgg16_extractor,
            'net_recog': net_recog,
            'img_name': img_name
        }
        start_time = time.time()
        style_transfer_3D(datas)
        # Report total time
        end_time = time.time()
        total_time = (end_time - start_time) / 60
        print('\nFinished after {:04.3f} minutes\n'.format(total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('High Fidelity 3D Face Texture with 3D Style Transfer')
    parser.add_argument('--source_dir', '-i', type=str, help='directory of input images', required=True)
    parser.add_argument('--style_dir', '-s', type=str, help='directory of style images')
    parser.add_argument('--content_dir', '-c', type=str, help='directory of content images')
    parser.add_argument('--mask_dir', '-m', type=str, help='directory of valid texture masks')
    parser.add_argument('--output_dir', '-o', type=str, help='directory of stylized images', required=True)
    parser.add_argument('--inter_save_dir', type=str, help='directory of Intermediate result(content img: coarse tex, style img: valid_tex, mask)')
    parser.add_argument('--save_inter_results', action='store_true', help='do not save intermidiate results')
    parser.add_argument('--use_preprocess', action='store_false', help='use data preprocess')
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=512, help='we process all of images with resolution of 512 * 512')
    parser.add_argument('--max_scales', type=int, default=2)
    parser.add_argument('--max_iter', type=int, default=150)
    parser.add_argument('--checkpoint_iter', type=int, default=50)
    parser.add_argument('--content_weight', type=float, default=8.0)
    parser.add_argument('--style_weight', type=float, default=1.7)
    parser.add_argument('--recon_weight', type=float, default=20.0)
    parser.add_argument('--learning_rate', type=float, default=0.3)
    parser.add_argument('--check_results_exist', action='store_true', help='check if results exsit in the output directory')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--face_model', type=str, default='BFM', help='type of uv coords. [BFM, multi_pie]')
    parser.add_argument('--style_transfer_num', type=int, default=5)
    parser.add_argument('--add_noise', action='store_false', help='if add noise on the feature maps extracted from content image.')

    # renderer parameters
    parser.add_argument('--focal_256', type=float, default=1160.)
    parser.add_argument('--focal_512', type=float, default=2320.)
    parser.add_argument('--center_256', type=float, default=128.)
    parser.add_argument('--center_512', type=float, default=256.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    parser.add_argument('--net_recog', type=str, default='r50', choices=['r18', 'r43', 'r50'],
                                help='face recog network structure')
    parser.add_argument('--net_recog_path', type=str,
                                default='external/deep3dfacerecon_pytorch/checkpoints/recog_model/ms1mv3_arcface_r50_fp16/backbone.pth')
    parser.add_argument('--use_predef_M', type=util.str2bool, nargs='?', const=True, default=False,
                                help='use predefined M for predicted face')
    
    parser.add_argument('--use_style_mask', action='store_true', help='if use the mask to guide transfer of style image')
    parser.add_argument('--use_content_mask', action='store_true', help='if use the mask to guide transfer of content image')

    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    # print and save options
    print_and_save_options(args)

    main(args)