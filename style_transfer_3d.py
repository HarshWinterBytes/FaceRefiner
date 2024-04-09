import torch
import os
import cv2
import torch.nn.functional as F
import numpy as np
from utils.utils_misc import convert_image, sample_indices, sample_indices_by_mask, spatial_feature_extract, get_opp_pose, tensor_erode, fill_UV
from utils import utils_misc
from PIL import Image
from utils.utils_pyramid import create_laplacian_pyramid, synthetize_image_from_laplacian_pyramid, laplacian
from loss import content_loss, remd_loss, moment_loss, perceptual_loss, content_loss_new


def style_transfer_3D(datas):
    args = datas['option']
    recon_dict = datas['recon_dict']
    device = datas['device']
    renderers = datas['renderers']
    content_im = datas['content_im']
    style_im = datas['style_im']
    style_im_flip = datas['style_im_flip']
    mask_im = (datas['mask_im'] + 1) / 2
    full_mask = datas['full_mask']
    raw_im = datas['raw_im']
    vgg16_extractor = datas['vgg16_extractor']
    net_recog = datas['net_recog']
    uv_helper = datas['uv_helper']
    img_name = datas['img_name']
    max_scales = args.max_scales
    max_iteration = args.max_iter
    checkpoint_iter = args.checkpoint_iter
    content_weight = args.content_weight
    style_weight = args.style_weight
    recon_weight = args.recon_weight
    img_size = args.img_size
    output_dir = os.path.join(args.output_dir, args.exp_name)
    camera_d = args.camera_d

    face_buf = recon_dict['trilist']
    face_buf = torch.tensor(face_buf).to(device)
    face_vertex = recon_dict['vertices_rotated']
    face_vertex = torch.tensor(face_vertex).to(device).unsqueeze(0)
    angle = torch.tensor(recon_dict['euler_angles']).to(device).unsqueeze(0)
    face_vertex_opp = get_opp_pose(face_vertex.clone(), angle, camera_d, device)
    face_poses = [face_vertex, face_vertex_opp]

    new_content_im = None
    learning_rate = args.learning_rate
    for transfer_index in range(args.style_transfer_num):
        if new_content_im is not None:
            content_im = new_content_im
    
        # 初始化待优化的参数
        # temp_img_pil = Image.fromarray(convert_image(content_im.clone()[0]))
        # temp_img_pil = Image.fromarray(fill_UV(np.array(temp_img_pil)))
        # full_content_im = utils_misc.pil_resize_to_tensor(temp_img_pil, args.img_size).to(device)
        laplacian_pyramid = create_laplacian_pyramid(content_im, pyramid_depth=5)
        laplacian_pyramid = [torch.nn.Parameter(li.data, requires_grad=True) for li in laplacian_pyramid]
        # Define parameters to be optimized
        parameters = [{'params': si} for si in laplacian_pyramid]
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9)
        # if transfer_index >= 1:
        #     learning_rate = learning_rate / 10

        for scale in range(max_scales):
            down_fac = 2 ** (max_scales - 1 - scale)
            begin_ind = (max_scales - 1 - scale)
            content_weight_scaled = content_weight * max(1, down_fac)
            style_weight_scaled = style_weight * max(1, down_fac)
            recon_weight_scaled = recon_weight * max(1, down_fac)
            if down_fac > 1:
                renderer = renderers[0]
            else:
                renderer = renderers[1]
            cur_size = (img_size // down_fac, img_size // down_fac)
            content_im_scaled = F.interpolate(content_im, cur_size, mode='bilinear')
            style_im_scaled = F.interpolate(style_im, cur_size, mode='bilinear')
            mask_im_scaled = F.interpolate(mask_im, cur_size, mode='bilinear')
            full_mask_scaled = torch.tensor(cv2.resize(full_mask, cur_size)).permute((2, 0, 1)).unsqueeze(0).to(device)
            raw_im_scaled = F.interpolate(raw_im, cur_size, mode='bilinear')
            style_im_flip_scaled = F.interpolate(style_im_flip, cur_size, mode='bilinear')
            styles_scaled = [style_im_scaled, style_im_flip_scaled]

            with torch.no_grad():
                # 为content image添加噪声
                # batch, _, height, width = content_im_scaled.shape
                # noise = content_im_scaled.new_empty(batch, 1, height, width).normal_()
                # noise = noise * (full_mask_scaled - mask_im_scaled)
                # temp_img = content_im_scaled + noise * (full_mask_scaled - mask_im_scaled)
                # test_img = convert_image((noise).clone()[0])
                # Image.fromarray(test_img).show()
                # input('wait')
                feat_content = vgg16_extractor(content_im_scaled, args.add_noise)
                feat_style = None
                # 可见纹理在图像空间特征
                # feat_style_im = None
                # face_color = uv_helper.sample_color(style_im_scaled, cur_size).to(device)
                # pred_mask, _, pred_face = renderer.forward(face_vertex, face_buf, face_color)
                # style_valid_im = pred_mask * pred_face
                for i in range(5):
                    if not args.use_style_mask:
                        feat_e = vgg16_extractor.forward_samples_hypercolumn(styles_scaled[(i + 1) % 2] , samps=1000)
                    else:
                        feat_e = vgg16_extractor.forward_samples_hypercolumn_by_mask(mask_im_scaled.cpu(), style_im_scaled , samps=1000)
                    # feat_e = vgg16_extractor.forward_samples_hypercolumn(style_im_scaled, samps=1000, add_noise=True)
                    feat_style = feat_e if feat_style is None else torch.cat((feat_style, feat_e), dim=2)

                    # feat_e_im = vgg16_extractor.forward_samples_hypercolumn(style_valid_im, samps=1000)
                    # feat_style_im = feat_e_im if feat_style_im is None else torch.cat((feat_style_im, feat_e_im), dim=2)
                feat_max = 3 + 2 * 64 + 2 * 128 + 3 * 256 + 2 * 512  # 2179 = sum of all extracted channels
                # feat_style -> spatial_style: [1, 2179, 5000] -> [1, 2179, 5000, 1]
                spatial_style = feat_style.view(1, feat_max, -1, 1)
                # spatial_style_im = feat_style_im.view(1, feat_max, -1, 1)

                # xx: [16384, ], yy: [16384, ], 按照这个坐标提取出来的特征用于计算content loss
                if not args.use_content_mask:
                    xx, xy = sample_indices(feat_content[0])
                else:
                    xx, xy = sample_indices_by_mask(full_mask_scaled.cpu())
                # print(xx, xy)
                # input('wait')
                # xx1: [16384, ], yy1: [16384, ], 按照这个坐标提取出来的特征用于计算style loss，主要用来在可见区域提取输出图像特征
                # xx1, xy1 = sample_indices_by_mask(full_mask_scaled[..., 0])
                # xx2, xy2 = sample_indices_by_mask(np.array(mask_im_scaled.cpu())[0][0])
            
            # begin optimization for this scale
            print('begin to optimize stylized image at {} * {}'.format(cur_size[0], cur_size[1]))
            for i in range(max_iteration):
                
                optimizer.zero_grad()
                stylized_im = synthetize_image_from_laplacian_pyramid(laplacian_pyramid[begin_ind:])

                feat_stylized = vgg16_extractor(stylized_im)

                # Sample features to calculate losses with
                n = 2048
                np.random.shuffle(xx)
                np.random.shuffle(xy)
                # print(np.max(xx[:n]), np.min(xx[:n]), np.max(xy[:n]), np.min(xy[:n]))
                # input('wait')
                spatial_stylized, spatial_content = spatial_feature_extract(feat_stylized, feat_content, xx[:n], xy[:n])

                # compute some loss in image space for all poses
                face_color = uv_helper.sample_color(stylized_im, cur_size).to(device)
                # loss_perceptual = torch.tensor(0.).to(device)
                # l1_loss = torch.tensor(0.).to(device)
                # loss_style_im = torch.tensor(0.).to(device)
                # 将UV纹理通过可微渲染器渲染到图像空间
                pred_mask, _, pred_face = renderer.forward(face_vertex, face_buf, face_color)
                pred_mask = (tensor_erode(pred_mask[0]).to(device) + 1) / 2
                l1_loss = torch.mean(torch.abs(raw_im_scaled - pred_face) * pred_mask) * recon_weight_scaled
                # for j in range(len(face_poses)):
                #     face_pose = face_poses[j]
                    
                #     # # 计算perceptual loss
                #     # # TODO 在杨蛟龙的论文中，获取身份信息id的函数中有一个trans_m，这里先同一设置为None，后面要更改
                #     # trans_m = None
                #     # pred_feat = net_recog(pred_face * pred_mask, trans_m)
                #     # gt_feat = net_recog(raw_im_scaled, trans_m)
                #     # loss_perceptual += perceptual_loss(pred_feat, gt_feat)
                #     # compute l1 loss in image space for raw pose
                #     if j == 0:
                        
                    
                    # 计算图像空间上的风格损失
                    # feat_face_im = vgg16_extractor(pred_face)
                    
                    # style_color = uv_helper.sample_color(styles_scaled[j], cur_size).to(device)
                    # pred_mask, _, pred_face = renderer.forward(face_pose, face_buf, style_color)
                    # # test_img = convert_image((pred_face).clone()[0])
                    # # Image.fromarray(test_img).show()
                    # # input('wait')
                    # feat_style_im = vgg16_extractor(pred_face)

                    # spatial_face, spatial_face_style = spatial_feature_extract(feat_face_im, feat_style_im, xx[:n], xy[:n])
                    # loss_remd, _ = remd_loss(spatial_face, spatial_face_style, cos_d=True)
                    # loss_moment = moment_loss(spatial_face, spatial_face_style, moments=[1, 2])
                    # loss_color, _ = remd_loss(spatial_face[:, :3, :, :], spatial_face_style[:, :3, :, :], cos_d=False)
                    # loss_style_im += (loss_remd + loss_moment + (1. / max(content_weight_scaled, 1.)) * loss_color)
                # input('wait')

                # compute L1 loss in UV space
                # l1_loss = torch.mean(torch.abs(stylized_im - style_im_scaled) * mask_im_scaled) * recon_weight_scaled
                # compute content loss in UV space
                loss_content = content_loss(spatial_stylized, spatial_content) * content_weight_scaled
                # loss_content = content_loss_new(feat_content, feat_stylized) * content_weight_scaled
                # compute style loss in UV space
                loss_remd, _ = remd_loss(spatial_stylized, spatial_style, cos_d=True)
                loss_moment = moment_loss(spatial_stylized, spatial_style, moments=[1, 2])
                loss_color, _ = remd_loss(spatial_stylized[:, :3, :, :], spatial_style[:, :3, :, :], cos_d=False)
                loss_style = (loss_remd + loss_moment + (1. / max(content_weight_scaled, 1.)) * loss_color) * style_weight_scaled
                # the total loss
                all_loss = loss_content + loss_style + l1_loss

                if i == 0 or (i + 1) % checkpoint_iter == 0:
                    str_epoch = 'epoch:[{} / {}] '.format(transfer_index + 1, args.style_transfer_num)
                    str_scale = 'scale:[{} / {}] '.format(scale + 1, max_scales)
                    str_iter = 'iteration:[{} / {}] '.format(i if i == 0 else i + 1, max_iteration)
                    info_1 = str_epoch + str_scale + str_iter
                    print(info_1 + 'content loss: {:04.3f}, style_loss: {:04.3f}, l1_loss: {:04.3f}'.format(loss_content, loss_style, l1_loss))
                    # save_im = convert_image(stylized_im.clone()[0])
                    # save_im = Image.fromarray(save_im)
                    # if not os.path.exists(output_dir):
                    #     os.makedirs(output_dir)
                    # save_path = os.path.join(output_dir, img_name.split('.')[0] + '{}_{}.png'.format(scale, i))
                    # save_im.save(save_path)

                # # 将UV纹理通过可微渲染器渲染到图像空间
                # face_color = uv_helper.sample_color(stylized_im, cur_size).to(device)
                # pred_mask, _, pred_face = renderer.forward(face_vertex, face_buf, face_color)
                # test_img = convert_image((pred_face * pred_mask).clone()[0])
                # Image.fromarray(test_img).show()
                # input('wait')
                
                all_loss.backward()
                optimizer.step()
        
        # 保存结果图像
        save_im = convert_image(stylized_im.clone()[0])
        save_im = Image.fromarray(save_im * full_mask)
        if not os.path.exists(os.path.join(output_dir, str(transfer_index + 1))):
            os.makedirs(os.path.join(output_dir, str(transfer_index + 1)))
        save_path = os.path.join(output_dir, str(transfer_index + 1), img_name)
        save_im.save(save_path)
        print('Save the stylized image {} to {}'.format(img_name, save_path))

        new_content_im = utils_misc.pil_loader(save_path)
        new_content_im = utils_misc.pil_resize_to_tensor(new_content_im, args.img_size).to(device)

        content_weight = content_weight / 1.1
        recon_weight = recon_weight * 1.1
