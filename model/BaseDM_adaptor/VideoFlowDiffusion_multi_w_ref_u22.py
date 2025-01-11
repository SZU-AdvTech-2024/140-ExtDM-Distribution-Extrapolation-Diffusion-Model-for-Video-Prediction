# use diffusion model to generate pseudo ground truth flow volume based on RegionMM
# 3D noise to 3D flow
# flow size: 2*32*32*40
# some codes based on https://github.com/lucidrains/video-diffusion-pytorch

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from model.LFAE.generator import Generator
from model.LFAE.bg_motion_predictor import BGMotionPredictor
from model.LFAE.region_predictor import RegionPredictor
from torchvision import models
from model.BaseDM_adaptor.DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22 import Unet3D
from model.BaseDM_adaptor.Diffusion import GaussianDiffusion
class Vgg19(torch.nn.Module):
    """
    Vgg19 network for perceptual loss.
    """

    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights='DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        # self.mean = torch.nn.Parameter(data=torch.Tensor(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))),
        #                                requires_grad=False)
        # self.std = torch.nn.Parameter(data=torch.Tensor(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))),
        #                               requires_grad=False)
        self.mean = None
        self.std = None
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, x):
        if x.ndim ==5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        # 动态初始化 mean 和 std
        if self.mean is None or self.std is None:
            C = x.shape[1]  # 获取输入的通道数
            self.mean = torch.nn.Parameter(torch.zeros(1, C, 1, 1), requires_grad=False).to(x.device())
            self.std = torch.nn.Parameter(torch.ones(1, C, 1, 1), requires_grad=False).to(x.device())

        # 归一化输入
        x = (x - self.mean) / self.std
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """

    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        # if self.scale == 1.0:
        #     return input
        # B, C, T, H, W = input.shape
        #
        # # 展平时间维度到批量维度
        # input = input.reshape(B * T, C, H, W)  # [80, 3, 68, 68]
        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out
class ImagePyramide(torch.nn.Module):
    """
    Create image pyramide for computing pyramide perceptual loss.
    """

    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        if x.ndim == 5:  # 处理 5D 输入
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict
class FlowDiffusion(nn.Module):
    def __init__(self,
                 config="",
                 pretrained_pth="",
                 is_train=True,
                 ddim_sampling_eta=1.,
                 timesteps=1000,
                 dim_mults=(1, 2, 4, 4),
                 learn_null_cond=False,
                 use_deconv=True,
                 padding_mode="zeros",
                 withFea=True,
                 Unet3D_architecture="DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada_u22",
                 device_ids=['cuda:0','cuda:1','cuda:2']
                 ):
        super(FlowDiffusion, self).__init__()
        self.device_ids = device_ids
        flow_params = config['flow_params']['model_params']
        diffusion_params = config['diffusion_params']['model_params']
        dataset_params = config['dataset_params']

        self.estimate_occlusion_map = flow_params["generator_params"]["pixelwise_flow_predictor_params"][
            "estimate_occlusion_map"]

        self.use_residual_flow = diffusion_params['use_residual_flow']
        self.only_use_flow = diffusion_params['only_use_flow']
        self.withFea = withFea
        if pretrained_pth != "":
            checkpoint = torch.load(pretrained_pth)
        print(flow_params['generator_params'])
        self.generator = Generator(num_regions=flow_params['num_regions'],
                                   num_channels=flow_params['num_channels'],
                                   revert_axis_swap=flow_params['revert_axis_swap'],
                                   **flow_params['generator_params']
                                   ).to(self.device_ids[0])
        if pretrained_pth != "":
            self.generator.load_state_dict(checkpoint['generator'], strict=False)
            self.generator.eval()
            self.set_requires_grad(self.generator, False)

        self.region_predictor = RegionPredictor(num_regions=flow_params['num_regions'],
                                                num_channels=flow_params['num_channels'],
                                                estimate_affine=flow_params['estimate_affine'],
                                                **flow_params['region_predictor_params']).to(self.device_ids[0])

        if pretrained_pth != "":
            self.region_predictor.load_state_dict(checkpoint['region_predictor'])
            self.region_predictor.eval()
            self.set_requires_grad(self.region_predictor, False)

        self.bg_predictor = BGMotionPredictor(num_channels=flow_params['num_channels'],
                                              **flow_params['bg_predictor_params']).to(self.device_ids[0])
        if pretrained_pth != "":
            self.bg_predictor.load_state_dict(checkpoint['bg_predictor'])
            self.bg_predictor.eval()
            self.set_requires_grad(self.bg_predictor, False)

        self.unet = Unet3D(
            dim=64,
            channels=3 + 256,
            out_grid_dim=2,
            out_conf_dim=1,
            dim_mults=dim_mults,
            use_bert_text_cond=False,
            learn_null_cond=learn_null_cond,
            use_final_activation=False,
            use_deconv=use_deconv,
            padding_mode=padding_mode,
            cond_num=dataset_params['train_params']['cond_frames'],
            pred_num=dataset_params['train_params']['pred_frames'],
            framesize=int(dataset_params['frame_shape'] * flow_params['region_predictor_params']['scale_factor']),
        ).to(self.device_ids[1])

        self.diffusion = GaussianDiffusion(
            self.unet,
            image_size=dataset_params['frame_shape'] // 2,
            num_frames=dataset_params['train_params']['cond_frames'] + dataset_params['train_params']['pred_frames'],
            sampling_timesteps=diffusion_params['sampling_timesteps'],
            timesteps=timesteps,  # number of steps
            loss_type=diffusion_params['loss_type'],  # L1 or L2
            use_dynamic_thres=True,
            null_cond_prob=diffusion_params['null_cond_prob'],
            ddim_sampling_eta=ddim_sampling_eta,
        ).to(self.device_ids[1])

        self.cond_frame_num = dataset_params['train_params']['cond_frames']
        self.pred_frame_num = dataset_params['train_params']['pred_frames']
        self.frame_num = self.cond_frame_num + self.pred_frame_num

        # training
        self.is_train = is_train
        if self.is_train:
            self.unet.train()
            self.diffusion.train()
        # self.scales = [1, 0.5, 0.25]
        # self.pyramid = ImagePyramide(self.scales, flow_params['num_channels']).to(self.device_ids[0])
        # self.vgg = Vgg19().to(device_ids[0])
    def forward(self, real_vid):
        # real_vid [bs, c, length(cond+pred), h, w]

        # compute pseudo ground-truth flow
        #nf 总帧数
        b, _, nf, H, W = real_vid.size()
        ret = {}
        # loss = torch.tensor(0.0).cuda()
        #生成的光流信息
        real_grid_list = []
        #生成的遮挡地图
        real_conf_list = []
        # 生成的预测帧
        real_out_img_list = []
        #使用光流变形后的参考帧图像
        real_warped_img_list = []

        # 停止梯度计算，减少内存消耗和加速推理过程
        with torch.no_grad():

            #提取参考帧  及计算初始区域的参数
            ref_img = real_vid[:, :, self.cond_frame_num - 1, :, :]
            #参考帧的运动信息
            source_region_params = self.region_predictor(ref_img.to(self.device_ids[0]))

            #遍历帧序列
            for idx in range(nf):
                # print(f"idx=={idx}")

                # print(real_vid[:, :, idx, :, :])
                #每一帧作为驱动帧
                driving_region_params = self.region_predictor(real_vid[:, :, idx, :, :].to(self.device_ids[0]))
                #参考帧和每一帧
                bg_params = self.bg_predictor(ref_img.to(self.device_ids[0]), real_vid[:, :, idx, :, :].to(self.device_ids[0]))

                # print('front_update driving_region_params', driving_region_params)

                # determinant = torch.det(driving_region_params['affine'])
                # # if (determinant == 0).any():
                #     print("this the data errror ")

                #从参考帧 生成目标帧

                generated = self.generator(ref_img.to(self.device_ids[0]),
                                           driving_region_params,
                                           source_region_params,
                                           bg_params)
                generated.update({'source_region_params': source_region_params,
                                  'driving_region_params': driving_region_params})

                #提取光流结果，并调整张量维度顺序（NHwC-> NCHW)
                real_grid_list.append(generated["optical_flow"].permute(0, 3, 1, 2).to(self.device_ids[0]))

                # normalized occlusion map
                real_conf_list.append(generated["occlusion_map"])
                real_out_img_list.append(generated["prediction"])
                real_warped_img_list.append(generated["deformed"])

            ref_img_fea = []
            #遍历每一个条件帧 进行多层下采样得到的深层特征
            for idx in range(self.cond_frame_num - 1):
                ref_img_fea.append(self.generator.forward_bottle(real_vid[:, :, idx, :, :]).detach().to(self.device_ids[0]))
            #添加条件帧的最后一帧的瓶颈特征
            for _ in range(1 + self.pred_frame_num):
                ref_img_fea.append(generated["bottle_neck_feat"].detach().to(self.device_ids[0]))
            ref_img_fea = torch.stack(ref_img_fea, dim=2)
            # ref_img_fea = rearrange(ref_img_fea, 'n c t h w->(n t) c h w')
            # ref_img_fea = F.interpolate(ref_img_fea, size=generated["optical_flow"].shape[-3:-1], mode='bilinear')
            # ref_img_fea = rearrange(ref_img_fea, '(n t) c h w->n c t h w',t=self.cond_frame_num+self.pred_frame_num)

        if self.is_train:
            # cond_frames pred frames
            pred_frames = real_vid[:, :, self.cond_frame_num: self.cond_frame_num + self.pred_frame_num]

        del real_vid
        torch.cuda.empty_cache()

        #将之前逐帧处理累积的结果沿时间维度的堆叠，形成完整的视频张量
        real_vid_grid = torch.stack(real_grid_list, dim=2) #BCHW->NCTHW
        real_vid_conf = torch.stack(real_conf_list, dim=2)
        real_out_vid = torch.stack(real_out_img_list, dim=2)
        real_warped_vid = torch.stack(real_warped_img_list, dim=2)
        # if self.withFea:
        #     ref_img_fea = F.interpolate(ref_feas, size=real_vid_conf.shape[-2:], mode='bilinear')
        # else:
        #     ref_img_fea = None

        # reference images are the same for different time steps, just pick the final one
        ret['real_vid_grid'] = real_vid_grid
        ret['real_vid_conf'] = real_vid_conf
        ret['real_out_vid'] = real_out_vid
        ret['real_warped_vid'] = real_warped_vid

        if self.is_train:
            if self.use_residual_flow:
                h, w = real_vid_grid.shape[-2:]
                identity_grid = self.get_grid(b, 1, h, w, normalize=True).cuda()
                frames = torch.cat((real_vid_grid - identity_grid, real_vid_conf * 2 - 1), dim=1)
            else:
                #将光流信息和遮挡地图在dim=1进行拼接  遮挡地图从[0,1]映射到[-1,1]
                frames = torch.cat((real_vid_grid, real_vid_conf * 2 - 1), dim=1)
            # print(ref_img_fea.shape)
            #duffusuion  条件帧 1：cond-1， 预测帧 cond-cond+pred 两者并没有重叠
            # print(f"frame_size()={frames.size()}")
            loss, pred = self.diffusion(frames[:, :, :self.cond_frame_num].to(self.device_ids[1]),
                                        frames[:, :, self.cond_frame_num:self.cond_frame_num + self.pred_frame_num].to(self.device_ids[1]),
                                        cond_fea=ref_img_fea.to(self.device_ids[1]))
            ret['loss'] = loss

            with torch.no_grad():
                fake_out_img_list = []
                fake_warped_img_list = []
                if self.use_residual_flow:
                    fake_vid_grid = pred[:, :2, :, :, :] + identity_grid
                else:
                    #预测出来的遮挡图在光流
                    fake_vid_grid = pred[:, :2, :, :, :]
                #预测出来的遮挡图 使用线性变换映射到【0,1】范围
                fake_vid_conf = (pred[:, 2, :, :, :].unsqueeze(dim=1) + 1) * 0.5
                #逐帧生成图像与重建损失计算
                perceptual_loss = 0
                # pyramide_loss = 0
                for idx in range(self.pred_frame_num):
                    #预测出来的光流和遮挡图
                    fake_grid = fake_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1)
                    fake_conf = fake_vid_conf[:, :, idx, :, :]


                    # predict fake out image and fake warped image
                    generated = self.generator.forward_with_flow(source_image=ref_img.to(self.device_ids[0]), optical_flow=fake_grid.to(self.device_ids[0]),
                                                                 occlusion_map=fake_conf.to(self.device_ids[0]))
                    #预测出来的图像
                    fake_out_img_list.append(generated["prediction"])
                    # fake_vgg = self.vgg(generated["prediction"].to(self.device_ids[0]))
                    # real_vgg = self.vgg(pred_frames[:,:,idx,:,:].to(self.device_ids[0]))

                    # for i, weight in enumerate([10, 10, 10, 10, 10]):
                    #     layer_loss = torch.abs(fake_vgg[i] - real_vgg[i].detach()).mean()
                    #     perceptual_loss += weight * layer_loss
                    # print(f"{idx} pre_loss = {perceptual_loss}")
                    # #预测出来
                    # pyramide_real = self.pyramid(generated["prediction"].to(self.device_ids[0]))
                    # pyramide_generated = self.pyramid(pred_frames[:,:,idx,:,:].to(self.device_ids[0]))
                    # print(pyramide_real)
                    # for scale in self.scales:
                    #     real_img = pyramide_real['prediction_' + str(scale)].to(self.device_ids[0])
                    #     generated_img = pyramide_generated['prediction_' + str(scale)].to(self.device_ids[0])
                    #
                    #     real_vgg = self.vgg(real_img)
                    #     generated_vgg = self.vgg(generated_img)
                    #
                    #     for i, weight in enumerate([10, 10, 10, 10, 10]):
                    #         pyramide_loss += weight * torch.abs(real_vgg[i] - generated_vgg[i].detach()).mean()
                    fake_warped_img_list.append(generated["deformed"])

                fake_out_vid = torch.stack(fake_out_img_list, dim=2)
                fake_warped_vid = torch.stack(fake_warped_img_list, dim=2)

                rec_loss = nn.L1Loss()(pred_frames * 10, fake_out_vid * 10)
                rec_warp_loss = nn.L1Loss()(pred_frames * 10, fake_warped_vid * 10)





                ret['fake_vid_grid'] = fake_vid_grid
                ret['fake_vid_conf'] = fake_vid_conf
                ret['fake_out_vid'] = fake_out_vid
                ret['fake_warped_vid'] = fake_warped_vid
                ret['rec_loss'] = rec_loss
                # ret['vgg_loss'] = perceptual_loss
                # ret['pyramide_loss'] = pyramide_loss
                ret['rec_warp_loss'] = rec_warp_loss

        return ret

    def sample_one_video(self, cond_scale, real_vid):
        ret = {}
        real_grid_list = []
        real_conf_list = []
        real_out_img_list = []
        real_warped_img_list = []
        ref_img_fea = []

        with torch.no_grad():
            ref_img = real_vid[:, :, self.cond_frame_num - 1]
            # reference image = condition frames [t-1]
            source_region_params = self.region_predictor(ref_img.to(self.device_ids[0]))
            for idx in range(self.cond_frame_num):
                driving_region_params = self.region_predictor(real_vid[:, :, idx, :, :].to(self.device_ids[0]))
                bg_params = self.bg_predictor(ref_img.to(self.device_ids[0]), real_vid[:, :, idx, :, :].to(self.device_ids[0]))
                generated = self.generator(ref_img.to(self.device_ids[0]), source_region_params=source_region_params,
                                           driving_region_params=driving_region_params, bg_params=bg_params)
                if idx != self.cond_frame_num - 1:
                    ref_img_fea.append(self.generator.forward_bottle(real_vid[:, :, idx, :, :]).detach())
                generated.update({'source_region_params': source_region_params,
                                  'driving_region_params': driving_region_params})
                real_grid_list.append(generated["optical_flow"].permute(0, 3, 1, 2))
                # normalized occlusion map
                if self.estimate_occlusion_map:
                    real_conf_list.append(generated["occlusion_map"])
                real_out_img_list.append(generated["prediction"])
                real_warped_img_list.append(generated["deformed"])

            for _ in range(1 + self.pred_frame_num):
                ref_img_fea.append(generated["bottle_neck_feat"].detach())
            ref_img_fea = torch.stack(ref_img_fea, dim=2)
            # ref_img_fea = rearrange(ref_img_fea, 'n c t h w->(n t) c h w')
            # ref_img_fea = F.interpolate(ref_img_fea, size=generated["optical_flow"].shape[-3:-1], mode='bilinear')
            # ref_img_fea = rearrange(ref_img_fea, '(n t) c h w->n c t h w',t=self.cond_frame_num+self.pred_frame_num)

            del real_vid
            torch.cuda.empty_cache()

            real_vid_grid = torch.stack(real_grid_list, dim=2)
            if self.estimate_occlusion_map:
                real_vid_conf = torch.stack(real_conf_list, dim=2)
            real_out_vid = torch.stack(real_out_img_list, dim=2)
            real_warped_vid = torch.stack(real_warped_img_list, dim=2)
            # if self.withFea:
            #     ref_img_fea = F.interpolate(generated["bottle_neck_feat"].detach(), size=real_vid_conf.shape[-2:], mode='bilinear')
            # else:
            #     ref_img_fea = None
            ret['real_vid_grid'] = real_vid_grid
            if self.estimate_occlusion_map:
                ret['real_vid_conf'] = real_vid_conf
            ret['real_out_vid'] = real_out_vid
            ret['real_warped_vid'] = real_warped_vid
            if self.estimate_occlusion_map:
                x_cond = torch.cat((real_vid_grid, real_vid_conf * 2 - 1), dim=1)
            else:
                x_cond = torch.cat((real_vid_grid, torch.zeros_like(real_vid_grid)[:, 0:1]), dim=1)


        # if cond_scale = 1.0, not using unconditional model
        pred = self.diffusion.sample(x_cond.to(self.device_ids[1]), cond_fea=ref_img_fea.to(self.device_ids[1]), batch_size=1, cond_scale=cond_scale)
        if self.use_residual_flow:
            b, _, nf, h, w = pred[:, :2, :, :, :].size()
            identity_grid = self.get_grid(b, 1, h, w, normalize=True).cuda()
            sample_vid_grid = torch.cat(
                [real_vid_grid[:, :, :self.cond_frame_num], pred[:, :2, :, :, :] + identity_grid], dim=2)
        else:
            sample_vid_grid = torch.cat([real_vid_grid[:, :, :self.cond_frame_num].to(self.device_ids[1]), pred[:, :2, :, :, :].to(self.device_ids[1])], dim=2)
        if self.estimate_occlusion_map:
            sample_vid_conf = torch.cat(
                [real_vid_conf[:, :, :self.cond_frame_num].to(self.device_ids[1]), (pred[:, 2, :, :, :].to(self.device_ids[1]).unsqueeze(dim=1) + 1) * 0.5], dim=2)

        with torch.no_grad():
            sample_out_img_list = []
            sample_warped_img_list = []
            for idx in range(sample_vid_grid.size(2)):
                sample_grid = sample_vid_grid[:, :, idx, :, :].permute(0, 2, 3, 1)
                if self.estimate_occlusion_map:
                    sample_conf = sample_vid_conf[:, :, idx, :, :]
                else:
                    sample_conf = None
                # predict fake out image and fake warped image
                generated = self.generator.forward_with_flow(source_image=ref_img.to(self.device_ids[0]),
                                                             optical_flow=sample_grid.to(self.device_ids[0]),
                                                             occlusion_map=sample_conf.to(self.device_ids[0]))
                sample_out_img_list.append(generated["prediction"])
                sample_warped_img_list.append(generated["deformed"])
            sample_out_vid = torch.stack(sample_out_img_list, dim=2)
            sample_warped_vid = torch.stack(sample_warped_img_list, dim=2)
            ret['sample_vid_grid'] = sample_vid_grid
            if self.estimate_occlusion_map:
                ret['sample_vid_conf'] = sample_vid_conf
            ret['sample_out_vid'] = sample_out_vid
            ret['sample_warped_vid'] = sample_warped_vid
            # ret['sample_refined_out_vid'] = self.refine(cond = cond_frames, pred = sample_out_vid[:,:,self.cond_frame_num:])

        return ret

    def get_grid(self, b, nf, H, W, normalize=True):
        if normalize:
            h_range = torch.linspace(-1, 1, H)
            w_range = torch.linspace(-1, 1, W)
        else:
            h_range = torch.arange(0, H)
            w_range = torch.arange(0, W)
        grid = torch.stack(torch.meshgrid([h_range, w_range]), -1).repeat(b, 1, 1, 1).flip(3).float()  # flip h,w to x,y
        return grid.permute(0, 3, 1, 2).unsqueeze(dim=2).repeat(1, 1, nf, 1, 1)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    bs = 5
    img_size = 64
    num_frames = 40
    ref_text = ["play basketball"] * bs
    ref_img = torch.rand((bs, 3, img_size, img_size), dtype=torch.float32)
    real_vid = torch.rand((bs, 3, num_frames, img_size, img_size), dtype=torch.float32)
    model = FlowDiffusion(use_residual_flow=False,
                          sampling_timesteps=10,
                          img_size=16,
                          config_path="/workspace/code/CVPR23_LFDM/config/mug128.yaml",
                          pretrained_pth="")
    model.cuda()
    # model.train()
    # model.set_train_input(ref_img=ref_img, real_vid=real_vid, ref_text=ref_text)
    # model.optimize_parameters()
    model.eval()
    model.set_sample_input(sample_img=ref_img, sample_text=ref_text)
    model.sample_one_video(cond_scale=1.0)