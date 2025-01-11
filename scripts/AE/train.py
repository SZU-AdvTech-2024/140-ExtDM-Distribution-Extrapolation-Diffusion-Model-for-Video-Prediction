# train a LFAE
# this code is based on RegionMM (MRAA): https://github.com/snap-research/articulated-animation
import os.path
from shutil import copy2
import torch
from draw_curve_ae import *
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
# from torch import autograd
import torch.nn as nn
import torch.backends.cudnn as cudnn
from utils.seed import setup_seed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from model.LFAE.model import ReconstructionModel
from model.LFAE.util import Visualizer
from model.LFAE.sync_batchnorm import DataParallelWithCallback

from data.two_frames_dataset import TwoFramesDataset
from data.video_dataset import VideoDataset, dataset2videos
from data.two_frames_dataset import DatasetRepeater

import timeit
import imageio
import math
import random
from einops import rearrange

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils.meter import AverageMeter
from model.LFAE.flow_autoenc import FlowAE

import os
import logging


def train(
        config,
        dataset_params,
        train_params,
        generator,
        region_predictor,
        bg_predictor,
        log_dir,
        checkpoint,
        device_ids,
):
    log_dir = '/mnt/d/ExtDM/logs_training/AE/based_KTH5/kth'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training.log")

    # dist.init_process_group(backend='nccl', init_method='env://')

    logging.basicConfig(
        filename=log_file,
        filemode='a',  # 'w' 表示写入（覆盖），'a' 表示追加
        level=logging.INFO,  # 记录级别
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    print(config)
    # device = torch.device(f"cuda:{device_ids[rank]}")
    # torch.cuda.set_device(device)
    # dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23456', rank=rank, world_size=world_size)

    train_dataset = TwoFramesDataset(
        root_dir=dataset_params['root_dir'],
        type=dataset_params['train_params']['type'],
        total_videos=-1,
        frame_shape=dataset_params['frame_shape'],
        max_frame_distance=dataset_params['max_frame_distance'],
        min_frame_distance=dataset_params['min_frame_distance'],
        augmentation_params=dataset_params['augmentation_params']
    )

    valid_dataset = VideoDataset(
        data_dir=dataset_params['root_dir'],
        type=dataset_params['valid_params']['type'],
        total_videos=dataset_params['valid_params']['total_videos'],
        num_frames=dataset_params['valid_params']['cond_frames'] + dataset_params['valid_params']['pred_frames'],
        image_size=dataset_params['frame_shape'],
        random_horizontal_flip=dataset_params['augmentation_params']['flip_param']['horizontal_flip']
    )

    # 计算一个 epoch 有多少 step
    steps_per_epoch = math.ceil(train_params['num_repeats'] * len(train_dataset) / float(train_params['batch_size']))
    # 多少 step 保存一次模型
    save_ckpt_freq = train_params['save_ckpt_freq']
    # 总共只保存 10 次模型，计算每次需要多少 step
    # save_ckpt_freq = steps_per_epoch * (train_params['max_epochs'] // 10)
    print("save ckpt freq:", save_ckpt_freq)
    # generator = generator.to(device)
    # region_predictor = region_predictor.to(device)
    # bg_predictor = bg_predictor.to(device)
    loss_weights = {
        'perceptual':nn.Parameter(torch.tensor(1.0, requires_grad=True)),
        'equivariance_shift': nn.Parameter(torch.tensor(1.0, requires_grad=True)),
        'equivariance_affine': nn.Parameter(torch.tensor(1.0, requires_grad=True)),
        'reconstruction': nn.Parameter(torch.tensor(1.0, requires_grad=True))
    }
    optimizer = torch.optim.Adam(
        list(generator.parameters()) +
        list(region_predictor.parameters()) +
        list(bg_predictor.parameters()) +
        list(loss_weights.values()),
        lr=train_params['lr'],
        betas=(0.5, 0.999),
        # weight_decay = 1e-5
    )

    start_epoch = 0
    start_step = 0
    best_fvd = 1e5

    if checkpoint != "":
        ckpt = torch.load(checkpoint)
        if config["set_start"]:
            start_step = int(math.ceil(ckpt['example'] / train_params['batch_size']))
            start_epoch = ckpt['epoch']

            print("start_step", start_step)
            print("start_epoch", start_epoch)

        if config['dataset_params']['frame_shape'] == 64:
            if config['flow_params']['model_params']['region_predictor_params']['scale_factor'] != 1.0 and \
                    config['flow_params']['model_params']['generator_params']['pixelwise_flow_predictor_params'][
                        'scale_factor'] != 1.0:
                ckpt['generator'][
                    'pixelwise_flow_predictor.down.weight'] = generator.pixelwise_flow_predictor.down.weight
                ckpt['region_predictor']['down.weight'] = region_predictor.down.weight
            else:
                print('scale == 1.0, delete down')
                del ckpt['generator']['pixelwise_flow_predictor.down.weight']
                del ckpt['region_predictor']['down.weight']

        generator.load_state_dict(ckpt['generator'])
        region_predictor.load_state_dict(ckpt['region_predictor'])
        bg_predictor.load_state_dict(ckpt['bg_predictor'])

        # // 处理新的loss_weight
        if 'loss_weights' in ckpt:
            loss_weights.load_state_dict(ckpt['optimizer'])
        else:
            # 如果没有loss_weights，就按需要设置初始值
            loss_weights['perceptual'] = nn.Parameter(torch.tensor(1.0, requires_grad=True))
            loss_weights['equivariance_shift'] = nn.Parameter(torch.tensor(1.0, requires_grad=True))
            loss_weights['equivariance_affine'] = nn.Parameter(torch.tensor(1.0, requires_grad=True))
            loss_weights['reconstruction'] = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        if 'optimizer' in list(ckpt.keys()):
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                print(f' Error loading optimizer state_dict:{e}')
        else:
            print("No optimizer state found in chekpoint, initialingzing oprimizer")

        # 在加载检查点后手动设置学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = train_params['lr']
        #     param_group['lr' ] = optimizer
    scheduler = MultiStepLR(optimizer, last_epoch=start_step - 1, **train_params['scheduler_param'])

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        train_dataset = DatasetRepeater(train_dataset, train_params['num_repeats'])

    # #add
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False)


    train_dataloader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True,
                                  num_workers=train_params['dataloader_workers'], drop_last=False)
                                  # num_workers=0, drop_last=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=train_params['valid_batch_size'], shuffle=False,
                                  num_workers=train_params['dataloader_workers'], drop_last=False)
                                  # num_workers=0, drop_last=False)
    #
    model = ReconstructionModel(region_predictor, bg_predictor, generator, train_params, device_ids)

    # world_size = torch.cuda.device_count()

    visualizer = Visualizer(**config['visualizer_params'])
    # device = torch.device(f"cuda:{device_ids[0]}")  # 选择首个设备作为主设备
    # if torch.cuda.is_available():
    #
    #     model = model.to(device)
    #     if ('use_sync_bn' in train_params) and train_params['use_sync_bn']:
    #         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # 转换为同步 BatchNorm
    #         model = torch.nn.DataParallel(model, device_ids=device_ids)
    #     else:
    #         model = torch.nn.DataParallel(model, device_ids=device_ids)
    # model = model.to(device)
    # if 'use_sync_bn' in train_params and train_params['use_sync_bn']:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 使用 DistributedDataParallel
    # model = DDP(model, device_ids= [device.index], output_device=device)

    # rewritten by nhm
    batch_time = AverageMeter()
    data_time = AverageMeter()

    total_losses = AverageMeter()
    losses_perc = AverageMeter()
    losses_equiv_shift = AverageMeter()
    losses_equiv_affine = AverageMeter()


    cnt = 0
    epoch_cnt = start_epoch
    actual_step = start_step
    final_step = steps_per_epoch * train_params["max_epochs"]
    train_iter = []
    train_loss = []
    train_loss_perceptual = []
    train_loss_equivariance_shift = []
    train_loss_equivariance_affine = []
    train_loss_reconstruction = []
    train_path = log_dir
    test_step = []
    test_fvd = []
    test_ssim = []
    test_psnr = []
    test_lpips = []
    test_path = train_path
    while actual_step < final_step:
        iter_end = timeit.default_timer()

        for i_iter, x in enumerate(train_dataloader):
            actual_step = int(start_step + cnt)
            data_time.update(timeit.default_timer() - iter_end)

            optimizer.zero_grad()
            # print("iter:",i_iter)
            # with autograd.detect_anomaly():
            # if isinstance(x, dict):
            #     x = {key: value.to(device) if torch.is_tensor(value) else value for key, value in x.items()}
            # elif isinstance(x, list):
            #     x = [item.to(device) if torch.is_tensor(item) else item for item in x]
            # else:
            #     x = x.to(device)
            # model = model.to(device)


            losses, generated = model(x)

            # print(losses['perceptual'].mean().item())
            # print(losses['equivariance_shift'].mean().item())
            # print(losses['equivariance_affine'].mean().item())

            loss_reconsruction  = losses['reconstruction'].to(device_ids[0]).mean() * loss_weights['reconstruction'].to(device_ids[0])
            loss_perceptual = losses['perceptual'].to(device_ids[0]).mean()* loss_weights['perceptual'].to(device_ids[0])
            loss_equivariance_shift = losses['equivariance_shift'].to(device_ids[0]).mean() * loss_weights['equivariance_shift'].to(device_ids[0])
            loss_equivariance_affine = losses['equivariance_affine'].to(device_ids[0]).mean() * loss_weights['equivariance_affine'].to(device_ids[0])




            loss = (loss_perceptual
                    + loss_equivariance_shift
                    + loss_equivariance_affine
                    +loss_reconsruction )
            # loss = loss.to(device)
            train_iter.append(actual_step)
            train_loss.append(loss)
            train_loss_perceptual.append(loss_perceptual)
            train_loss_equivariance_affine.append(loss_equivariance_affine)
            train_loss_equivariance_shift.append(loss_equivariance_shift)
            train_loss_reconstruction.append(loss_reconsruction)
            draw_curve_train(os.path.join(train_path, 'train_loss.jpg'), train_iter,train_loss, train_loss_perceptual,train_loss_equivariance_affine, train_loss_equivariance_shift, train_loss_reconstruction)


            loss.backward()

            optimizer.step()
            scheduler.step()
            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            bs = x['source'].size(0)
            total_losses.update(loss.item(), bs)
            losses_perc.update(loss_perceptual.item(), bs)
            losses_equiv_shift.update(loss_equivariance_shift.item(), bs)
            losses_equiv_affine.update(loss_equivariance_affine.item(), bs)

            if actual_step % train_params["print_freq"] == 0 or epoch_cnt%100==0:
                print('iter: [{0}]{1}/{2}\t'
                      'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'loss_perc {loss_perc.val:.4f} ({loss_perc.avg:.4f})\n'
                      'loss_shift {loss_shift.val:.4f} ({loss_shift.avg:.4f})\t'
                      'loss_affine {loss_affine.val:.4f} ({loss_affine.avg:.4f})\t'
                      'time {batch_time.val:.4f} ({batch_time.avg:.4f})'
                .format(
                    cnt, actual_step, final_step,
                    loss=total_losses,
                    loss_perc=losses_perc,
                    loss_shift=losses_equiv_shift,
                    loss_affine=losses_equiv_affine,
                    batch_time=batch_time
                ))
                logging.info(
                    f"Epoch[{epoch_cnt}], Loss: {total_losses.avg:.4f}, "
                    f"loss_prec: [{losses_perc.avg:.4f}], "
                    f"loss_shift: [{losses_equiv_shift.avg:.4f}], "
                    f"loss_affine: [{losses_equiv_affine.avg:.4f}]"
                )#
                # wandb.log({
                #     "actual_step": actual_step,
                #     "lr": optimizer.param_groups[0]["lr"],
                #     "loss": total_losses.val,
                #     "loss_perc": losses_perc.val,
                #     "loss_shift": losses_equiv_shift.val,
                #     "loss_affine": losses_equiv_affine.val,
                #     "batch_time": batch_time.avg
                # })

            if actual_step % train_params['save_img_freq'] == 0:
                save_image = visualizer.visualize(x['driving'], x['source'], generated, index=0)
                save_name = 'B' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") \
                            + '_' + str(x["frame"][0][0].item()) + '_to_' + str(x["frame"][0][1].item()) + '.png'
                save_file = os.path.join(config["imgshots"], save_name)
                imageio.imsave(save_file, save_image)
                # wandb.log({
                #     "save_img": wandb.Image(save_image)
                # })

            if actual_step % save_ckpt_freq == 0 and cnt != 0:
                print('taking snapshot...')
                checkpoint_save_path = os.path.join(
                    config["snapshots"],
                    'RegionMM_' + format(train_params["batch_size"], "04d") + '_S' + format(actual_step, "06d") + '.pth'
                )
                torch.save({
                    'example': actual_step * train_params["batch_size"],
                    'epoch': epoch_cnt,
                    'generator': generator.state_dict(),
                    'bg_predictor': bg_predictor.state_dict(),
                    'region_predictor': region_predictor.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                    checkpoint_save_path
                )

            if actual_step % train_params["update_ckpt_freq"] == 0 and cnt != 0:
                print('updating snapshot...')
                checkpoint_save_path = os.path.join(config["snapshots"], 'RegionMM.pth')
                torch.save({'example': actual_step * train_params["batch_size"],
                            'epoch': epoch_cnt,
                            'generator': generator.state_dict(),
                            'bg_predictor': bg_predictor.state_dict(),
                            'region_predictor': region_predictor.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           checkpoint_save_path)
                metrics = valid(config, valid_dataloader, checkpoint_save_path, log_dir, actual_step)
                test_step.append(actual_step)
                test_fvd.append(metrics['metrics/fvd'])
                test_psnr.append((metrics['metrics/psnr']))
                test_ssim.append(metrics['metrics/ssim'])
                test_lpips.append(metrics['metrics/lpips'])
                draw_curve_test(os.path.join(test_path,'test_metrics.jpg'),test_step,test_fvd,test_ssim,test_psnr,test_lpips)
                if metrics['metrics/fvd'] < best_fvd:
                    best_fvd = metrics['metrics/fvd']
                    copy2(os.path.join(config["snapshots"], 'RegionMM.pth'),
                          os.path.join(config["snapshots"], f'RegionMM_best_{actual_step}_{best_fvd:.3f}.pth'))

                # wandb.log(metrics)

            if actual_step >= final_step:
                break

            cnt += 1
            # 按 step 进行 warmup 策略


        epoch_cnt += 1
        print("epoch %d, lr= %.7f" % (epoch_cnt, optimizer.param_groups[0]["lr"]))

    print('save the final model...')
    torch.save({'example': actual_step * train_params["batch_size"],
                'epoch': epoch_cnt,
                'generator': generator.state_dict(),
                'bg_predictor': bg_predictor.state_dict(),
                'region_predictor': region_predictor.state_dict(),
                'optimizer': optimizer.state_dict()},
               os.path.join(config["snapshots"],
                            'RegionMM_' + format(train_params["batch_size"], "04d") +
                            '_S' + format(actual_step, "06d") + '.pth'))


def valid(config, valid_dataloader, checkpoint_save_path, log_dir, actual_step):
    cudnn.enabled = True
    cudnn.benchmark = True
    setup_seed(1234)

    model = FlowAE(is_train=False, config=config)
    model.cuda()

    checkpoint = torch.load(checkpoint_save_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.generator.load_state_dict(checkpoint['generator'])
    model.region_predictor.load_state_dict(checkpoint['region_predictor'])
    model.bg_predictor.load_state_dict(checkpoint['bg_predictor'])

    model.eval()

    dataset_params = config['dataset_params']
    train_params = config['flow_params']['train_params']

    from math import ceil
    NUM_ITER = ceil(dataset_params['valid_params']['total_videos'] / train_params['valid_batch_size'])
    cond_frames = dataset_params['valid_params']['cond_frames']
    pred_frames = dataset_params['valid_params']['pred_frames']

    origin_videos = []
    result_videos = []

    for i_iter, batch in enumerate(valid_dataloader):
        if i_iter >= NUM_ITER:
            break

        total_vids, video_names = batch
        # (b t c h)/(b t h w c) -> (b t c h w)
        total_vids = dataset2videos(total_vids)
        origin_videos.append(total_vids)

        # real_vids
        # torch.Size([b, 50, 3, 64, 64]) -> b c t h w
        # tensor(0.0431) tensor(0.9647)
        total_vids = total_vids.permute(0, 2, 1, 3, 4).contiguous()

        cond_vids = total_vids[:, :, :cond_frames, :, :]
        real_vids = total_vids[:, :, cond_frames:, :, :]

        # use first frame of each video as reference frame (vids: B C T H W)
        # 在 flowae 阶段，固定 ref_imgs 为初始帧，不随自回归变化
        ref_imgs = cond_vids[:, :, -1, :, :].clone().detach()

        assert real_vids.size(2) == pred_frames

        out_img_list = []
        warped_img_list = []
        warped_grid_list = []
        conf_map_list = []

        for frame_idx in range(pred_frames):
            dri_imgs = real_vids[:, :, frame_idx, :, :]
            with torch.no_grad():
                model.set_train_input(ref_img=ref_imgs, dri_img=dri_imgs)
                model.forward()
            out_img_list.append(model.generated['prediction'].clone().detach())
            warped_img_list.append(model.generated['deformed'].clone().detach())
            warped_grid_list.append(model.generated['optical_flow'].clone().detach())
            conf_map_list.append(model.generated['occlusion_map'].clone().detach())

        out_img_list_tensor = torch.stack(out_img_list, dim=0)
        warped_img_list_tensor = torch.stack(warped_img_list, dim=0)
        warped_grid_list_tensor = torch.stack(warped_grid_list, dim=0)
        conf_map_list_tensor = torch.stack(conf_map_list, dim=0)

        from utils.visualize import LFAE_visualize
        LFAE_visualize(
            ground=real_vids,
            prediction=out_img_list_tensor,
            deformed=warped_img_list_tensor,
            optical_flow=warped_grid_list_tensor,
            occlusion_map=conf_map_list_tensor,
            video_names=video_names,
            save_path=f"{log_dir}/flowae_result",
            save_num=8,
            epoch_or_step_num=actual_step,
            image_size=ref_imgs.shape[-1]
        )

        # out_img_list_tensor      [40, 8, 3, 64, 64]
        # warped_img_list_tensor   [40, 8, 3, 64, 64]
        # warped_grid_list_tensor  [40, 8, 32, 32, 2]
        # conf_map_list_tensor     [40, 8, 1, 32, 32]

        tmp_result = torch.cat([
            rearrange(cond_vids.cpu(), 'b c t h w -> b t c h w'),
            rearrange(out_img_list_tensor.cpu(), 't b c h w -> b t c h w')
        ],
            dim=1
        )
        result_videos.append(tmp_result)

        print('Test:[{0}/{1}]\t'.format(i_iter, NUM_ITER))

    origin_videos = torch.cat(origin_videos)
    result_videos = torch.cat(result_videos)
    print(origin_videos.shape, result_videos.shape)

    from utils.visualize import visualize
    visualize(
        save_path=f"{log_dir}/video_result",
        origin=origin_videos,
        result=result_videos,
        save_pic_num=8,
        select_method='linspace',
        grid_nrow=4,
        save_gif_grid=True,
        save_gif=True,
        save_pic_row=False,
        save_pic=False,
        epoch_or_step_num=actual_step,
        cond_frame_num=cond_frames,
    )

    from metrics.calculate_fvd import calculate_fvd1
    from metrics.calculate_psnr import calculate_psnr1
    from metrics.calculate_ssim import calculate_ssim1
    from metrics.calculate_lpips import calculate_lpips1
    device = torch.device("cuda")
    videos1 = origin_videos
    videos2 = result_videos

    fvd = calculate_fvd1(videos1, videos2, device, mini_bs=4)
    videos1 = videos1[:, cond_frames:]
    videos2 = videos2[:, cond_frames:]
    ssim = calculate_ssim1(videos1, videos2)[0]
    psnr = calculate_psnr1(videos1, videos2)[0]
    lpips = calculate_lpips1(videos1, videos2, device)[0]

    print("[fvd    ]", fvd)
    print("[ssim   ]", ssim)
    print("[psnr   ]", psnr)
    print("[lpips  ]", lpips)
    logging.info(f'Valid fvd:{fvd},ssim:{ssim},psnr:{psnr},lpis:{lpips}')




    return {
        'actual_step': actual_step,
        'metrics/fvd': fvd,
        'metrics/ssim': ssim,
        'metrics/psnr': psnr,
        'metrics/lpips': lpips
    }

# flowae_result     warp的一个视频结果
# imgshots          warp的一帧结果
# snapshots         保存模型
# video_result      warp的一组视频结果