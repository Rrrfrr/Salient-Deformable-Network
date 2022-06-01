#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/4/2 21:18
# @Author  : Eric Ching
from config import FubuConfig
import torch
import matplotlib.pyplot as plt
from torch.nn import functional as F
from timeit import default_timer as timer
import os
from schedulers import PolyLR
from collections import defaultdict
from utils import mkdir, load_checkpoint, get_learning_rate, save_checkpoint
from utils import set_logger, AverageMeter, time_to_str, init_env, dict_to_csv,list_to_csv
from losses import l2_smooth3d, ncc_loss,patch_ncc_loss,residual_complexity_loss
from metrics import parse_lable_subpopulation,IOU_subpopulation,dice_coef,dice_coef_np
from model.wnet import WNet3D
from data.abdmon_dataloader_counter import abdomen_data
import numpy as np
import nibabel as nib
import visdom
from tqdm import trange
import module.common_module as cm
from module.dice_loss import DiceCoefficientLF
from module.visualize_attention import visualize_Seg, visualize_Rec, visualize_loss
from module.eval_attention_BraTS_slidingwindow import eval_net_dice, eval_net_mse, test_net_dice
from network import ssl_3d_attention
import time
import copy


def requires_grad(param):
    return param.requires_grad

device = torch.device('cuda')

def train_one_peoch(model_reg, modelDataLoader, optimizer, losses, metrics, epoch, since ,model_seg,
                    device, root_path, network_switch, criterion, optimizer_seg, scheduler_seg,
                num_epochs=25, loss_weighted=True, jointly=False, mode='fuse',clip_grident=False):
    since = time.time()
    batch_fix = 0
    batch_move = 0
    seg_fix = 0
    # mask_fix =0
    # mask_move =0
    seg_move = 0
    inputs = 0
    labels = 0
    image = 0
    image2 = 0
    outputsL = 0
    outputsU_back = 0
    outputsU_fore = 0
    labels_back = 0
    labels_fore = 0
    labels_fore_vis = 0

    loss = 0
    loss_reg = 0

    PREVIEW = True

    dict = defaultdict(list)
    best_model_wts = copy.deepcopy(model_seg.state_dict())
    best_val_dice = 0.0
    best_val_mse = 1.0
    best_epoch = 0

    epoch_val_loss = np.array([0.0, 1.0])

    epoch_val_dice = 0.0
    epoch_val_mse = 1.0

    # set TQDM iterator
    # tqiter = trange(num_epochs, desc='BraTS')
    epoch_train_loss = np.array([0.0, 1.0])
    fig_loss = plt.figure(num='loss', figsize=[12, 3.8])
    print('---epoch---',epoch)

    for i, (sample1, sample2) in enumerate(zip(modelDataLoader['trainLabeled'], modelDataLoader['trainUnlabeled'])):
        if i < (len(modelDataLoader['trainLabeled']) - 1) and i < (len(modelDataLoader['trainUnlabeled']) - 1):
            procedure = ['trainLabeled', 'trainUnlabeled']
        else:

            procedure = ['trainLabeled', 'trainUnlabeled', 'val_labeled', 'val_unlabeled']

        # run training and validation alternatively
        for phase in procedure:
            if phase == 'trainLabeled':
                scheduler_seg[0].step()
                scheduler_seg[2].step()
                model_seg.train()
            elif phase == 'trainUnlabeled':
                scheduler_seg[1].step()
                model_seg.train()
            else:
                model_seg.eval()

            running_loss = 0.0

            # If 'labeled', then use segmentation mask; else use image for reconstruction
            if phase == 'trainLabeled':

                inputs = sample1['image'][:].float().to(device)  # batch, FLAIR
                labels = sample1['mask'][:].float().to(device)
                #seg_fix = sample1['seg'][:].float().to(device)
                dist = sample1['dist'][:].float().to(device)
                image = sample1['image'][:].float().to(device)

                mask_fix = labels

            elif phase == 'trainUnlabeled':

                inputs = sample2['image'][:].float().to(device)
                labels = sample2['mask'][:].float().to(device)  # batch, FLAIR
                image = sample2['image'][:].float().to(device)  # batch, FLAIR
                #seg_move = sample1['seg'][:].float().to(device)

            optimizer_seg[0].zero_grad()
            optimizer_seg[1].zero_grad()
            optimizer_seg[2].zero_grad()

            # update model parameters and compute loss

            with torch.set_grad_enabled(phase == 'trainLabeled' or phase == 'trainUnlabeled' or phase == 'reg'):
                if phase == 'trainLabeled':

                    outputsL, outputsU,out_dist = model_seg(inputs, phase=phase, network_switch=network_switch)

                    if mode == 'fuse':

                        outputsU_back = outputsU[:, 0]
                        outputsU_fore = outputsU[:, 1]
                        labels_back = (1.0 - outputsL) * image.float()
                        labels_fore = outputsL * image.float()

                    w1 = 1.0
                    w2 = 1.0
                    w3 = 0.0

                    loss = w1 * criterion[0](outputsL.float(), labels.float()) + \
                           w2 * criterion[1](outputsU_back.float(), labels_back.float()) + \
                           w3 * criterion[1](outputsU_fore.float(), labels_fore.float())+ w1 * criterion[0](out_dist.float(), dist.float())

                elif phase == 'trainUnlabeled':

                    outputsL, outputsU,out_dist = model_seg(inputs, phase=phase, network_switch=network_switch)

                    if mode == 'fuse':
                        outputsU_back = outputsU[:, 0]
                        outputsU_fore = outputsU[:, 1]
                        labels_back = (1.0 - outputsL) * image.float()
                        labels_fore = outputsL * image.float()

                    if loss_weighted:
                        w2 = torch.sum((1.0 - outputsL))
                        w3 = torch.sum(outputsL)
                        total = w2 + w3
                        w2 = w2 / total
                        w3 = w3 / total

                    # loss = criterion[1](outputsU.float(), outputsL.float())

                    w2 = 0.0
                    w3 = 1.0
                    loss = w2 * criterion[1](outputsU_back.float(), labels_back.float()) + \
                           w3 * criterion[1](outputsU_fore.float(), labels_fore.float())

                outputsL_vis = outputsL.cpu().detach().numpy()
                outputsU_back_vis = outputsU_back.cpu().detach().numpy()
                outputsU_fore_vis = outputsU_fore.cpu().detach().numpy()
                inputs_vis = inputs.cpu().detach().numpy()
                labels_vis = labels.cpu().detach().numpy()
                labels_back_vis = labels_back.cpu().detach().numpy()
                labels_fore_vis = labels_fore.cpu().detach().numpy()
                dst_out_dist = out_dist.cpu().detach().numpy()
                label_out_dist = dist.cpu().detach().numpy()

                # visualize training set at the end of each epoch
                if PREVIEW:
                    if i == (len(modelDataLoader['trainLabeled']) - 1):
                        if phase == 'trainLabeled' or phase == 'trainUnlabeled':

                            if phase == 'trainLabeled':

                                fig = visualize_Seg(inputs_vis[0][0], labels_vis[0][0], outputsL_vis[0][0],
                                                    figsize=(6, 6), epoch=epoch)
                                plt.savefig(root_path + 'preview/train/Labeled/' + 'epoch_%s.jpg' % epoch)
                                plt.close(fig)
                                save_volume(inputs_vis[0][0],
                                            root_path + 'preview/train/Labeled/' + 'epoch_%s_volume.nii.gz' % epoch)
                                save_label(labels_vis[0][0],
                                           root_path + 'preview/train/Labeled/' + 'epoch_%s_gt.nii.gz' % epoch)
                                save_mask(outputsL_vis[0][0],
                                          root_path + 'preview/train/Labeled/' + 'epoch_%s_pred.nii.gz' % epoch)
                                save_lever(dst_out_dist[0][0],
                                            root_path + 'preview/train/Labeled/' + 'epoch_%s_pred_dist.nii.gz' % epoch)
                                save_lever(label_out_dist[0][0],
                                            root_path + 'preview/train/Labeled/' + 'epoch_%s_label_dist.nii.gz' % epoch)

                            elif phase == 'trainUnlabeled':

                                fig = visualize_Rec(inputs_vis[0][0], labels_back_vis[0, 0], labels_fore_vis[0, 0],
                                                    outputsU_back_vis[0], outputsU_fore_vis[0], figsize=(6, 6),
                                                    epoch=epoch)
                                plt.savefig(root_path + 'preview/train/Unlabeled/' + 'epoch_%s.jpg' % epoch)
                                plt.close(fig)
                                save_volume(labels_back_vis[0, 0],
                                            root_path + 'preview/train/Unlabeled/' + 'epoch_%s_label_back.nii.gz' % epoch)
                                save_volume(labels_fore_vis[0, 0],
                                            root_path + 'preview/train/Unlabeled/' + 'epoch_%s_label_fore.nii.gz' % epoch)
                                save_volume(outputsU_back_vis[0],
                                            root_path + 'preview/train/Unlabeled/' + 'epoch_%s_pred_back.nii.gz' % epoch)
                                save_volume(outputsU_fore_vis[0],
                                            root_path + 'preview/train/Unlabeled/' + 'epoch_%s_pred_fore.nii.gz' % epoch)
                                # save_fore(outputsU_fore_vis[0],
                                #           root_path + 'preview/train/Unlabeled/' + 'epoch_%s_pred_fore_2.nii.gz' % epoch)

                if phase == 'trainLabeled':
                    loss.backward(retain_graph=True)
                    optimizer_seg[0].step()
                    running_loss += loss.item() * inputs.size(0)

                elif phase == 'trainUnlabeled':
                    loss.backward(retain_graph=True)
                    optimizer_seg[1].step()
                    running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss
            # compute loss
            if phase == 'trainLabeled':
                epoch_train_loss[0] += epoch_loss
            elif phase == 'trainUnlabeled':
                epoch_train_loss[1] += epoch_loss

            # compute validation accuracy, update training and validation loss, and calculate DICE and MSE
            if epoch % 20 == 19:
                print('-------val------------')
                if phase == 'val_labeled':
                    running_val_dice, epoch_val_loss[0] = eval_net_dice(model_seg, criterion, phase, network_switch,
                                                                        modelDataLoader['val_labeled'],
                                                                        preview=PREVIEW, gpu=True, visualize_batch=0,
                                                                        epoch=epoch, slice=62, root_path=root_path)
                    epoch_val_dice = running_val_dice
                elif phase == 'val_unlabeled':
                    running_val_mse, epoch_val_loss[1] = eval_net_mse(model_seg, criterion, phase, network_switch,
                                                                      modelDataLoader['val_unlabeled'],
                                                                      preview=PREVIEW, gpu=True, visualize_batch=0,
                                                                      epoch=epoch, slice=62, root_path=root_path)
                    epoch_val_mse = running_val_mse

            # # display TQDM information
            # tqiter.set_description('MASSL (TSL=%.4f, TUL=%.4f, VSL=%.4f, VUL=%.4f, vdice=%.4f, vmse=%.4f)'
            #                        % (epoch_train_loss[0] / (i + 1), epoch_train_loss[1] / (i + 1), epoch_val_loss[0],
            #                           epoch_val_loss[1],
            #                           epoch_val_dice, epoch_val_mse))
            print('MASSL (epoch = %d,TSL=%.4f, TUL=%.4f, VSL=%.4f, VUL=%.4f, vdice=%.4f, vmse=%.4f)'
                  % (epoch, epoch_train_loss[0] / (i + 1), epoch_train_loss[1] / (i + 1), epoch_val_loss[0],
                     epoch_val_loss[1],
                     epoch_val_dice, epoch_val_mse))

            # save and visualize training information
            if phase == 'val_unlabeled':
                if epoch == 0:
                    title = 'Epoch   Train_L_loss   Train_U_loss   Val_L_loss   Val_U_loss   Val_dice   Val_MSE   ' \
                            'best_epoch\n'
                    cm.history_log(root_path + 'history_log.txt', title, 'w')
                    history = (
                        '{:3d}        {:.4f}         {:.4f}        {:.4f}       {:.4f}      {:.4f}     {:.4f}       {:d}\n'
                            .format(epoch, epoch_train_loss[0] / (i + 1), epoch_train_loss[1] / (i + 1),
                                    epoch_val_loss[0],
                                    epoch_val_loss[1], epoch_val_dice, epoch_val_mse, best_epoch))
                    cm.history_log(root_path + 'history_log.txt', history, 'a')

                    title = title.split()
                    data = history.split()
                    for ii, key in enumerate(title):
                        if ii == 0:
                            dict[key].append(int(data[ii]))
                        else:
                            dict[key].append(float(data[ii]))
                    visualize_loss(fig_loss, dict=dict, title=title, epoch=epoch)
                    plt.savefig(root_path + 'Log.jpg')
                    plt.close(fig_loss)

                else:
                    title = 'Epoch   Train_L_loss   Train_U_loss   Val_L_loss   Val_U_loss   Val_dice   Val_MSE   ' \
                            'best_epoch\n'
                    history = (
                        '{:3d}        {:.4f}         {:.4f}        {:.4f}       {:.4f}      {:.4f}     {:.4f}       {:d}\n'
                        .format(epoch, epoch_train_loss[0] / (i + 1), epoch_train_loss[1] / (i + 1), epoch_val_loss[0],
                                epoch_val_loss[1], epoch_val_dice, epoch_val_mse, best_epoch))
                    cm.history_log(root_path + 'history_log.txt', history, 'a')

                    title = title.split()
                    data = history.split()
                    for ii, key in enumerate(title):
                        if ii == 0:
                            dict[key].append(int(data[ii]))
                        else:
                            dict[key].append(float(data[ii]))
                    visualize_loss(fig_loss, dict=dict, title=title, epoch=epoch)
                    plt.savefig(root_path + 'Log.jpg')
                    plt.close(fig_loss)

            # save best validation model, figure preview and dice
            if phase == 'val_labeled' and (epoch_val_dice > best_val_dice):
                best_epoch = epoch
                best_val_dice = epoch_val_dice
                best_model_wts = copy.deepcopy(model_seg.state_dict())
                torch.save(model_seg.state_dict(), root_path + 'model/val_unet.pth')

            if epoch % 200 == 199 and best_val_dice < 0.1:
                model_seg.apply(ssl_3d_attention.weights_init)
    meters = defaultdict(AverageMeter)
    if epoch% 5 ==0:
        model_reg.train()

        for batch_id, (batch_fix, fix_label, batch_move, move_label, fix_mask, move_mask) in enumerate(
                modelDataLoader['train_reg']):
            batch_fix, batch_move = batch_fix.cuda(async=True), batch_move.cuda(async=True)
            fix_mask = fix_mask.cuda(async=True)
            batch_warp, batch_df_grid, batch_flow, batch_affine, batch_affine_grid = model_reg(batch_fix,
                                                                                               batch_move)
            phase = 'trainLabeled'
            outputsL_fix, outputsU_fix,out_lever_fix= model_seg(batch_fix, phase=phase, network_switch=network_switch)
            outputsL_move, outputsU_move,out_liver_move = model_seg(batch_move, phase=phase, network_switch=network_switch)

            mask_move = np.zeros_like(outputsL_move.cpu().detach().numpy())
            mask_move[outputsL_move.cpu().detach().numpy() > 0.5] = 1
            mask_fix = np.zeros_like(outputsL_fix.cpu().detach().numpy())
            mask_fix[outputsL_fix.cpu().detach().numpy() > 0.5] = 1

            dice_score = dice_coef(outputsL_fix,fix_mask)

            batch_affine_label = F.grid_sample(torch.from_numpy(mask_move).cuda(), batch_affine_grid,
                                               mode='nearest')
            batch_warp_label = F.grid_sample(batch_affine_label, batch_df_grid, mode='nearest')
            sim_loss_att = losses['one_ncc'](batch_fix * torch.from_numpy(mask_fix).cuda(),
                                             batch_warp * batch_warp_label)

            affine_loss = losses['one_ncc'](batch_fix, batch_affine)
            smooth_loss = losses['smooth'](batch_flow)
            sim_loss = losses['one_ncc'](batch_fix, batch_warp)

            loss = 1000.0 * smooth_loss + affine_loss + sim_loss + sim_loss_att + dice_score
            meters['loss'].update(loss.item())
            meters['affine_loss'].update(affine_loss.item())
            meters['smooth'].update(smooth_loss.item() * 1000)
            meters['sim_loss'].update(sim_loss.item())
            meters['sim_loss_att'].update(sim_loss_att.item())
            meters['sim_dice'].update(dice_score.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            str_list = ['%s: %8.5f' % (item[0], item[1].avg) for item in meters.items()]
            print('Epoch %d ' % epoch,
                  'Batch %d|%d  ' % (batch_id, len(modelDataLoader['train_reg'])),
                  'dice score %f' % (dice_score),
                  ' || '.join(str_list), )
            train_log = ['Epoch %d ' % epoch,
                      'Batch %d|%d  ' % (batch_id, len(modelDataLoader['train_reg'])),
                      ' || '.join(str_list),
                      'Time: %s' % time_to_str((timer() - since), 'min')]
            open(os.path.join(cfg.log_dir, 'train_log.txt'), 'a').write(str(train_log) + '\n')

    return meters

def save_flow(flow, name):
    flow =flow.cpu().numpy()
    flow =flow.astype(np.float32)
    volume = nib.Nifti1Image(np.squeeze(flow), affine=None)
    nib.save(volume, name)
def eval_one_epoch(model, loader, metrics, epoch, save_res=True):
    n_batches = len(loader['eval_reg'])
    since = timer()
    model.eval()
    meters = defaultdict(AverageMeter)
    save_dir = os.path.join(cfg.log_dir, 'epoch'+ str(epoch))
    with torch.no_grad():
        for batch_id, (batch_fix, fix_label, batch_move, move_label,fix_mask,move_mask) in enumerate(loader['eval_reg']):
            batch_fix, batch_move = batch_fix.cuda(async=True), batch_move.cuda(async=True)
            batch_warp, batch_df_grid, batch_flow, batch_affine, batch_affine_grid = model(batch_fix, batch_move)
            batch_move_label = move_label.cuda()

            batch_affine_label = F.grid_sample(batch_move_label, batch_affine_grid, mode='nearest')
            batch_warp_label = F.grid_sample(batch_affine_label, batch_df_grid, mode='nearest')
            warp_label = torch.squeeze(batch_warp_label.cpu())
            affine_label = torch.squeeze(batch_affine_label.cpu())
            metric_fn = metrics['sub_dice']
            fix_label = torch.squeeze(fix_label)
            move_label = torch.squeeze(move_label)
            sub_dice = metric_fn(warp_label.numpy(), fix_label.numpy(), cfg.label)

            mean_dice = [d for d in sub_dice.values()]
            mean_dice = np.mean(mean_dice)
            meters['dice'].update(mean_dice)
            str_list = ['%s: %8.5f' % (item[0], item[1].avg) for item in meters.items()]
            print('Epoch %d ' % epoch,
                  'Batch %d|%d  ' % (batch_id, n_batches),
                  ' || '.join(str_list),

                  'Time: %s' % time_to_str((timer() - since), 'min'))
            if save_res:
                save_volume(batch_fix.cpu().numpy(), os.path.join(save_dir, str(batch_id) + "_fix.nii.gz"))
                save_volume(batch_move.cpu().numpy(), os.path.join(save_dir, str(batch_id) + "_move.nii.gz"))
                save_volume(batch_warp.cpu().numpy(), os.path.join(save_dir, str(batch_id) + "_warp.nii.gz"))
                save_volume(batch_affine.cpu().numpy(), os.path.join(save_dir, str(batch_id) + "_affine.nii.gz"))
                save_label(fix_label.numpy(), os.path.join(save_dir, str(batch_id) + "_fix_label.nii.gz"))
                save_label(affine_label.numpy(), os.path.join(save_dir, str(batch_id) + "_affine_label.nii.gz"))
                save_label(move_label.numpy(), os.path.join(save_dir, str(batch_id) + "_move_label.nii.gz"))
                save_label(warp_label.numpy(), os.path.join(save_dir, str(batch_id) + "_warp_label.nii.gz"))

                # if epoch%20 == 0:
                #     save_flow(batch_affine_grid, os.path.join(save_dir, str(batch_id) + "_affinr_grid.nii.gz"))
                #     save_flow(batch_flow, os.path.join(save_dir, str(batch_id) + "_flow.nii.gz"))
                #     save_flow(batch_df_grid, os.path.join(save_dir, str(batch_id) + "_def_grid.nii.gz"))

    return meters

def save_volume(batch_x, name):
    volume = np.squeeze(batch_x) * 255
    volume = volume.astype("uint8")
    volume = nib.Nifti1Image(volume, np.eye(4))
    nib.save(volume, name)

def save_lever(batch_x, name):

    volume = nib.Nifti1Image(batch_x, np.eye(4))
    nib.save(volume, name)


def save_label(pred_mask, name):
    # pred_mask = pred_mask.astype("uint8")

    volume = nib.Nifti1Image(pred_mask, np.eye(4))
    nib.save(volume, name)
def save_mask(pred_mask, name):
    # pred_mask = pred_mask.astype("uint8")
    pred = np.zeros_like(pred_mask)
    pred[pred_mask > 0.5] = 1
    volume = nib.Nifti1Image(pred, np.eye(4))
    nib.save(volume, name)


def train_eval(model_reg, loaders, optimizer, scheduler, losses, metrics):
    start = timer()
    train_meters = defaultdict(list)
    val_meters = defaultdict(list)

    val_dice = 0
    test_results = 0

    print('-' * 64)
    print('Training start')
    job = 'MASSL_alter'
    folder_name = 'results_abdmon_lever/'
    data_split =  '10L100U'
    data_seed = 1
    basic_path = folder_name + str(job) + '/' + str(data_split)[:]
    print('basice_path', basic_path)

    #################################################

    if job == 'MASSL_alter':
        print('MASSL_alter')

        switch = {'trainL_encoder': True,
                  'trainL_decoder_seg': True,
                  'trainL_decoder_rec': False,

                  'trainU_encoder': True,
                  'trainU_decoder_seg': False,
                  'trainU_decoder_rec': True}

        root_path = basic_path + '/seed' + str(data_seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        cm.mkdir(root_path + 'preview/train/Unlabeled')

        base_features = 16
        #model_reg = WNet3D(use_dialte=True).cuda()
        model_seg = ssl_3d_attention.MASSL_norm(1, 1, base_features).to(device)

        use_existing = False

        if use_existing:
            model_seg.load_state_dict(torch.load(root_path + 'model/val_unet.pth'))


        criterionDICE = DiceCoefficientLF(device)
        criterionMSE = torch.nn.MSELoss()
        criterion = (criterionDICE, criterionMSE)

        optimizer_ft = (torch.optim.Adam(model_seg.parameters(), lr=1e-2),
                        torch.optim.Adam(model_seg.parameters(), lr=1e-3),
                        torch.optim.Adam(model_seg.parameters(), lr=1e-3),
                       )

        exp_lr_scheduler = (torch.optim.lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                            torch.optim.lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5),
                            torch.optim.lr_scheduler.StepLR(optimizer_ft[2], step_size=500, gamma=0.5))

        # save training information
        train_info = (
            'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\n'
            'base features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
            'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                job,
                data_seed,
                data_split,
                10,
                type(model_seg),
                base_features,
                switch,
                criterion,
                optimizer_ft,
                exp_lr_scheduler))

        cm.history_log(root_path + 'info.txt', train_info, 'w')

        print('data random seed: ', data_seed)
        print('device: ', device)

        print('-' * 64)

        print('MASSL_alter finished')
    for epoch in range(0, cfg.epoch):
        scheduler.step(epoch)
        mkdir(os.path.join(cfg.log_dir, 'epoch'+str(epoch)))
        cur_lr = get_learning_rate(optimizer)
        print('Learning rate is ', cur_lr)
        meter = train_one_peoch(model_reg, loaders, optimizer, losses, metrics, epoch, start, model_seg , device, root_path,
                                switch, criterion, optimizer_ft,exp_lr_scheduler, num_epochs=500, loss_weighted=True)
        train_meters['loss'].append(meter['loss'].avg)
        if epoch%5==0:
            meter = eval_one_epoch(model_reg, loaders, metrics, epoch)
            val_meters['dice'].append(meter['dice'].avg)
        file_name = os.path.join(cfg.log_dir, 'epoch'+str(epoch), "epoch_{}.pth".format(epoch))
        state = {"model": model_reg.state_dict(), "optimizer": optimizer.state_dict()}
        if np.argmax(val_meters['dice']) == epoch:
            save_checkpoint(state, True, file_name)
        else:
            save_checkpoint(state, False, file_name)

def train_baseline():
    task_name = "train_lever_dice"
    cfg.log_dir = os.path.join(cfg.log_dir, task_name)
    mkdir(cfg.log_dir)
    set_logger(os.path.join(cfg.log_dir, task_name + '.log'))
    device, data_sizes, modelDataloader = abdomen_data(0, '10L100U')
    loaders = modelDataloader
    model = WNet3D(use_dialte=True).cuda()
    lr = cfg.lr
    print('initial learning rate is ', lr)
    optimizer = torch.optim.Adam(filter(requires_grad,model.parameters()), lr=lr, weight_decay=1e-5)
    losses = {'patch_ncc': patch_ncc_loss, 'smooth': l2_smooth3d, 'rc': residual_complexity_loss, 'one_ncc': ncc_loss,}
    metrics = {'sub_dice': parse_lable_subpopulation,'iou_dice':IOU_subpopulation}
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=1)
    train_eval(model, loaders, optimizer, scheduler, losses, metrics=metrics)


if __name__ == '__main__':
    init_env('3')
    cfg = FubuConfig()
    train_baseline()