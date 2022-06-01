#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/3/27 10:18
# @Author  : Eric Ching
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def dice_coef(input, target, threshold=0.5):
    smooth = 1.
    iflat = (input.view(-1) > threshold).float()
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def dice_coef_np(input, target, eps=1e-7):
    input = np.ravel(input)
    target = np.ravel(target)
    intersection = (input * target).sum()

    return (2. * intersection) / (input.sum() + target.sum() + eps)

def parse_lable_subpopulation(input, target, label_dict):
    sub_dice = {}
    for label_id, label_name in label_dict.items():
        lid = int(label_id)
        if lid == 0 :
            continue
        sub_input = input == lid
        sub_target = target == lid
        dsc = dice_coef_np(sub_input, sub_target)
        sub_dice[label_id] = dsc

    return sub_dice

def parse_subpopulation(input, target, label_dict):
    sub_dice = {}

    for lid in range(55):
        if lid >= 1:

            sub_input = input == lid
            sub_target = target == lid
            dsc = dice_coef_np(sub_input, sub_target)
            sub_dice[lid] = dsc

    return sub_dice

all_label = [label_F,label_P,label_O,label_T,label_C]
def IOU_subpopulation(input,target,all_label):
    vol1 = np.zeros((192, 192, 192))
    vol2 = np.zeros((192, 192, 192))
    fina_dice=[]
    for i,j in enumerate(all_label):
        #print(i,j)
        fenzi = 0.0
        fenmu = 0.0
        for label_id,label_name in enumerate(j):

            lid = int(label_name)
            if lid == 0 :
                continue
            sub_input = input == lid
            sub_target = target == lid
            fenzi_sub= (np.ravel(sub_input)*np.ravel(sub_target)).sum()
            fenmu_sub= sub_input.sum()+sub_target.sum()
            fenzi += fenzi_sub
            fenmu += fenmu_sub
        dice = (2. * fenzi) / ( fenmu + 1e-7)
        fina_dice.append(dice)
    return fina_dice

#左右dice合并
def averagenum(num):
    nsum =0.0
    for i in range(len(num)):
        nsum += float(num[i])
    return nsum/len(num)


import SimpleITK as sitk
def hausdorff_distance(predictions, labels, one_hot=False, unindexed_classes=0, spacing=[1, 1, 1]):
    def one_class_hausdorff_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred[i].cpu().numpy())
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(lab[i].cpu().numpy())
            lab_img.SetSpacing(spacing)
            hausdorff_distance_filter.Execute(pred_img, lab_img)
            result.append(hausdorff_distance_filter.GetHausdorffDistance())
        return torch.tensor(np.asarray(result))

    return multi_class_score(one_class_hausdorff_distance, predictions, labels, one_hot=one_hot,
                             unindexed_classes=unindexed_classes)

def multi_class_score(one_class_fn, predictions, labels, one_hot=False, unindexed_classes=0):
    result = {}
    shape = labels.shape
    for label_index in range(shape[1] + unindexed_classes):
        if one_hot:
            class_predictions = torch.round(predictions[:, label_index, :, :, :])
        else:
            class_predictions = predictions.eq(label_index)
            class_predictions = class_predictions.squeeze(1)  # remove channel dim
        class_labels = labels.eq(label_index).float()
        class_labels = class_labels.squeeze(1)  # remove channel dim
        class_predictions = class_predictions.float()

        result[str(label_index)] = one_class_fn(class_predictions, class_labels).mean()

    return result

def computeQualityMeasures(lP, lT):
    quality = dict()
    labelPred = sitk.GetImageFromArray(lP, isVector=False)
    labelTrue = sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer = sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue > 1, labelPred > 1)
    quality["avgHausdorff"] = hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"] = hausdorffcomputer.GetHausdorffDistance()

    # dicecomputer = sitk.LabelOverlapMeasuresImageFilter()
    # dicecomputer.Execute(labelTrue > 0.5, labelPred > 0.5)
    # quality["dice"] = dicecomputer.GetDiceCoefficient()
    return quality

def get_metrics():
    metrics = {}
    metrics["mse"] = nn.MSELoss().cuda()
    metrics["l1"] = nn.L1Loss().cuda()
    # metrics["psnr"] = PSNR
    # metrics['ssim'] = ssim
    metrics['dice'] = dice_coef
    metrics['sub_dice'] = parse_lable_subpopulation
    metrics['iou_dice'] = IOU_subpopulation
    metrics['hd'] = computeQualityMeasures

    return metrics