#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2019/3/27 10:17
# @Author  : Eric Ching
import torch
from typing import List
from torch.nn import functional as F
from torch.nn import KLDivLoss
import numpy as np
from model import dct
from torch import nn as nn
import torch
import SimpleITK as sitk
def dice_loss(input, target):

    """soft dice loss"""
    eps = 1e-7
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - (2. * intersection / ((iflat**2).sum() + (tflat**2).sum() + eps))

# def dice_loss_multi_class(input, target):
#     loss = 0.
#     class_weights = [1.0, 1.0, 1.0, 1.0]
#     n_classes = len(class_weights)
#
#     for c in range(n_classes):
#         inputc = input[:, c]
#         targetc = target[:, c]
#         loss += class_weights[c] * dice_loss(inputc, targetc)
#
#     return loss

def dice_coef(input, target, threshold=0.5):
    smooth =  1e-7
    demo = input[0][0][80]
    print(demo)
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1-(2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

def dice_new_coef(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    input_sum = (iflat * iflat).sum()
    target_sum = (tflat * tflat).sum()

    a = (2. * intersection + smooth).float()
    b = (input_sum + target_sum + smooth).float()  ##整型除以整型，小于1会自动归零，所以浮点型弄错
    c = a / b

    return 1.-c


def muti_dice(input,target,label_dict):
    sub_dice = 0
    print(target.grad_fn)
    demo = torch.ones([1, 1, 160, 192, 224], dtype=torch.float,requires_grad=True).cuda()
    demo.grad_fn = target.grad_fn

    #demo =target
    for label_id, label_name in label_dict.items():
        lid = int(label_id)
        if lid == 0 or lid == 181 or lid == 182:
            continue
        sub_input = input == lid
        sub_target = target == lid
        demo.copysub_target.int()
        demo.grad_fn = target.grad_fn
        # dsc = dice_new_coef(sub_input, sub_target)
        # sub_dice += dsc

    return sub_dice / 54.



def vae_loss(recon_x, x, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    L2 = F.mse_loss(recon_x, x, reduction='mean')

    return KLD + L2

def patch_ncc_loss(I, J, win=(9, 9, 9), eps=1e-5):
    # compute CC squares
    ndims = len(I.size()) - 2
    I2 = I * I
    J2 = J * J
    IJ = I * J
    conv_fn = getattr(F, 'conv%dd' % ndims)
    sum_filt = torch.ones([1, 1, *win], dtype=torch.float).cuda()
    strides = [1] * ndims
    I_sum = conv_fn(I, sum_filt, stride=strides, padding=win[0]//2)
    J_sum = conv_fn(J, sum_filt, stride=strides, padding=win[0] // 2)
    I2_sum = conv_fn(I2, sum_filt, stride=strides, padding=win[0] // 2)
    J2_sum = conv_fn(J2, sum_filt, stride=strides, padding=win[0] // 2)
    IJ_sum = conv_fn(IJ, sum_filt, stride=strides, padding=win[0] // 2)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    cc = cross * cross / (I_var * J_var + eps)

    return -torch.mean(cc)

def ncc_loss(I, J):

    mean_I = I.mean([1, 2, 3, 4], keepdim=True)
    mean_J = J.mean([1, 2, 3, 4], keepdim=True)
    I2 = I * I
    J2 = J * J

    mean_I2 = I2.mean([1, 2, 3, 4], keepdim=True)
    mean_J2 = J2.mean([1, 2, 3, 4], keepdim=True)

    stddev_I = torch.sqrt(mean_I2 - mean_I * mean_I).sum([1, 2, 3, 4], keepdim=True)
    stddev_J = torch.sqrt(mean_J2 - mean_J * mean_J).sum([1, 2, 3, 4], keepdim=True)

    return -torch.mean((I - mean_I) * (J - mean_J) / (stddev_I * stddev_J))

def l1_smooth3d(flow):
    """computes TV loss over entire composed image since gradient will
     not be passed backward to input
    计算图像梯度平均值
    Args:
        flow: 5d tensor, [batch, height, width, depth, channel(translation)]
    """
    loss = torch.mean(torch.mean(torch.abs(flow[:, 1:, :,  :, :] - flow[:, :-1, :,   :, :])) +
                      torch.mean(torch.abs(flow[:, :, 1:,  :, :] - flow[:, :,   :-1, :, :])) +
                      torch.mean(torch.abs(flow[:, :,  :, 1:, :] - flow[:, :,   :,   :-1, :])))
    return loss


def l2_smooth3d(flow):
    """computes TV loss over entire composed image since gradient will
     not be passed backward to input
    计算图像梯度平均值
    Args:
        flow: 5d tensor, [batch, height, width, depth, channel(translation)]
    """
    loss = torch.mean(torch.mean(torch.pow(flow[:, 1:, :, :, :] - flow[:, :-1, :, :, :], 2)) +
                      torch.mean(torch.pow(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :], 2)) +
                      torch.mean(torch.pow(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :], 2)))

    return loss

def residual_complexity_loss(mov, fix, alpha=0.05):
    bs, c, d, h, w = mov.size()
    rbig = mov - fix
    rbig = rbig.view([bs, d, h, w])
    qr = dct.dct(rbig)
    li = qr * qr + alpha
    f = 0.5 * torch.log(li.view([bs, -1]) / alpha).mean()
    # r = dct.idct(qr / li)

    return f

def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match
    n_classes = 55


    target = expand_as_one_hot(target, C=n_classes)
    input = expand_as_one_hot(input, C=n_classes)
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = target.clone().ne_(ignore_index)
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)

    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)



    denominator = (input + target).sum(-1)

    return 2. * intersect / denominator.clamp(min=epsilon)


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            print('True')
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits

        # input = self.normalization(input)

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    )

        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice[1:])


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

def expand_as_one_hot(input, C):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """

    assert input.dim() == 5

    # expand the input tensor to Nx1xDxHxW before scattering

    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    print('in',input.long())

    # scatter to get the one-hot tensor
    return torch.zeros(shape).to(input.device).scatter_(1, input.long(), 1)






def dice_loss_multi_class(input, target):

    target = expand_as_one_hot(target, C=55)
    print(target)
    input = expand_as_one_hot(input, C=55)

    loss = 0.
    result =0
    n_classes = 55
    # input = input.squeeze(1)
    # target = target.squeeze(1)

    for c in range(n_classes):
        if c >= 1:
            inputc = input[0,c,:]
            targetc = target[0,c, :]

            loss += dice_loss(inputc, targetc)
            result = loss/54

    return result



def JSsandu(input, target):
    target = expand_as_one_hot(target, C=55)
    input = expand_as_one_hot(input, C=55)
    n_classes = 55
    input = input.squeeze(1)
    target = target.squeeze(1)
    loss = 0.
    result = 0
    for c in range(n_classes):
        if c >= 1:
            inputc = input[0,c,:]
            targetc = target[0,c, :]

            loss += JS(inputc, targetc)
            result = loss/54

    return result

criterion = nn.KLDivLoss()

def JS(input, target):
    input = input.view(-1).float()
    target = target.view(-1).float()
    input = F.softmax(input)
    input = torch.log(input)
    target = F.softmax(target)

    xxx = criterion(input,target)
    return xxx




def multi_class_score(one_class_fn,predictions, labels, label_dict,one_hot=False, unindexed_classes=0):
    result = 0
    shape = labels.shape

    for label_index, label_name in label_dict.items():
        lid = int(label_index)
        if lid == 0 or lid == 181 or lid == 182:
            continue
        if one_hot:
            class_predictions = torch.round(predictions[:, label_index, :, :, :])
        else:

            class_predictions = predictions == lid
            class_predictions = class_predictions.squeeze(1)  # remove channel dim

        class_labels = labels == lid
        class_labels = class_labels.float()
        class_labels = class_labels.squeeze(1)  # remove channel dim
        class_predictions = class_predictions.float()
        jieguo = one_class_fn(class_predictions, class_labels).mean()
        result += jieguo


    return (result/54.).float()

def average_surface_distance(predictions, labels,label_dict, one_hot=False, unindexed_classes=0, spacing=[1, 1, 1]):
    def one_class_average_surface_distance(pred, lab):
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        batch = pred.shape[0]
        result = []
        for i in range(batch):
            pred_img = sitk.GetImageFromArray(pred[i].cpu().numpy())
            pred_img.SetSpacing(spacing)
            lab_img = sitk.GetImageFromArray(lab[i].cpu().numpy())
            lab_img.SetSpacing(spacing)
            hausdorff_distance_filter.Execute(pred_img, lab_img)

            result.append(hausdorff_distance_filter.GetAverageHausdorffDistance())
        return torch.Tensor(np.asarray(result))

    return multi_class_score(one_class_average_surface_distance, predictions, labels, one_hot=one_hot,
                             unindexed_classes=unindexed_classes,label_dict =label_dict )



class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")


    def __call__(self, probs, dist_maps):
        # assert simplex(probs)
        # assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = torch.einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss
class BDLoss(nn.Module):
    def __init__(self):
        """
        compute boudary loss
        only compute the loss of foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # self.do_bg = do_bg

    def forward(self, net_output, bound):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """
        #net_output = softmax_helper(net_output)

        pc = net_output[:, :, ...].type(torch.float32)
        dc = bound[:,:, ...].type(torch.float32)


        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc)

        bd_loss = multipled.mean()

        return bd_loss



def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def get_losses():
    losses = {}
    losses['vae'] = vae_loss
    losses['dice'] = dice_loss
    losses['ncc'] = ncc_loss
    losses['patch_ncc'] = patch_ncc_loss
    losses['rc'] = residual_complexity_loss
    losses['l1_smooth'] = l1_smooth3d
    losses['l2_smooth'] = l2_smooth3d

    return losses
