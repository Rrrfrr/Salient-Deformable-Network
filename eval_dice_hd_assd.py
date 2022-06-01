# 2020/5/27 eval
import os
import glob
from medpy import metric
from xml.dom import minidom
from metrics import parse_lable_subpopulation, IOU_subpopulation, dice_coef_np
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import csv


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('exist path: ', path)


def dice_coef_np(input, target, eps=1e-7):
    input = np.ravel(input)
    target = np.ravel(target)
    intersection = (input * target).sum()
    return (2. * intersection) / (input.sum() + target.sum() + eps)


def read_volume(path):
    itk_volume = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(itk_volume)
    return volume


def list_to_csv(dct, csv_path):
    with open(csv_path, 'a') as f:  # Just use 'w' mode in 3.x
        w = csv.writer(f)
        w.writerow(dct)


def IOU_sub_dice(input, target, label_dict):
    sub_dice = {}
    for label_id, label_name in label_dict.items():
        lid = int(label_id)
        if lid == 0:
            continue
        sub_input = input == lid
        sub_target = target == lid
        dsc = dice_coef_np(sub_input, sub_target)
        sub_dice[label_id] = dsc
    return sub_dice


def IOU_sub_hd(input, target, label_dict):
    sub_dice = {}
    for label_id, label_name in label_dict.items():
        lid = int(label_id)
        if lid == 0:
            continue
        sub_input = input == lid
        sub_target = target == lid
        dsc = metric.hd(sub_input, sub_target)
        sub_dice[label_id] = dsc
    return sub_dice


def IOU_sub_assd(input, target, label_dict):
    sub_dice = {}
    for label_id, label_name in label_dict.items():
        lid = int(label_id)
        if lid == 0:
            continue
        sub_input = input == lid
        sub_target = target == lid
        dsc = metric.assd(sub_input, sub_target)
        sub_dice[label_id] = dsc
    return sub_dice


if __name__ == '__main__':
    label_dict = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "11": 11, "12": 12, "13": 13}
    # root_data = '/raid/raoyi/raid/raoyi/join_reg_and_seg/logs/Fubu/MASSL_weakly/1L49U/pred_label_ncc_and_binary_dice/epoch495'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/wnet_weakly_global/epoch149'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/vm_weakly_global/epoch60'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/vm_weakly_global_and_local/epoch60'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/vm2/epoch48'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/vm2_weakly_global/epoch27'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/vm2_weakly_global_and_local/epoch65'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/wnet_weakly_global_and_local/epoch86'
    # root_data = '/raid/raoyi/raid/raoyi/join_reg_and_seg/logs/Fubu/MASSL_weakly/10L40U/pred_label_ncc_and_binary_dice/epoch445'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/wnet_weakly_global_and_local_13/epoch66'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/wnet_weakly_global_and_local_weight13/epoch82'
    # root_data = '/home/raoyi/PycharmProjects/R-abdomen/logs/Fubu/ants_syn90'
    root_data = '/raid/raoyi/raid/raoyi/join_reg_and_seg/logs/Fubu/train_lever_mse_10/epoch45'
    fix_label_list = sorted(glob.glob(os.path.join(root_data, '*fix_label.nii.gz')))
    move_label_list = sorted(glob.glob(os.path.join(root_data, '*move_label.nii.gz')))
    affine_label_list = sorted(glob.glob(os.path.join(root_data, '*affine_label.nii.gz')))
    warp_label_list = sorted(glob.glob(os.path.join(root_data, '*warp_label.nii.gz')))
    print(len(fix_label_list))

    num = 0
    sum_dice, sum_hd, sum_assd = [], [], []
    save_dice_path = root_data + '/dice'
    save_hd_path = root_data + '/hd'
    save_assd_path = root_data + '/assd'
    mkdir(save_dice_path)
    mkdir(save_hd_path)
    mkdir(save_assd_path)
    for path in zip(fix_label_list[0:90], warp_label_list[0:90]):
        fix_label = read_volume(path[0])
        warp_label = read_volume(path[1])

        sub_dice = IOU_sub_dice(fix_label, warp_label, label_dict)
        mean_dice = [d for d in sub_dice.values()]
        print("mean_dice: ", mean_dice)
        sum_dice.append(mean_dice)
        list_to_csv(mean_dice, os.path.join(save_dice_path, str(num) + "_dice.csv"))

        sub_hd = IOU_sub_hd(fix_label, warp_label, label_dict)
        mean_hd = [d for d in sub_hd.values()]
        print("mean_hd: ", mean_hd)
        sum_hd.append(mean_hd)
        list_to_csv(mean_hd, os.path.join(save_hd_path, str(num) + "_hd.csv"))

        sub_assd = IOU_sub_assd(fix_label, warp_label, label_dict)
        mean_assd = [d for d in sub_assd.values()]
        print("mean_assd: ", mean_assd)
        sum_assd.append(mean_assd)
        list_to_csv(mean_assd, os.path.join(save_assd_path, str(num) + "_assd.csv"))

        num += 1

    print('---sum dice----')
    print(len(sum_dice), len(sum_dice[0]))
    print(sum_dice)
    list_to_csv(sum_dice, os.path.join(save_dice_path, "sum_dice.csv"))
    print('---mean dice----')
    mean_dice13 = np.mean(sum_dice, axis=0)  # 求每一列的均值，即90 pairs中的13个器官，每个器官的平均值。
    print(mean_dice13)
    list_to_csv(mean_dice13, os.path.join(save_dice_path, "mean_dice13.csv"))
    print(np.mean(mean_dice13))
    open(os.path.join(save_dice_path, "mean_dice13.txt"), 'a').write(str(np.mean(mean_dice13)) + '\n')
    print('---std dice----')
    std_dice13 = np.std(sum_dice, axis=0, ddof=1)
    print(std_dice13)
    list_to_csv(std_dice13, os.path.join(save_dice_path, "std_dice13.csv"))
    print(np.mean(std_dice13))
    open(os.path.join(save_dice_path, "std_dice13.txt"), 'a').write(str(np.mean(std_dice13)) + '\n')

    print('---sum hd----')
    print(len(sum_hd), len(sum_hd[0]))
    print(sum_hd)
    list_to_csv(sum_hd, os.path.join(save_hd_path, "sum_hd.csv"))
    print('---mean hd----')
    mean_hd13 = np.mean(sum_hd, axis=0)
    print(mean_hd13)
    list_to_csv(mean_hd13, os.path.join(save_hd_path, "mean_hd13.csv"))
    print(np.mean(mean_hd13))
    open(os.path.join(save_hd_path, "mean_hd13.txt"), 'a').write(str(np.mean(mean_hd13)) + '\n')
    print('---std hd----')
    std_hd13 = np.std(sum_hd, axis=0, ddof=1)
    print(std_hd13)
    list_to_csv(std_hd13, os.path.join(save_hd_path, "std_hd13.csv"))
    print(np.mean(std_hd13))
    open(os.path.join(save_hd_path, "std_hd13.txt"), 'a').write(str(np.mean(std_hd13)) + '\n')

    print('---sum assd----')
    print(len(sum_assd), len(sum_assd[0]))
    print(sum_assd)
    list_to_csv(sum_assd, os.path.join(save_assd_path, "sum_assd.csv"))
    print('---mean assd----')
    mean_assd13 = np.mean(sum_assd, axis=0)
    print(mean_assd13)
    list_to_csv(mean_assd13, os.path.join(save_assd_path, "mean_assd13.csv"))
    print(np.mean(mean_assd13))
    open(os.path.join(save_assd_path, "mean_assd13.txt"), 'a').write(str(np.mean(mean_assd13)) + '\n')
    print('---std assd----')
    std_assd13 = np.std(sum_assd, axis=0, ddof=1)
    print(std_assd13)
    list_to_csv(std_assd13, os.path.join(save_assd_path, "std_assd13.csv"))
    print(np.mean(std_assd13))
    open(os.path.join(save_assd_path, "std_assd13.txt"), 'a').write(str(np.mean(std_assd13)) + '\n')