# 2020/5/27 deformable aval
from config import MindboggleConfig

import os
import glob
from medpy import metric

from metrics import parse_lable_subpopulation,IOU_subpopulation,dice_coef_np
import numpy as np
import nibabel as nib
import SimpleITK as sitk



all_label = {"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"11":11,"12":12,"13":13}
def read_volume(path):
    itk_volume = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(itk_volume)
    spacing = itk_volume.GetSpacing()
    return volume


def IOU_sub_hd(input,target,all_label):
    fina_hd=[]
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
            hd = metric.hd(sub_input,sub_target)
        fina_hd.append(hd)
    return fina_hd
#dilate6

root_data = 'Y:/reg_and_semi_seg/30 label\epoch380'
#root_data ='Y:/Desktop/python_code/reg3d/logs/Mind/baseline_mindboggle_10000/epoch40'
fix_label_list = sorted(glob.glob(os.path.join(root_data,'*fix_label.nii.gz')))
move_label_list = sorted(glob.glob(os.path.join(root_data,'*move_label.nii.gz')))
affine_label_list = sorted(glob.glob(os.path.join(root_data,'*affine_label.nii.gz')))
warp_label_list = sorted(glob.glob(os.path.join(root_data,'*warp_label.nii.gz')))
print(len(move_label_list))
num = 0
sum_dice = np.zeros((90,13))

for path in zip(fix_label_list[0:90],move_label_list[0:90],warp_label_list[0:90]):

    fix = read_volume(path[0])
    move = read_volume(path[1])
    warp = read_volume(path[2])
    #sub_dice = IOU_subpopulation(fix,move,all_label)\

    #before_hd = IOU_sub_hd(fix,move,all_label)
    #after_hd = parse_lable_subpopulation(fix, warp, all_label)
    after_hd = IOU_sub_hd(fix, move, all_label)


    #print(before_hd)
    print(after_hd)
    #mean_dice = [d for d in after_hd.values()]

    sum_dice[num] = after_hd
    num +=1
mean_dice = np.mean(sum_dice, axis=0)
print('---mean dice----')
print(mean_dice)

print(np.std(sum_dice,axis=0,ddof =1 ))
print(np.std(sum_dice,ddof= 0))
print('---mean hd----')





