import os
import glob
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from skimage.filters import gaussian
def read_volume(path):
    itk_volume = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(itk_volume)
    spacing = itk_volume.GetSpacing()

    return volume

def normlize(x):
    return (x - x.min()) / (x.max() - x.min())
slice =113
root_fix = 'Y:/reg_and_semi_seg/abdomen_wnet/epoch133/4_fix.nii.gz'
root_move = 'Y:/reg_and_semi_seg/abdomen_wnet/epoch133/4_move.nii.gz'
root_VM = 'Y:/reg_and_semi_seg/abdomen_VM/epoch222/4_warp.nii.gz'
root_Faim = 'Y:/reg_and_semi_seg/abdomen_mutifc/epoch108/4_warp.nii.gz'
root_mutifc = 'Y:/reg_and_semi_seg/abdomen_mutifc/epoch107/4_warp.nii.gz'
root_wnet = 'Y:/reg_and_semi_seg/abdomen_wnet/epoch133/4_warp.nii.gz'
root_wnet_att = 'Y:/reg_and_semi_seg/abdomen_wnet_att/epoch100/4_warp.nii.gz'
volume_fix = read_volume(root_fix)
volume_move = read_volume(root_move)
volume_VM = read_volume(root_VM)
volume_Faim = read_volume(root_Faim)
volume_mutifc = read_volume(root_mutifc)
volume_wnet = read_volume(root_wnet)
volume_wnet_att = read_volume(root_wnet_att)
slice_fix =volume_fix[slice]
slice_move = volume_move[slice]
slice_VM = volume_VM[slice]
slice_Faim = volume_Faim[slice]
slice_mutifc = volume_mutifc[slice]
slice_wnet =volume_wnet[slice]
slice_wnet_att = volume_wnet_att[slice]
muti_channel = True
fix_slice = normlize(np.flip(volume_fix[slice]))
fix_slice_filter = gaussian(fix_slice,sigma=1,mode='constant',preserve_range=False)

syn_slice = normlize(np.flip(volume_move[slice]))
syn_slice_match = match_histograms(syn_slice, fix_slice, multichannel=muti_channel)
syn_slice_filter = gaussian(syn_slice,sigma=1,mode='constant',preserve_range=True)

vm_slice = normlize(np.flip(volume_VM[slice]))
vm_slice_match =match_histograms(vm_slice, fix_slice, multichannel=muti_channel)
vm_slice_filter = gaussian(vm_slice, sigma=1, mode='constant', preserve_range=True)

faim_slice = normlize(np.flip(volume_Faim[slice]))
faim_slice_match = match_histograms(faim_slice, fix_slice, multichannel=muti_channel)
faim_slice_filter = gaussian(faim_slice, sigma=1, mode='constant', preserve_range=True)

muti_slice = normlize(np.flip(volume_mutifc[slice]))
muti_slice_match = match_histograms(muti_slice, fix_slice, multichannel=muti_channel)
muti_slice_filter = gaussian(muti_slice, sigma=1, mode='constant', preserve_range=True)

wnet_slice = normlize(np.flip(volume_wnet[slice]))
wnet_slice_match = match_histograms(wnet_slice, fix_slice, multichannel=muti_channel)
wnet_slice_filter = gaussian(wnet_slice, sigma=1, mode='constant', preserve_range=True)

our_slice = normlize(np.flip(volume_wnet_att[slice]))
our_slice_match = match_histograms(our_slice, fix_slice, multichannel=muti_channel)
our_slice_filter = gaussian(our_slice, sigma=1,mode='constant',preserve_range=True)
write_root = './visual/'
cmap = 'gray' #bwr_r
plt.imshow(fix_slice, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(write_root + '/fix_' + str(slice) + '.png', bbox_inches='tight')
plt.show()
plt.imshow(syn_slice, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(write_root + '/move_' + str(slice) + '.png', bbox_inches='tight')
plt.show()
plt.imshow(vm_slice, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(write_root + '/vm_' + str(slice) + '.png', bbox_inches='tight')
plt.show()
plt.imshow(faim_slice, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(write_root + '/faim_' + str(slice) + '.png', bbox_inches='tight')
plt.show()
plt.imshow(muti_slice, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(write_root + '/muti_' + str(slice) + '.png', bbox_inches='tight')
plt.show()
plt.imshow(wnet_slice, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(write_root + '/wnet_' + str(slice) + '.png', bbox_inches='tight')
plt.show()
plt.imshow(our_slice, cmap=cmap)
plt.xticks([])
plt.yticks([])
plt.savefig(write_root + '/ours_' + str(slice) + '.png', bbox_inches='tight')
plt.show()




# vm = vm_slice - fix_slice
# plt.show()
# plt.xticks([])
# plt.yticks([])
# plt.imshow(normlize(vm), cmap=cmap)
# plt.savefig(write_root + '/diff_VM_' + str(slice) + '.png', bbox_inches='tight')
# plt.show()
# faim = faim_slice - fix_slice
# plt.imshow(normlize(faim), cmap=cmap)
# plt.xticks([])
# plt.yticks([])
# plt.savefig(write_root + '/diff_Faim_' + str(slice) + '.png', bbox_inches='tight')
# plt.show()
# muti = muti_slice - fix_slice
# plt.imshow(normlize(muti), cmap=cmap)
# plt.xticks([])
# plt.yticks([])
# plt.savefig(write_root + '/diff_Mutifc_' + str(slice) + '.png', bbox_inches='tight')
# plt.show()
# SYM = wnet_slice - fix_slice
# plt.imshow(normlize(SYM), cmap=cmap)
# plt.xticks([])
# plt.yticks([])
# plt.savefig(write_root + '/diff_wnet' + str(slice) + '.png', bbox_inches='tight')
# plt.show()
# ours = our_slice - fix_slice
# plt.imshow(normlize(ours), cmap=cmap)
# plt.xticks([])
# plt.yticks([])
# plt.savefig(write_root + '/diff_Wnet_att' + str(slice) + '.png', bbox_inches='tight')
# plt.show()


