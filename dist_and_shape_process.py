import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.ndimage as ndimage
import os
import nibabel as nib
import glob
import SimpleITK as sitk
def seg_contour_dist_trans(img):
    # one-hot encoding
    img_one_hot = np.eye(14)[np.uint8(img)] > 0.0

    contour = np.uint8(np.zeros(img_one_hot.shape))
    edt = np.zeros(img_one_hot.shape)

    for i in range(1, 14):
        if np.sum(img_one_hot[:, :, i]) != 0:
            # fill holes
            img = ndimage.morphology.binary_fill_holes(img_one_hot[:, :, i])

            # extract contour
            contour[:, :, i] = ndimage.morphology.binary_dilation(img == 0.0) & img

            # distance transform
            tmp = ndimage.morphology.distance_transform_edt(img)
            edt[:, :, i] = (tmp - np.amin(tmp)) / (np.amax(tmp) - np.amin(tmp))

    return np.sum(contour, axis=-1) > 0.0, np.sum(edt, axis=-1)

def seg_contour_dist_trans_volume(img):
    # one-hot encoding
    img_one_hot = np.eye(14)[np.uint8(img)] > 0.0
    print(img_one_hot.shape)

    contour = np.uint8(np.zeros(img_one_hot.shape))
    edt = np.zeros(img_one_hot.shape)

    for i in range(1, 14):
        if np.sum(img_one_hot[:, :, :,i]) != 0:
            # fill holes
            img = ndimage.morphology.binary_fill_holes(img_one_hot[:, :,:, i])

            # extract contour
            contour[:,:, :, i] = ndimage.morphology.binary_dilation(img == 0.0) & img

            # distance transform
            tmp = ndimage.morphology.distance_transform_edt(img)
            edt[:,:, :, i] = (tmp - np.amin(tmp)) / (np.amax(tmp) - np.amin(tmp))

    return np.sum(contour, axis=-1) , np.sum(edt, axis=-1)
data_root = 'F:\datesets\challange\label/label0001.nii.gz'
def read_volume(path):
    itk_volume = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(itk_volume)
    return volume

def save_volume(volume, path):

    dst_volume = np.transpose(volume,(2,1,0))

    # dst_volume = np.flip(dst_volume, axis=1)
    # dst_volume = np.flip(dst_volume, axis=0)



    dst_volume_nii = nib.Nifti1Image(dst_volume, np.eye(4))
    nib.save(dst_volume_nii, path)
write_counter = 'G:\data_fubu/fubu_processed/abdominal_ct_160\counter'
write_dist = 'G:\data_fubu/fubu_processed/abdominal_ct_160\dist'
path = 'G:/data_fubu/fubu_processed/abdominal_ct_160/label/'
path_list = glob.glob(os.path.join(path,'*'))
for i in path_list:
    name = i.split('\\',7)[-1].split('.',3)[0].split('-',2)[-1]
    print(name)
    volume = read_volume(i)
    mask = np.zeros_like(volume)
    mask[volume>=1] =1
    counter,dist = seg_contour_dist_trans_volume(volume)

    save_volume(counter,write_counter+'/counter-'+str(name)+'.nii.gz')
    save_volume(dist,write_dist+'/dist-'+str(name)+'.nii.gz')


