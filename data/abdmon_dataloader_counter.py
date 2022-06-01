
import numpy as np
import module.common_module as cm
from glob import glob
from torch.utils.data import Dataset, DataLoader
import random
import pickle
from torchvision import transforms, utils
import module.transform as trans
import torch
import nibabel as nib
class LPBA(Dataset):
    def __init__(self, path_list,
                 target_shape=(192, 192, 192),
                 return_label=False,
                 augment=True):
        self.path_list = path_list
        n = len(self.path_list)
        self.n_pairs = n * (n - 1)
        self.target_shape = target_shape
        self.return_label = return_label
        self.augment = augment

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        fix_idx = idx // (len(self.path_list) - 1)
        fix_path = self.path_list[fix_idx]
        move_idx = idx % (len(self.path_list) - 1)
        if move_idx >= fix_idx:
            move_idx = move_idx + 1
        move_path = self.path_list[move_idx]
        move_volume, move_label= self.read_sample(move_path[0], move_path[1])
        fix_volume, fix_label = self.read_sample(fix_path[0], fix_path[1])



        move_mask = np.zeros_like(move_label)
        move_mask[move_label>=1] =1
        fix_mask =np.zeros_like(fix_label)
        fix_mask[fix_label>=1] = 1
        move_volume = np.expand_dims(move_volume, axis=0)
        move_label = np.expand_dims(move_label, axis=0)
        fix_volume = np.expand_dims(fix_volume, axis=0)
        fix_label = np.expand_dims(fix_label, axis=0)
        move_mask = np.expand_dims(move_mask, axis=0)
        fix_mask = np.expand_dims(fix_mask, axis=0)

        if not self.return_label:
            return (torch.tensor(fix_volume.copy(), dtype=torch.float),
                    torch.tensor(fix_label.copy(), dtype=torch.float),
                    torch.tensor(move_volume.copy(), dtype=torch.float),
                    torch.tensor(move_label.copy(), dtype=torch.float),
                    torch.tensor(fix_mask.copy(), dtype=torch.float),
                    torch.tensor(move_mask.copy(), dtype=torch.float),



                    )

        else:
            return (torch.tensor(fix_volume.copy(), dtype=torch.float),
                    torch.tensor(fix_label.copy(), dtype=torch.float),
                    torch.tensor(move_volume.copy(), dtype=torch.float),
                    torch.tensor(move_label.copy(), dtype=torch.float),
                    torch.tensor(fix_mask.copy(), dtype=torch.float),
                    torch.tensor(move_mask.copy(), dtype=torch.float)

                   )
    def read_sample(self, volume_path, label_path):
        volume = nib.load(volume_path).get_data()
        label = nib.load(label_path).get_data()

        volume = self.normlize(volume)

        return volume, label
    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
class BraTSDataset(Dataset):
    """Segmentation dataset

    Image: T1, Flair [Channel, Z, H, W]
    Labels: 0 - background, 1 - White matter lesion
    """

    def __init__(self, img_list, mask_list,counter_list, transform=None):
        self.img_list = img_list
        self.mask_list = mask_list
        self.counter_list = counter_list
        self.transform = transform

    def __len__(self):
        # assert len(self.img_list) == len(self.mask_list)
        return len(self.img_list)

    def __getitem__(self, idx):
        image = nib.load(self.img_list[idx]).get_data()
        image = self.normlize(image)
        image = np.expand_dims(image, axis=0)


        mask = nib.load(self.mask_list[idx]).get_data()
        dst_mask = mask.copy()
        dst_mask[dst_mask >= 1] = 1
        dst_mask = np.expand_dims(dst_mask, axis=0)

        counter = nib.load(self.counter_list[idx]).get_data()
        dst_counter = counter.copy()
        dst_counter[counter==1]=1
        dst_counter = np.expand_dims(dst_counter, axis=0)





        sample = {'image': image, 'mask': dst_mask,'dist':dst_counter}

        if self.transform:
            sample = self.transform(sample)

        return sample
    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())


def abdomen_data(data_seed, data_split):

    # Set random seed
    data_seed = data_seed
    np.random.seed(data_seed)


    # Create image list
    # data_list = sorted(glob('/home/zzy/origin_data/abdominal_ct_160/img/*.nii.gz'))
    # label_list = sorted(glob('/home/zzy/origin_data/abdominal_ct_160/label/*.nii.gz'))
    # counter_list = sorted(glob('/home/zzy/origin_data/abdominal_ct_160/lever_set/*.nii.gz'))

    data_list = sorted(glob('/raid/raoyi/raid/raoyi/dataset/abdominal_ct_160_AttentionUnet/img_attention/*.nii.gz'))
    label_list = sorted(glob('/raid/raoyi/raid/raoyi/dataset/abdominal_ct_160_AttentionUnet/label_attention/*.nii.gz'))
    counter_list = sorted(glob('/raid/raoyi/raid/raoyi/dataset/abdominal_ct_160_AttentionUnet/lever_set_attention/*.nii.gz'))
    print(len(data_list),len(label_list),len(counter_list))
    for path in zip(data_list,label_list,counter_list):
        print(path)


    # Random selection for training, validation and testing
    volume_list = [list(pair) for pair in zip(data_list, label_list,counter_list)]

    # np.random.shuffle(volume_list)

    train_labeled_img_list = []
    train_labeled_mask_list = []
    train_unlabeled_img_list = []
    train_unlabeled_mask_list = []
    val_labeled_img_list = []
    val_labeled_mask_list = []
    val_unlabeled_img_list = []
    val_unlabeled_mask_list = []

    # if data_split == '20L0U' or data_split == '10L100U':
    #     train_labeled_img_list, train_labeled_mask_list,train_label_counter_list =map(list, zip(*(volume_list[0:10])))
    #     train_unlabeled_img_list, train_unlabeled_mask_list,train_unlabel_counter_list = map(list, zip(*(volume_list[10:50])))
    #     val_labeled_img_list, val_labeled_mask_list,val_label_counter_list = map(list, zip(*(volume_list[50:60])))
    #     val_unlabeled_img_list, val_unlabeled_mask_list,val_unlabel_counter_list = map(list, zip(*(volume_list[50:60])))

    if data_split == '20L0U' or data_split == '10L100U':
        train_labeled_img_list, train_labeled_mask_list,train_label_counter_list =map(list, zip(*(volume_list[50:60])))
        train_unlabeled_img_list, train_unlabeled_mask_list,train_unlabel_counter_list = map(list, zip(*(volume_list[10:50])))
        val_labeled_img_list, val_labeled_mask_list,val_label_counter_list = map(list, zip(*(volume_list[0:10])))
        val_unlabeled_img_list, val_unlabeled_mask_list,val_unlabel_counter_list = map(list, zip(*(volume_list[0:10])))

    # Reset random seed
    seed = random.randint(1, 9999999)
    np.random.seed(seed + 1)
    volume_path_list = data_list
    label_path_list = label_list
    print("Total ", len(volume_path_list), " volumes")
    path_idx = np.arange(len(volume_path_list))
    np.random.seed(42)
    np.random.shuffle(path_idx)
    print('nfold', 6)
    nfold_list = np.split(path_idx, 6)
    val_path_list = []
    train_path_list = []
    for i, fold in enumerate(nfold_list):
        if i == 0:
            for idx in fold:
                val_path_list.append((volume_path_list[idx], label_path_list[idx]))
        else:
            for idx in fold:
                train_path_list.append((volume_path_list[idx], label_path_list[idx]))

    for path in val_path_list:
        print('^^^^', path[0], path[1])

    print("length of train list is ", len(train_path_list))
    print("length of validation list is ", len(val_path_list))

    # Build Dataset Class


    # Iterating through the dataset
    trainLabeledDataset = BraTSDataset(train_labeled_img_list, train_labeled_mask_list,train_label_counter_list,
                                     transform=transforms.Compose([
                                         #trans.RandomCrop(cm.BraTSshape),
                                         # trans.Elastic(),
                                         #trans.Flip(horizontal=True),
                                         trans.ToTensor()
                                     ])
                                     )

    trainUnlabeledDataset = BraTSDataset(train_unlabeled_img_list, train_unlabeled_mask_list,train_unlabel_counter_list,
                                       transform=transforms.Compose([
                                           #trans.RandomCrop(cm.BraTSshape),
                                           # trans.Elastic(),
                                           #trans.Flip(horizontal=True),
                                           trans.ToTensor()
                                       ])
                                       )

    valLabeledDataset = BraTSDataset(val_labeled_img_list, val_labeled_mask_list,val_label_counter_list,
                                   transform=transforms.Compose([
                                       # trans.Crop(cm.BraTSshape),
                                       # trans.Flip(),
                                       trans.ToTensor()
                                   ])
                                   )

    valUnlabeledDataset = BraTSDataset(val_unlabeled_img_list, val_unlabeled_mask_list,val_unlabel_counter_list,
                                     transform=transforms.Compose([
                                         #trans.Crop(cm.BraTSshape),
                                         #trans.Flip(),
                                         trans.ToTensor()
                                     ])
                                     )

    train_ds = LPBA(train_path_list, augment=False)
    val_ds = LPBA(val_path_list, augment=False, return_label=True)

    # device_type = 'cpu'
    device_type = 'cuda'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    print(device)
    dataset_sizes = {'trainLabeled': len(trainLabeledDataset), 'trainUnlabeled': len(trainUnlabeledDataset),
                     'val_labeled': len(valLabeledDataset), 'val_unlabeled': len(valUnlabeledDataset)}

    modelDataLoader = {'trainLabeled': DataLoader(trainLabeledDataset, batch_size=1, shuffle=True, num_workers=1),
                       'trainUnlabeled': DataLoader(trainUnlabeledDataset, batch_size=1, shuffle=True, num_workers=1),
                       'val_labeled': DataLoader(valLabeledDataset, batch_size=1, shuffle=True, num_workers=1),
                       'val_unlabeled': DataLoader(valUnlabeledDataset, batch_size=1, shuffle=True, num_workers=1),
                       'test_labeled': DataLoader(valUnlabeledDataset, batch_size=1, shuffle=True, num_workers=1),
                       'eval_reg': DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4),
                       'train_reg': DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=4)
                       }

    return device, dataset_sizes, modelDataLoader

if __name__ == '__main__':
    device, data_sizes, modelDataloader =abdomen_data(1, '10L100U')
    x = modelDataloader['trainLabeled']
    print(len(x))
    y = modelDataloader['trainUnlabeled']
    print(len(y))
    #print(enumerate(x).__next__())
    for n in range(50):
        print ('________',n,'_____')
        for i, (sample1, sample2) in enumerate(zip(x,y)):
            print(sample1['image'].shape)
            print(i,len(sample1),len(sample2))



