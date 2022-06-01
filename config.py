#encoding: utf-8
from __future__ import print_function
import os
import platform
from utils import mkdir
from xml.dom import minidom
import json
class BrainAtlasConfig(object):

    def __init__(self):
        if 'Win' in platform.system():
            self.data_root = 'K:/adult_brain_atlases/dataset'
        self.n_split_folds = 6
        self.select = 0
        self.seed = 42
        self.n_labels = 95

    def __str__(self):
        print("net work config")
        print("*"*80)
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        return '\n'.join(str_list)

class Brats2018Config(object):
    def __init__(self):
        if "Win" in platform.system():
            self.data_dir = "G:/data_repos/Brats2018"
            self.n_workers = 4
        else:
            self.data_dir = "/home/share/data_repos/Brats2018"
            self.n_workers = 10
        self.input_shape = (160, 192, 128)
        self.modalities = ("t1", "t2", "flair", "t1ce")
        self.log_dir = './logs'
        # self.n_split_folds = 6
        self.select = 0
        self.seed = 42
        # self.n_labels = 95
        self.lr = 1e-3
        self.batch_size = 1
        self.n_epochs = 300
        self.pretrain_model = None
        self.save_result = True
        self.clip_grident = False
        self.tta = True

    def __str__(self):
        print("net work config")
        print("*"*80)
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]

        return '\n'.join(str_list)

class LPBAunaffineConfig(object):
    """LPBA40数据集"""
    def __init__(self):

        self.data_dir = 'F:/IR/LPBA40_affine/processed_nii'
        self.n_workers = 1

        self.label_xml_file = os.path.join('./lpba40.label.xml')
        self.n_split_folds = 4
        self.select = 0
        self.seed = 42
        self.n_labels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/LPBA'
        self.epoch = 300
        mkdir(self.log_dir)
        self.parse_label()

    def parse_label(self):
        label_xml = minidom.parse(self.label_xml_file)
        label_list = label_xml.getElementsByTagName('label')
        print('number of labels is ', len(label_list))
        self.label = {}
        for label in label_list:
            self.label[label.attributes['id'].value] = label.attributes['fullname'].value

    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*"*80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)

class LPBAConfig(object):
    """LPBA40数据集"""

    def __init__(self):
        if 'Win' in platform.system():
            print("using data from windows")
            #self.data_dir = 'F:/datesets/Dataset/LPBA40/delineation_space/LPBA40/'
            #self.data_dir = 'F:/datesets/Dataset/LPBA40/delineation_space/LPBA40/'

            self.n_workers = 1
        else:
            print("using data from linux")
            self.data_dir = '/home/zzy/origin_data/brain_registration/'

            #self.data_dir = '/home/zzy/origin_data/lpba_96/'
            self.n_workers = 15
        self.label_xml_file = os.path.join('./lpba40.label.xml')
        self.n_split_folds = 4
        self.select = 0
        self.seed = 42
        self.n_labels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/LPBA'
        self.epoch = 300
        mkdir(self.log_dir)
        self.parse_label()

    def parse_label(self):
        label_xml = minidom.parse(self.label_xml_file)
        label_list = label_xml.getElementsByTagName('label')
        print('number of labels is ', len(label_list))
        self.label = {}
        for label in label_list:
            self.label[label.attributes['id'].value] = label.attributes['fullname'].value

    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*" * 80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)
cortex_numbers_names = [
    [1002, "left caudal anterior cingulate"],
    [1003, "left caudal middle frontal"],
    [1005, "left cuneus"],
    [1006, "left entorhinal"],
    [1007, "left fusiform"],
    [1008, "left inferior parietal"],
    [1009, "left inferior temporal"],
    [1010, "left isthmus cingulate"],
    [1011, "left lateral occipital"],
    [1012, "left lateral orbitofrontal"],
    [1013, "left lingual"],
    [1014, "left medial orbitofrontal"],
    [1015, "left middle temporal"],
    [1016, "left parahippocampal"],
    [1017, "left paracentral"],
    [1018, "left pars opercularis"],
    [1019, "left pars orbitalis"],
    [1020, "left pars triangularis"],
    [1021, "left pericalcarine"],
    [1022, "left postcentral"],
    [1023, "left posterior cingulate"],
    [1024, "left precentral"],
    [1025, "left precuneus"],
    [1026, "left rostral anterior cingulate"],
    [1027, "left rostral middle frontal"],
    [1028, "left superior frontal"],
    [1029, "left superior parietal"],
    [1030, "left superior temporal"],
    [1031, "left supramarginal"],
    [1034, "left transverse temporal"],
    [1035, "left insula"],
    [2002, "right caudal anterior cingulate"],
    [2003, "right caudal middle frontal"],
    [2005, "right cuneus"],
    [2006, "right entorhinal"],
    [2007, "right fusiform"],
    [2008, "right inferior parietal"],
    [2009, "right inferior temporal"],
    [2010, "right isthmus cingulate"],
    [2011, "right lateral occipital"],
    [2012, "right lateral orbitofrontal"],
    [2013, "right lingual"],
    [2014, "right medial orbitofrontal"],
    [2015, "right middle temporal"],
    [2016, "right parahippocampal"],
    [2017, "right paracentral"],
    [2018, "right pars opercularis"],
    [2019, "right pars orbitalis"],
    [2020, "right pars triangularis"],
    [2021, "right pericalcarine"],
    [2022, "right postcentral"],
    [2023, "right posterior cingulate"],
    [2024, "right precentral"],
    [2025, "right precuneus"],
    [2026, "right rostral anterior cingulate"],
    [2027, "right rostral middle frontal"],
    [2028, "right superior frontal"],
    [2029, "right superior parietal"],
    [2030, "right superior temporal"],
    [2031, "right supramarginal"],
    [2034, "right transverse temporal"],
    [2035, "right insula"]]

class MindboggleConfig(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        #self.data_dir = '/home/zzy/origin_data/mindboggle/'
        self.data_dir = '/home/zzy/origin_data/mindboggle_seg_and_reg/'
        #self.data_dir = '/home/zzy/origin_data/mindboggle_data_in152aligned_freesurfer_192^3/'
        self.n_workers = 1
        self.n_split_folds =4
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/Mind'
        self.epoch =300
        mkdir(self.log_dir)
        self.parse_label()
        self.vis = True
        self.env = 'registration'
        self.plot_every = 100

    def parse_label(self):
        self.label = {}
        # for i in cortex_numbers_names:
        #     pix = i[0]
        #     self.label[str(pix)] = i[1]
        with open("./Bindboggle.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.label = load_dict


    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*"*80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)
class XIncangConfig(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        #self.data_dir = '/home/zzy/origin_data/mindboggle/'
        self.data_dir = '/home/zzy/origin_data/data_xinzao/'
        #self.data_dir = '/home/zzy/origin_data/mindboggle_data_in152aligned_freesurfer_192^3/'
        self.n_workers = 1
        self.n_split_folds =4
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/Mind'
        self.epoch =300
        mkdir(self.log_dir)
        self.parse_label()
        self.vis = True
        self.env = 'registration'
        self.plot_every = 100

    def parse_label(self):
        self.label = {"1":1,"2":2,"3":3}


    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*"*80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)
class FubuConfig(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        #self.data_dir = '/home/zzy/origin_data/mindboggle/'
        self.data_dir = '/home/zzy/origin_data/abdominal_ct_192/'
        #self.data_dir = '/home/zzy/origin_data/mindboggle_data_in152aligned_freesurfer_192^3/'
        self.n_workers = 1
        self.n_split_folds =6
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/Fubu'
        self.epoch =500
        mkdir(self.log_dir)
        self.parse_label()
        self.vis = True
        self.env = 'registration'
        self.plot_every = 100

    def parse_label(self):
        self.label = {"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"11":11,"12":12,"13":13}


    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*"*80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)


class FubuConfig_muti(object):
    """Mind101 dataset"""

    def __init__(self):
        # self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        # self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        # self.data_dir = '/home/zzy/origin_data/mindboggle/'
        self.data_dir = '/home/zzy/origin_data/abdominal_ct_160/'
        # self.data_dir = '/home/zzy/origin_data/mindboggle_data_in152aligned_freesurfer_192^3/'
        self.n_workers = 1
        self.n_split_folds = 6
        self.select = 0
        self.seed = 42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/Fubu'
        self.epoch = 300
        mkdir(self.log_dir)
        self.parse_label()
        self.vis = True
        self.env = 'registration'
        self.plot_every = 100

    def parse_label(self):
        self.label = {"1": 1}

    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*" * 80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)


class Brain2018Config(object):
    def __init__(self):
        if "Win" in platform.system():
            self.data_dir = "G:/data_repos/Brats2018"
            self.n_workers = 4
        else:
            self.data_dir = "/home/zzy/origin_data/mindboggle_seg_and_reg/"
            self.n_workers = 10
        self.input_shape = (192, 192, 192)
        self.modalities = ("t1", "t2", "flair", "t1ce")
        self.log_dir = './logs'
        # self.n_split_folds = 6
        self.select = 0
        self.seed = 42
        # self.n_labels = 95
        self.lr = 1e-4
        self.batch_size = 1
        self.n_epochs = 300
        self.pretrain_model = None
        self.save_result = True
        self.clip_grident = False
        self.tta = True
        self.parse_label()

    def __str__(self):
        print("net work config")
        print("*"*80)
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]

        return '\n'.join(str_list)
    def parse_label(self):
        self.label = {}

        with open("./Bindboggle.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.label = load_dict

class MindboggleConfig_sligned(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/mindboggle_data_in152aligned_freesurfer_192^3/'
        self.data_dir = '/home/zzy/origin_data/mindboggle_data_in152aligned/'
        self.n_workers = 1
        self.n_split_folds =4
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/Mind'
        self.epoch =300
        mkdir(self.log_dir)
        self.parse_label()

    def parse_label(self):
        self.label = {}
        # for i in cortex_numbers_names:
        #     pix = i[0]
        #     self.label[str(pix)] = i[1]
        with open("./Bindboggle.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.label = load_dict


    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*"*80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)



class IXIConfig(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        self.data_dir = '/home/zzy/origin_data/IXI_processed_skullstrip/volume_skullstrip_192'
        self.n_workers = 4
        self.n_split_folds =3
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/IXI'
        self.epoch =300
        mkdir(self.log_dir)
        self.parse_label()

    def parse_label(self):
        self.label = {}
        # for i in cortex_numbers_names:
        #     pix = i[0]
        #     self.label[str(pix)] = i[1]
        with open("./ixi.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.label = load_dict


    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*"*80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)
class faimConfig(object):
    """Mind101 dataset"""

    def __init__(self):
        self.data_dir = 'F:/IR/code_cnn/Medical-image-registration/data_processed'
        self.n_workers = 2
        self.n_split_folds = 3
        self.select = 0
        self.seed = 42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/Mind'
        self.epoch = 300
        mkdir(self.log_dir)
        # self.parse_label()

    def parse_label(self):
        self.label = {}
        # for i in cortex_numbers_names:
        #     pix = i[0]
        #     self.label[str(pix)] = i[1]
        with open("./mindboogle_seg.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.label = load_dict

    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*" * 80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)


class CumcConfig(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        self.data_dir = '/home/zzy/origin_data/cumc12/'
        self.n_workers = 4
        self.n_split_folds =3
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1

        self.epoch =300

        self.parse_label()

    def parse_label(self):
        self.label = {}

        with open("../CUMC12_label.txt", 'r') as f:
            for line in f:
                id,name = line.split('%')[0].rstrip(),line.split('%')[-1].strip()
                self.label[id] = name



    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*"*80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)
class MghcConfig(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        self.data_dir = '/home/zzy/origin_data/mgh10/'
        self.n_workers = 4
        self.n_split_folds =3
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1

        self.epoch =300

        self.parse_label()

    def parse_label(self):
        self.label = {}

        with open("./g_labels.txt", 'r') as f:
            for line in f:
                id,name = line.split('%')[0].rstrip(),line.split('%')[-1].strip()
                self.label[id] = name



    def __str__(self):
        print("net work config")
        str_list = ['%-20s---%s' % item for item in self.__dict__.items()]
        str_list.insert(0, "*"*80)
        str_list.append("*" * 80)

        return '\n'.join(str_list)
class IXIConfig(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        self.data_dir = '/home/zzy/origin_data/IXI/volume_skullstrip/'
        self.n_workers = 4
        self.n_split_folds =3
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/IXI'
        self.epoch =300
        mkdir(self.log_dir)
        self.parse_label()

    def parse_label(self):
        self.label = {}
        # for i in cortex_numbers_names:
        #     pix = i[0]
        #     self.label[str(pix)] = i[1]
        with open("./ixi.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.label = load_dict
class CTConfig(object):
    """Mind101 dataset"""
    def __init__(self):
        #self.label_xml_file = os.path.join(self.data_dir, './bindboggle.xml')
        #self.data_dir = '/home/zzy/origin_data/Mindboggle/data_nii'
        #self.data_dir = '/home/zzy/origin_data/mindboggle/'
        self.data_dir = '/home/zzy/origin_data/task3/Training'
        #self.data_dir = '/home/zzy/origin_data/mindboggle_data_in152aligned_freesurfer_192^3/'
        self.n_workers = 1
        self.n_split_folds =5
        self.select =0
        self.seed =42
        self.n_babels = 40
        self.lr = 1e-4
        self.batch_size = 1
        self.log_dir = './logs/Mind'
        self.epoch =300
        mkdir(self.log_dir)
        self.parse_label()
        self.vis = True
        self.env = 'registration'
        self.plot_every = 100

    def parse_label(self):
        self.label = {}
        # for i in cortex_numbers_names:
        #     pix = i[0]
        #     self.label[str(pix)] = i[1]
        with open("./ct.json", 'r') as load_f:
            load_dict = json.load(load_f)
            self.label = load_dict


if __name__ == "__main__":
    cfg = LPBAConfig()
    print(cfg)