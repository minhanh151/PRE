import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


    # Code from https://github.com/SsnL/dataset-distillation/blob/master/datasets/pascal_voc.py , thanks to the authors
"""Dataset setting and data loader for PASCAL VOC 2007 as a classification task.

Modified from
https://github.com/Cadene/pretrained-models.pytorch/blob/56aa8c921819d14fb36d7248ab71e191b37cb146/pretrainedmodels/datasets/voc.py
"""

import os
import os.path
import tarfile
import xml.etree.ElementTree as ET

import torch.utils.data as data
import torchvision
from PIL import Image
from urllib.parse import urlparse
import torch

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

category_to_idx = {c: i for i, c in enumerate(object_categories)}
idx_to_category = {i: c for i, c in enumerate(object_categories)}

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


def download_url(url, path):
    root, filename = os.path.split(path)
    torchvision.datasets.utils.download_url(url, root=root, filename=filename, md5=None)


def download_voc2007(root):
    path_devkit = os.path.join(root, 'VOCdevkit')
    path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):

        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            download_url(urls['devkit'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(urls['trainval_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            download_url(urls['trainval_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test annotations
    test_anno = os.path.join(path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):

        # download test annotations
        parts = urlparse(urls['test_images_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            download_url(urls['test_images_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    test_image = os.path.join(path_devkit, 'VOC2007/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(urls['test_anno_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            download_url(urls['test_anno_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')


def read_split(root, dataset, split):
    base_path = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    filename = os.path.join(base_path, object_categories[0] + '_' + split + '.txt')

    with open(filename, 'r') as f:
        paths = []
        for line in f.readlines():
            line = line.strip().split()
            if len(line) > 0:
                assert len(line) == 2
                paths.append(line[0])

        return tuple(paths)


def read_bndbox(root, dataset, paths):
    xml_base = os.path.join(root, 'VOCdevkit', dataset, 'Annotations')
    instances = []
    for path in paths:
        xml = ET.parse(os.path.join(xml_base, path + '.xml'))
        for obj in xml.findall('object'):
            c = obj[0]
            assert c.tag == 'name', c.tag
            c = category_to_idx[c.text]
            bndbox = obj.find('bndbox')
            xmin = int(bndbox[0].text)  # left
            ymin = int(bndbox[1].text)  # top
            xmax = int(bndbox[2].text)  # right
            ymax = int(bndbox[3].text)  # bottom
            instances.append((path, (xmin, ymin, xmax, ymax), c))
    return instances



@DATASET_REGISTRY.register()
class VOC2007(DatasetBase):

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.root = root
        self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.dataset_dir = os.path.join(root, 'VOCdevkit', 'VOC2007')
        self.path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        
        # download dataset
        download_voc2007(self.root)
        
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "JPEGImages")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train = self.get_data("train")
        val = self.get_data("val")
        test = self.get_data("test")

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    def __getitem__(self, index):
        path, crop, target = self.bndboxes[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        img = img.crop(crop)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.bndboxes)
    
    def get_data(self, split):
        paths = read_split(self.root, 'VOC2007', split)
        self.bndboxes = read_bndbox(self.root, 'VOC2007', paths)
        data_dir = os.path.join(self.dataset_dir, "CroppedImages", split)

        items = []
        for instance in self.bndboxes:
            path, crop, target = instance
            classname = idx_to_category[target]
            cropimpath = os.path.join(data_dir, path + '.jpg')

            if os.path.exists(data_dir):
                item = Datum(impath=cropimpath, label=target, classname=classname)
                items.append(item)
            else:
                mkdir_if_missing(data_dir)
                impath = os.path.join(self.path_images, path + '.jpg')
                img = Image.open(impath).convert('RGB')
                img = img.crop(crop)
                img.save(cropimpath, 'JPEG')
                item = Datum(impath=cropimpath, label=target, classname=classname)
                items.append(item)

        return items

    

