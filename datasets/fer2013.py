import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

import os
from os import listdir
import warnings
import sys
import json
from subprocess import call
from collections import defaultdict
import torch
from torchvision.datasets import (
    VisionDataset, ImageFolder,
    CIFAR10, CIFAR100, ImageNet, CocoCaptions, Flickr8k, Flickr30k,
    MNIST, STL10, Kitti
)
from . import voc2007, caltech101, imagenetv2
from PIL import Image


def _load_classnames_and_classification_templates(dataset_name, current_folder, language):
    with open(os.path.join(current_folder, language + "_classnames.json"), "r") as f:
        classnames = json.load(f)

     # Zero-shot classification templates, collected from a bunch of sources
    # - CLIP paper (https://github.com/openai/CLIP/blob/main/data/prompts.md)
    # - Lit Paper (https://arxiv.org/pdf/2111.07991.pdf)
    # - SLIP paper (https://github.com/facebookresearch/SLIP/blob/main/templates.json)
    # Some are fixed mnaually

    with open(os.path.join(current_folder, language + "_zeroshot_classification_templates.json"), "r") as f:
        zeroshot_classification_templates = json.load(f)
    # default template to use when the dataset name does not belong to `zeroshot_classification_templates`
    DEFAULT_ZEROSHOT_CLASSIFICATION_TEMPLATES = zeroshot_classification_templates["imagenet1k"]

    if dataset_name.startswith("tfds/") or dataset_name.startswith("vtab/") or dataset_name.startswith("wds/"):
        name = dataset_name.split("/")[-1]
    else:
        name = dataset_name
    templates = zeroshot_classification_templates.get(name, DEFAULT_ZEROSHOT_CLASSIFICATION_TEMPLATES)

    return classnames, templates

def has_kaggle():
    return call("which kaggle", shell=True) == 0

@DATASET_REGISTRY.register()
class Fer2013(DatasetBase):
    dataset_dir = 'fer2013'

    def __init__(self, cfg):
        current_folder = os.path.dirname(__file__)
        allclassnames, templates = _load_classnames_and_classification_templates(self.dataset_dir, current_folder, language='en')
        classnames = allclassnames["fer2013"]
        print(classnames)

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if not os.path.exists(self.dataset_dir):
            # Automatic download
            print("Downloading fer2013...")
            if not has_kaggle():
                print("Kaggle is needed to download the dataset. Please install it via `pip install kaggle`")
                sys.exit(1)
            call(f"kaggle datasets download --unzip -p { self.dataset_dir} msambare/fer2013", shell=True)

        train = self.get_data("train", classnames)
        val = self.get_data("train", classnames)
        test = self.get_data("test", classnames)

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

        
    def get_data(self, split, classnames):
        items = []
        for i, c in enumerate(classnames):
            data_dir = os.path.join(self.dataset_dir, split, c)
            for f in listdir(data_dir):
                impath = os.path.join(data_dir, f)
                item = Datum(impath=impath, label=i, classname=c)
                items.append(item)

        return items


