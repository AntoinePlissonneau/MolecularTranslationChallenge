# -*- coding: utf-8 -*-
"""
Created on Mon May  3 19:53:24 2021

@author: aplissonneau
"""
import os
import pandas as pd
import random
from shutil import copyfile
from pathlib import Path

import glob

random.seed(2)
imgs_path = Path("test_atoms_bonds_detection/images")
imgs_train_path = Path("test_atoms_bonds_detection/images/train")
imgs_val_path = Path("test_atoms_bonds_detection/images/val")

labels_path = Path("test_atoms_bonds_detection/labels")
labels_train_path = Path("test_atoms_bonds_detection/labels/train")
labels_val_path = Path("test_atoms_bonds_detection/labels/val")


os.makedirs(imgs_train_path, exist_ok=True)
os.makedirs(imgs_val_path, exist_ok=True)

os.makedirs(labels_train_path, exist_ok=True)
os.makedirs(labels_val_path, exist_ok=True)

erase = False
if erase:
    for src in [imgs_train_path, imgs_val_path, labels_train_path, labels_val_path]:
        files = src.glob("**/*")
        for f in files:
            os.remove(f)


df = pd.read_csv(r"../bms-molecular-translation//train_labels.csv")
ids = df.image_id.tolist()
train = random.sample(ids,500000)
val = random.sample(ids,20000)

i = 0
for elem in train:
    i+=1
    if i% 1000==0:
        print(i)
    copyfile((imgs_path/elem).with_suffix(".png"),
             (imgs_train_path/elem).with_suffix(".png"))
    copyfile((labels_path/elem).with_suffix(".txt"),
             (labels_train_path/elem).with_suffix(".txt"))

for elem in val:
    copyfile((imgs_path/elem).with_suffix(".png"),
             (imgs_val_path/elem).with_suffix(".png"))
    copyfile((labels_path/elem).with_suffix(".txt"),
             (labels_val_path/elem).with_suffix(".txt"))
