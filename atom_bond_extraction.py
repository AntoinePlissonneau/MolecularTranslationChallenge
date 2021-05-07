# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:33:56 2021

@author: aplissonneau
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
# from detectron2.structures import BoxMode
import json
import multiprocessing
from collections import Counter, defaultdict
from xml.dom import minidom
import os
import numpy as np
import pandas as pd
from pqdm.processes import pqdm
from rdkit.Chem import Draw
from scipy.spatial.ckdtree import cKDTree

from utils import *
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


LABELS = {'C': 0,
         'P': 1,
         'TRIPLE': 2,
         'I': 3,
         'O': 4,
         'Br': 5,
         'Cl': 6,
         'H': 7,
         'AROMATIC': 8,
         'SINGLE': 9,
         'DOUBLE': 10,
         'N': 11,
         'Si': 12,
         'S': 13,
         'F': 14,
         'B': 15}

INVERSE_LABELS = {str(v): k for k, v in LABELS.items()}

COLOR_LIST = [[173, 195, 218],
                [86, 175, 87],
                [179, 154, 110],
                [169, 256, 4],
                [105, 217, 79],
                [138, 69, 134],
                [112, 179, 63],
                [213, 256, 256],
                [101, 172, 124],
                [150, 46, 144],
                [253, 253, 130],
                [166, 164, 146],
                [56, 129, 4],
                [76, 145, 134],
                [94, 156, 139],
                [240, 181, 53]]

    
def get_atoms(mol,d, atom_margin = 8):
    """ Compute the bounding box of each atom

    :param mol: rdkit mol object
    :param d: rdkit draw mol object
    :param atom_margin: bbox margin
    :return: Dict with bbox and category of each atom
    """
    atoms_labels = []
    for iatom in range(mol.GetNumAtoms()):
        p = d.GetDrawCoords(iatom)
        atom = mol.GetAtoms()[iatom]
        atom_type = atom.GetSymbol()
        _margin = atom_margin

        # better to predict close carbons (2 close instances affected by NMS)
        if atom_type == 'C':
            _margin /= 2

        # Because of the hydrogens normally the + sign falls out of the box
        if atom_type == 'N':
            _margin *= 2

        annotation = {'bbox':        [p.x/256,
                                      p.y/256,
                                      _margin*2/256,
                                      _margin*2/256],
                      'bbox_mode':   "XYWH_ABS",
                      'category_id': labels[atom_type]}
        # atoms_positions.append([atom_type, p.x, p.y])
        atoms_labels.append(annotation)
    return atoms_labels
    
def get_bonds(mol,d, bond_margin=5):
    """ Compute the bounding box of each bond

    :param mol: rdkit mol object
    :param d: rdkit draw mol object
    :param bond_margin: bbox margin
    :return: Dict with bbox and category of each bond
    """
    bonds_labels = []
    
    for ibond in range(mol.GetNumBonds()):
        bond = mol.GetBonds()[ibond]
        bond_type = bond.GetBondType().name
        start_pos = d.GetDrawCoords(bond.GetBeginAtomIdx())
        end_pos = d.GetDrawCoords(bond.GetEndAtomIdx())
        start_pos_x = start_pos.x
        start_pos_y = start_pos.y
        end_pos_x = end_pos.x
        end_pos_y = end_pos.y
        
        # make bigger margin for bigger bonds (double and triple)
        _margin = bond_margin
        if (bond_type == "DOUBLE") or (bond_type == "TRIPLE"):
            _margin *= 1.5
        
        x = (start_pos_x + end_pos_x) // 2  # left-most pos
        y = (start_pos_y + end_pos_y) // 2  # up-most pos
        width = abs(start_pos_x - end_pos_x) + _margin
        height = abs(start_pos_y - end_pos_y) + _margin
        
        annotation = {'bbox':        [x/256, y/256, width/256, height/256],
              'bbox_mode':   "XYWH_ABS",
              'category_id': labels[bond_type]}
        bonds_labels.append(annotation)
    return bonds_labels

def process_InChI(InChI):
    """ Compute the molecule image and its annotations

    :param InChI: InChI of the molecule
    :return: img, annotations
    """

    ## read International Chemical Identifier
    mol = Chem.inchi.MolFromInchi(InChI)

    ## draw molecule with 0 degree rotation
    d = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(256, 256)
    AllChem.Compute2DCoords(mol)
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    d.DrawMolecule(mol)
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().bondLineWidth = 1

    d.FinishDrawing()
    #d.drawOptions().rotate = 0
    d.WriteDrawingText("0.png")
    img0 = cv2.imread("0.png", cv2.IMREAD_GRAYSCALE)
    img0[img0 < 255] =0
    atoms = get_atoms(mol,d)
    bonds = get_bonds(mol, d)
    annotations= atoms+bonds
    return img0, annotations

def plot_bbox(img, annotations, show = True, save=False):
    """
    Plot bounding boxes and labels legend

    :param img: img
    :param annotations: Predicted bounding boxes. [dict]
    :return:
    """

    
    repr_cat = list(set([lab["category_id"] for lab in annotations]))
    patches = []
    for i, cat in enumerate(repr_cat):

        patches.append(mpatches.Patch(color=[col/256 for col in COLOR_LIST[int(cat)]],
                                      label=INVERSE_LABELS[str(cat)]))

    # draw rects
    for ins in annotations:
        ins_type = ins['category_id']
        x, y, width, height = ins['bbox']
        color = COLOR_LIST[int(ins_type)]

        cv2.rectangle(img, (int((x-width/2)*256), int((y-height/2)*256)),
                      (int((x + width/2)*256), int((y + height/2)*256)),color, 1)
    if show:
        plt.figure()
        plt.imshow(img)
        plt.legend(handles=patches)
        plt.show()
    if save:
        assert type(save)==str, "Save should be a path"
        plt.imshow(img)
        plt.legend(handles=patches)
        plt.savefig(save)
    return img

def label_txt_to_dict(annotations):
    
    labels = [{'category_id':int(ann[0]),
               'bbox':[*ann[1:]]} for ann in annotations]
    return labels

def process_and_save(InChI, img_id, output_path="test_atoms_bonds_detection"):
    img, annotations = process_InChI(InChI)
    cv2.imwrite(os.path.join(output_path, "img", img_id)+".png",img)
    with open(os.path.join(output_path, "labels", img_id)+".txt", "w") as labels_file:
        for annot in annotations:
            txt = f"{annot['category_id']} {annot['bbox'][0]} {annot['bbox'][1]} {annot['bbox'][2]} {annot['bbox'][3]} \n"
            labels_file.write(txt)

if __name__=='__main__':
    import os
    import time
    # ====================================================
    # Data Loading
    # ====================================================
    df = pd.read_csv(r"../bms-molecular-translation//train_labels.csv")
    test = True
    process_all = False
    output_path = "test_atoms_bonds_detection"
    # ====================================================
    # Drawing
    # ====================================================
    if test:
        # image_id = "000beabf176b"

        # InChI = df.loc[df.image_id == image_id,"InChI"].tolist()[0]
        dirpath = "yolov5/runs/detect/exp6/labels/"
        img_dir_path = "test_atoms_bonds_detection/images/"
        
        for img_id in os.listdir(dirpath)[0:100]:
            image_id = img_id[:-4]
    
            img_path = img_dir_path + image_id+".png"
            path = dirpath+image_id+".txt"
            
            annotations = np.loadtxt(path)
            annotations = label_txt_to_dict(annotations)
            img = cv2.imread(img_path)
            plot_bbox(img, annotations)
            # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        
    if process_all:
        dt_read = []
        dt_process = []
        dt_write = []
        dt_tot = []
        t = time.time()

        for i in range(df.shape[0]):
            row = df.iloc[i,:]
            img_id = row["image_id"]
            process_and_save(row["InChI"], img_id)
            if i % 1000 == 0:
                print("Iteration", i, "Time elapsed",time.time() - t)
                

                
        
                
                
                
                
                