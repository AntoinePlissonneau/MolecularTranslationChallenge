# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:09:12 2021

@author: aplissonneau
"""

import json
#Separate bonds and atoms 
#Add all atoms to rdkit mol and get their id
#Calculate nearests ids to bonds
#Add bonds to rdkit mol
#To InChI
from rdkit import Chem
import numpy as np
from scipy.spatial import distance
from atom_bond_extraction import label_txt_to_dict, plot_bbox
import cv2
import matplotlib.pyplot as plt
import pandas as pd

ATOMS_LABELS_KEYS = ['C', 'P', 'I', 'O', 'Br', 'Cl',
                'H', 'N', 'Si', 'S', 'F', 'B']

def distance_yolo(atom, bond):
    x, y , width, height = bond[1:]
    l_b_corner = (x-width/2, y-height/2)
    l_t_corner = (x-width/2, y+height/2)
    r_b_corner = (x+width/2, y-height/2)
    r_t_corner = (x+width/2, y+height/2)
    corners = [l_b_corner, l_t_corner, r_b_corner, r_t_corner]
    atom_coord = (atom[1], atom[2])
    dst = [distance.euclidean(corner, atom_coord) for corner in corners]
    return dst

def nearest_atoms(atoms, bond):
    distances = np.array([distance_yolo(atom, bond) for atom in atoms])
    min_idx = distances.argmin(axis = 0)
    min_value = distances.min(axis = 0)
    x, y , width, height = bond[1:]
    if min_value[[0,3]].sum() > min_value[[1,2]].sum():
        nearest = min_idx[[1,2]]
    else:
        nearest = min_idx[[0,3]]     
    return nearest

def build_mol(annotations):
    atoms_labels_values = [labels[k] for k in ATOMS_LABELS_KEYS]
    atoms = annotations[np.isin(annotations[:,0],atoms_labels_values)]
    bonds = annotations[~np.isin(annotations[:,0],atoms_labels_values)]
                
    mol = Chem.RWMol()
    for a in atoms:
        a1 = Chem.Atom(inverse_labels[str(int(a[0]))])
        a_i = mol.AddAtom(a1)
    
    for b in bonds:
        print("ok")
        nearest = nearest_atoms(atoms,b)
    
        lab = inverse_labels[str(int(b[0]))]
        if lab == "DOUBLE":
            lab = Chem.rdchem.BondType.DOUBLE
        elif lab == "SINGLE":
            lab = Chem.rdchem.BondType.SINGLE
        elif lab == "TRIPLE":
            lab = Chem.rdchem.BondType.TRIPLE
        elif lab == "AROMATIC":
            lab = Chem.rdchem.BondType.AROMATIC
        mol.AddBond(int(nearest[0]), int(nearest[1]), lab)

    # Chem.SanitizeMol(mol)
    return mol    

def plot_nearest_atoms(annotations, img):
    atoms_labels_values = [labels[k] for k in ATOMS_LABELS_KEYS]
    atoms = annotations[np.isin(annotations[:,0],atoms_labels_values)]
    bonds = annotations[~np.isin(annotations[:,0],atoms_labels_values)]

    for i,b in enumerate(bonds):
        if i%9==0:
            plt.figure(figsize = [12,12])
        img2 = img.copy()
        nearest = nearest_atoms(atoms,b)
        x, y, width, height = b[1:]
        cv2.rectangle(img2, (int((x-width/2)*256), int((y-height/2)*256)),
                      (int((x + width/2)*256), int((y + height/2)*256)),(255,0,0), 1)
        for a in nearest:
            x, y, width, height = atoms[a][1:]
            cv2.rectangle(img2, (int((x-width/2)*256), int((y-height/2)*256)),
                          (int((x + width/2)*256), int((y + height/2)*256)),(255,0,0), 1)
        plt.subplot(3, 3, (i%9)+1)
        plt.imshow(img2)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    with open("labels_id.json", "r") as f:
        labels = json.load(f) 
    
    df = pd.read_csv(r"../bms-molecular-translation//train_labels.csv")
    # img_name = "000beabf176b"
    img_name = "008b91d06d77"
    InChI = df.loc[df.image_id == img_name,"InChI"].tolist()[0]
    
    inverse_labels = {str(v): k for k, v in labels.items()}
    path_predicted_label = f"yolov5/runs/detect/exp6/labels/{img_name}.txt"
    path_true_label =  f"test_atoms_bonds_detection/labels/{img_name}.txt"
    
    img_dir_path = "test_atoms_bonds_detection/images/"
    img_path = img_dir_path + f"{img_name}.png" 
    img = cv2.imread(img_path)
    
    annotations = np.loadtxt(path_predicted_label)
    annotations_dict = label_txt_to_dict(annotations)
    plot_bbox(img, annotations_dict)
    
    img = cv2.imread(img_path)

    plot_nearest_atoms(annotations, img)
    mol = build_mol(annotations)
    Chem.MolToInchi(mol)
    print(InChI)





