# -*- coding: utf-8 -*-
"""
Created on Sat May  1 20:20:11 2021

@author: aplissonneau
"""

import pandas as pd
from rdkit import Chem
import json
import time
import functools
import operator

def functools_reduce_iconcat(a):
    return functools.reduce(operator.iconcat, a, [])

def get_labels(InChI):
    mol = Chem.inchi.MolFromInchi(InChI)
    atom_list = []
    bond_list = []
    for iatom in range(mol.GetNumAtoms()):
        atom = mol.GetAtoms()[iatom]
        atom_type = atom.GetSymbol()
        atom_list.append(atom_type)
    for ibond in range(mol.GetNumBonds()):
        bond = mol.GetBonds()[ibond]
        bond_type = bond.GetBondType().name
        bond_list.append(bond_type)
    return atom_list + bond_list

if __name__ == "__main__":
    df = pd.read_csv(r"../bms-molecular-translation//train_labels.csv")
    t = time.time()
    labels = []
    # n_jobs = multiprocessing.cpu_count() - 1

    # result = pqdm(df.InChI.to_list(), get_labels,
    #               n_jobs=n_jobs, desc='Calculating unique atom-smiles and rarity')
    # labels = functools_reduce_iconcat(result)
    for i in range(df.shape[0]):
        row = df.iloc[i,:]
        labels += get_labels(row.InChI)
        if i%10000==0:
            print("Iteration", i, "Time elapsed",time.time() - t)
    unique_labels = list(set(labels))
    unique_labels_id = {lab:i for i,lab in enumerate(unique_labels)}
    with open("labels_id.json","w") as f:
        json.dump(unique_labels_id,f)
            

    
    
    
    