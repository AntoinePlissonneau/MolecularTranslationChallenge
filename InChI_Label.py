# ====================================================
# Library
# ====================================================
import os
import re
import numpy as np
import pandas as pd
import sys
from tqdm.auto import tqdm
tqdm.pandas()

# ====================================================
# Data Loading
# ====================================================
df = pd.read_csv(r"F://bms-molecular-translation//train_labels.csv").copy()

atom = {'B': 0, 'Br': 1, 'C': 2, 'Cl': 3, 'D': 4, 'F': 5, 'H': 6, 'I': 7, 'N': 8, 'O': 9, 'P': 10, 'S': 11, 'Si': 12, 'T':13}

# ====================================================
# Preprocess functions
# ====================================================
def split_InChI(s):
    res  = re.findall('[A-Z][a-z]?|[0-9]+', s)
    l = ['0']*(len(atom))
    for i,j in enumerate(res):
        try:
            index = atom[j]
            try:
                nb = int(res[i+1])
                l[index]= str(nb)
            except:
                l[index]='1'
        except:
            pass
    return "/".join(l)

def get_train_file_path(image_id):
    return "F:/bms-molecular-translation/train/{}/{}/{}/{}.png".format(
        image_id[0], image_id[1], image_id[2], image_id)

# ====================================================
# main
# ====================================================
def main():
    # ====================================================
    # preprocess train.csv
    # ====================================================
    df['Ft'] = df['InChI'].progress_apply(lambda x: x.split('/')[1])
    df['NbAt'] = df['Ft'].progress_apply(split_InChI)
    df['file_path'] = df['image_id'].progress_apply(get_train_file_path)
    df.to_pickle("F://bms-molecular-translation//train_labels_p.pkl")
    sys.stderr.write('Data saved')

if __name__ == '__main__':
    main()

