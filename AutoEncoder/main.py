# ====================================================
# Library
# ====================================================
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from Trainer_AE import *
from preprocess_AE import *
import models_AE
from utils_AE import *

import warnings
warnings.filterwarnings('ignore')

# ====================================================
# Main
# ====================================================
if __name__ == '__main__':

    # Train Data
    train = pd.read_pickle("./train_labels_p.pkl")
    debug=False

    # Random seed
    seed_torch(seed=42)

    # Mode Debug
    if debug:
        train = train.sample(n=1000, random_state=42).reset_index(drop=True)
        train , val = train_test_split(train, random_state=42)

    else:
        train, val = train_test_split(train, test_size=0.10, random_state=42)
        train.reset_index(drop=True)
        val.reset_index(drop=True)

    # Dataset
    train_dataset = TrainDataset(train,transform=get_transforms(data='train'))
    val_dataset = TrainDataset(val,transform=get_transforms(data='train'))

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=True,
                            pin_memory=True)

    config = ConfigParser()
    config.read('./params.ini')

    # general parameters
    params = {'data_name': config['Data_parameters']['data_name'],
            'n_channels': int(config['Model_parameters']['n_channels']),
            'output_dim': int(config['Model_parameters']['output_dim']),
            'start_epoch': int(config['Training_parameters']['start_epoch']),
            'epochs': int(config['Training_parameters']['epochs']),
            'epochs_since_improvement': int(config['Training_parameters']['epochs_since_improvement']),
            'model_lr': float(config['Training_parameters']['model_lr']),
            'grad_clip': eval(config['Training_parameters']['grad_clip']),
            'best_mse': float(config['Training_parameters']['best_mse']),
            'print_freq': int(config['Training_parameters']['print_freq']),
            'checkpoint': (config['Training_parameters']['checkpoint']=='True'),
            'checkpoint_path':config['Training_parameters']['checkpoint_path']
            }
    
    mol = Trainer(params)
    mol.train_val_model(train_loader, val_loader)
    
    df_test = pd.read_csv("../input/bms-molecular-translation/sample_submission.csv").copy()

    def get_test_file_path(image_id):
      return "../input/bms-molecular-translation/test/{}/{}/{}/{}.png".format(
          image_id[0], image_id[1], image_id[2], image_id)

    df_test['file_path'] = df_test['image_id'].progress_apply(get_test_file_path)
    
    test_img_dataset = TestDataset(df_test,transform=get_transforms(data='valid'))
    
    index = 2003

    inp_ = (test_img_dataset[index].squeeze().numpy()*255).astype(np.uint8)
    pred_ = mol.model(test_img_dataset[index].unsqueeze(0).to(device))
    pred_ = (pred_.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50,100))

    # Display the image
    ax1.imshow(inp_)
    ax1.set_title('Val_Input', fontsize=50)
    ax2.imshow(pred_)
    ax2.set_title('Prediction', fontsize=50)