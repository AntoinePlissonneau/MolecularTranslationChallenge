# ====================================================
# Library
# ====================================================
from configparser import ConfigParser

config = ConfigParser()
config['Preprocessing_parameters'] = {}
config['Preprocessing_parameters']['size_image'] = '224'

with open('./preprocessing.ini', 'w') as configfile:
    config.write(configfile)

config = ConfigParser(allow_no_value=True)
config['Data_parameters'] = {}
config['Data_parameters']['data_name'] = 'AE_02052021'

config['Model_parameters'] = {}
config['Model_parameters']['n_channels'] = '1'
config['Model_parameters']['output_dim'] = '3'

config['Training_parameters'] = {}
config['Training_parameters']['start_epoch'] = '0'
config['Training_parameters']['epochs'] = '1'
config['Training_parameters']['epochs_since_improvement'] = '0'
config['Training_parameters']['model_lr'] = '5e-5'
config['Training_parameters']['grad_clip'] = 'False'
config['Training_parameters']['best_mse'] = '1000'
config['Training_parameters']['print_freq'] = '1'
config['Training_parameters']['checkpoint'] = 'True'
config['Training_parameters']['checkpoint_path'] = '../input/models02/checkpoint_AE_02052021.pth (1).tar'

with open('./params.ini', 'w') as configfile:
    config.write(configfile)
