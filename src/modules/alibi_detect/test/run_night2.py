import numpy as np    
import json 
import os
import sys
sys.path.append('/home/ubuntu/image-drift-monitoring/src')
import numpy as np
from modules.alibi_detect.untrained_encoder import UntrainedAutoencoder
import json 

import numpy as np
from modules.alibi_detect.trained_autoencoder import TrainedAutoencoder
from torch.utils.data import TensorDataset, DataLoader
import json 
import torch
def main():
    



    with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
            drift_detection_config = json.load(config_file)

    

    camelyon_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_5_ds.npz')
    camelyon_train_5 = camelyon_train_5_comp['arr_0']
    camelyon_train_5_0_50 = camelyon_train_5[:int(len(camelyon_train_5)*0.5)]
    camelyon_train_5_50_100 = camelyon_train_5[int(len(camelyon_train_5)*0.5):]

    camelyon_train_5_0_50 = torch.as_tensor(camelyon_train_5_0_50)
    camelyon_train_5_0_50_dl = DataLoader(TensorDataset(camelyon_train_5_0_50,camelyon_train_5_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

    myTAE = TrainedAutoencoder(drift_detection_config)
    myTAE.init_default_pt_autoencoder(camelyon_train_5_0_50_dl)
    myTAE.init_detector(detector_type='KS',reference_data=camelyon_train_5_50_100,detector_name='camelyon_TAE_5_KS',save_dec=True)

    camelyon_train_5_comp= None
    camelyon_train_5 = None
    camelyon_train_5_0_50 = None 
    camelyon_train_5_50_100 = None
    camelyon_train_5_0_50_dl = None
    myTAE = None 

    camelyon_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_10_ds.npz')
    camelyon_train_10 = camelyon_train_10_comp['arr_0']
    camelyon_train_10_0_50 = camelyon_train_10[:int(len(camelyon_train_10)*0.5)]
    camelyon_train_10_50_100 = camelyon_train_10[int(len(camelyon_train_10)*0.5):]

    camelyon_train_10_0_50 = torch.as_tensor(camelyon_train_10_0_50)
    camelyon_train_10_0_50_dl = DataLoader(TensorDataset(camelyon_train_10_0_50,camelyon_train_10_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

    myTAE = TrainedAutoencoder(drift_detection_config)
    myTAE.init_default_pt_autoencoder(camelyon_train_10_0_50_dl)
    myTAE.init_detector(detector_type='KS',reference_data=camelyon_train_10_50_100,detector_name='camelyon_TAE_10_KS',save_dec=True)

    camelyon_train_10_comp= None
    camelyon_train_10 = None
    camelyon_train_10_0_50 = None 
    camelyon_train_10_50_100 = None
    camelyon_train_10_0_50_dl = None
    myTAE = None 

    camelyon_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_15_ds.npz')
    camelyon_train_15 = camelyon_train_15_comp['arr_0']
    camelyon_train_15_0_50 = camelyon_train_15[:int(len(camelyon_train_15)*0.5)]
    camelyon_train_15_50_100 = camelyon_train_15[int(len(camelyon_train_15)*0.5):]

    camelyon_train_15_0_50 = torch.as_tensor(camelyon_train_15_0_50)
    camelyon_train_15_0_50_dl = DataLoader(TensorDataset(camelyon_train_15_0_50,camelyon_train_15_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

    myTAE = TrainedAutoencoder(drift_detection_config)
    myTAE.init_default_pt_autoencoder(camelyon_train_15_0_50_dl)
    myTAE.init_detector(detector_type='KS',reference_data=camelyon_train_15_50_100,detector_name='camelyon_TAE_15_KS',save_dec=True)

    camelyon_train_15_comp= None
    camelyon_train_15 = None
    camelyon_train_15_0_50 = None 
    camelyon_train_15_50_100 = None
    camelyon_train_15_0_50_dl = None
    myTAE = None



# ======================================================================================
# call
if __name__ == "__main__":
    main()