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

    
    for i in range(5,105,5):

        if i == 100 :
            camelyon_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')    
        else :
            camelyon_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_{}_ds.npz'.format(i))
        camelyon_train_5 = camelyon_train_5_comp['arr_0']
        camelyon_train_5_0_50 = camelyon_train_5[:int(len(camelyon_train_5)*0.5)]
        camelyon_train_5_50_100 = camelyon_train_5[int(len(camelyon_train_5)*0.5):]

        camelyon_train_5_0_50 = torch.as_tensor(camelyon_train_5_0_50)
        camelyon_train_5_0_50_dl = DataLoader(TensorDataset(camelyon_train_5_0_50,camelyon_train_5_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

        myTAE = TrainedAutoencoder(drift_detection_config)
        myTAE.init_default_pt_autoencoder(camelyon_train_5_0_50_dl)
        myTAE.init_detector(detector_type='KS',reference_data=camelyon_train_5_50_100,detector_name='camelyon_TAE_{}_KS'.format(i),save_dec=True)

        camelyon_train_5_comp= None
        camelyon_train_5 = None
        camelyon_train_5_0_50 = None 
        camelyon_train_5_50_100 = None
        camelyon_train_5_0_50_dl = None
        myTAE = None 

    for i in range(5,105,5):
        if i == 100:
            rxrx1_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_ds.npz'.format(i))
        else:
            rxrx1_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_{}_ds.npz'.format(i))
        rxrx1_train_5 = rxrx1_train_5_comp['arr_0']
        rxrx1_train_5_0_50 = rxrx1_train_5[:int(len(rxrx1_train_5)*0.5)]
        rxrx1_train_5_50_100 = rxrx1_train_5[int(len(rxrx1_train_5)*0.5):]

        rxrx1_train_5_0_50 = torch.as_tensor(rxrx1_train_5_0_50)
        rxrx1_train_5_0_50_dl = DataLoader(TensorDataset(rxrx1_train_5_0_50,rxrx1_train_5_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

        myTAE = TrainedAutoencoder(drift_detection_config)
        myTAE.init_default_pt_autoencoder(rxrx1_train_5_0_50_dl)
        myTAE.init_detector(detector_type='KS',reference_data=rxrx1_train_5_50_100,detector_name='rxrx1_TAE_{}_KS'.format(i),save_dec=True)

        rxrx1_train_5_comp= None
        rxrx1_train_5 = None
        rxrx1_train_5_0_50 = None 
        rxrx1_train_5_50_100 = None
        rxrx1_train_5_0_50_dl = None
        myTAE = None 


    for i in range(5,105,5):
        if i == 100:
            iwildcam_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_ds.npz')
        else:
            iwildcam_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_{}_ds.npz'.format(i))
        iwildcam_train_5 = iwildcam_train_5_comp['arr_0']
        iwildcam_train_5_0_50 = iwildcam_train_5[:int(len(iwildcam_train_5)*0.5)]
        iwildcam_train_5_50_100 = iwildcam_train_5[int(len(iwildcam_train_5)*0.5):]

        iwildcam_train_5_0_50 = torch.as_tensor(iwildcam_train_5_0_50)
        iwildcam_train_5_0_50_dl = DataLoader(TensorDataset(iwildcam_train_5_0_50,iwildcam_train_5_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

        myTAE = TrainedAutoencoder(drift_detection_config)
        myTAE.init_default_pt_autoencoder(iwildcam_train_5_0_50_dl)
        myTAE.init_detector(detector_type='KS',reference_data=iwildcam_train_5_50_100,detector_name='iwildcam_TAE_{}_KS'.format(i),save_dec=True)

        iwildcam_train_5_comp= None
        iwildcam_train_5 = None
        iwildcam_train_5_0_50 = None 
        iwildcam_train_5_50_100 = None
        iwildcam_train_5_0_50_dl = None
        myTAE = None 

    for i in range(5,105,5):
        if i == 100:
            globalwheat_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/globalwheat_train_ds.npz')
        else:
            globalwheat_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/globalwheat_train_{}_ds.npz'.format(i))
        globalwheat_train_5 = globalwheat_train_5_comp['arr_0']
        globalwheat_train_5_0_50 = globalwheat_train_5[:int(len(globalwheat_train_5)*0.5)]
        globalwheat_train_5_50_100 = globalwheat_train_5[int(len(globalwheat_train_5)*0.5):]

        globalwheat_train_5_0_50 = torch.as_tensor(globalwheat_train_5_0_50)
        globalwheat_train_5_0_50_dl = DataLoader(TensorDataset(globalwheat_train_5_0_50,globalwheat_train_5_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

        myTAE = TrainedAutoencoder(drift_detection_config)
        myTAE.init_default_pt_autoencoder(globalwheat_train_5_0_50_dl)
        myTAE.init_detector(detector_type='KS',reference_data=globalwheat_train_5_50_100,detector_name='globalwheat_TAE_{}_KS'.format(i),save_dec=True)

        globalwheat_train_5_comp= None
        globalwheat_train_5 = None
        globalwheat_train_5_0_50 = None 
        globalwheat_train_5_50_100 = None
        globalwheat_train_5_0_50_dl = None
        myTAE = None 

# ======================================================================================
# call
if __name__ == "__main__":
    main()