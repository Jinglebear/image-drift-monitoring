import numpy as np    
import json 
import os
import sys
sys.path.append('/home/ubuntu/image-drift-monitoring/src')
import numpy as np
from modules.alibi_detect.untrained_encoder import UntrainedAutoencoder
from modules.alibi_detect.principal_component_analysis import PrincipalComponentAnalysis
import json 

import numpy as np
from modules.alibi_detect.trained_autoencoder import TrainedAutoencoder
from torch.utils.data import TensorDataset, DataLoader
import json 

import pandas as pd
import torch
def main():
        with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
                drift_detection_config = json.load(config_file)

        for i in range(5,105,5):
                if(i == 100):
                        rxrx1_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_ds.npz')
                else: 
                        rxrx1_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_{}_ds.npz'.format(i))
                rxrx1_train = rxrx1_train_comp['arr_0']

                rxrx1_train_0_50   = rxrx1_train[:int(len(rxrx1_train)*0.5)]
                rxrx1_train_50_100 = rxrx1_train[ int(len(rxrx1_train)*0.5):]

                myPCA = PrincipalComponentAnalysis(drift_detection_config)
                myPCA.init_pca(x_ref=rxrx1_train_0_50)
                myPCA.init_detector(detector_type='LSDD',reference_data=rxrx1_train_50_100,detector_name='rxrx1_PCA_{}_LSDD'.format(i),save_dec=True)

                rxrx1_train_comp = None
                rxrx1_train = None
                rxrx1_train_0_50 = None
                rxrx1_train_50_100 =None
                myPCA = None
# ======================================================================================
# call
if __name__ == "__main__":
    main()