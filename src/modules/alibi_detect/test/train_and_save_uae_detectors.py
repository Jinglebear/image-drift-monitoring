import numpy as np    
import json 
import os
import sys
sys.path.append('/home/ubuntu/image-drift-monitoring/src')
import numpy as np
from modules.alibi_detect.untrained_encoder import UntrainedAutoencoder
import json 

import numpy as np
from modules.alibi_detect.principal_component_analysis import PrincipalComponentAnalysis
from modules.alibi_detect.trained_autoencoder import TrainedAutoencoder
from torch.utils.data import TensorDataset, DataLoader
import json 
import torch
import pandas as pd
from timeit import default_timer as timer
def main():

        with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
                drift_detection_config = json.load(config_file)

        DATASET_NAME ='camelyon'


        data_train_comp = np.load(drift_detection_config["PATHS"]["DATA_DIR_PATH"])
        data_train = data_train_comp['arr_0']
        size_training = data_train.shape[0]
        np.random.shuffle(data_train)

        data_train_init = data_train
        
        t = timer() # setup timer
        myUAE = UntrainedAutoencoder(drift_detection_config)
        myUAE.init_default_tf_encoder(encoding_dim=drift_detection_config["UAE"]["ENC_DIM"],input_shape=data_train.shape[1:])
        dt = timer() - t
        with open('{}/track_time_UAE_{}_trainingUAE.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME),'w') as f:
                f.write(str(dt))
        t2 = timer()
        for i in ['KS','CVM','MMD','LSDD']:
                # 50 / 15 split 
                if DATASET_NAME == 'camelyon' and (i == 'MMD' or i == 'LSDD'):
                        data_train_init = data_train[:int(data_train.shape[0]*0.15)]
                        size_training = data_train_init.shape[0]
                # 50 / 35 split
                if DATASET_NAME == 'iwildcam' and (i == 'MMD' or i == 'LSDD'):
                        data_train_init = data_train[:int(data_train.shape[0]*0.35)]
                        size_training = data_train_init.shape[0]
                myUAE.init_detector(detector_type='{}'.format(i),reference_data=data_train_init,detector_name='{}_UAE_n_32_{}_{}'.format(DATASET_NAME,size_training,i),save_dec=True)
                dt2 = timer() - t2
                with open('{}/track_time_{}_{}_init_{}.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME,i,size_training),'w') as f:
                        f.write(str(dt2))



        data_train_comp = None
        data_train = None
        myUAE = None
# ======================================================================================
# call
if __name__ == "__main__":
    main()