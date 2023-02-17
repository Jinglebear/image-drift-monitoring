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

        globalwheat_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/96_by_96_transform/globalwheat_96_train_ds.npz')
        globalwheat_train = globalwheat_train_comp['arr_0']

        size_training = globalwheat_train.shape[0]

        np.random.shuffle(globalwheat_train)
        
        t = timer() # setup timer
        myUAE = UntrainedAutoencoder(drift_detection_config)
        dt = timer() - t
        with open('{}/track_time_UAE_globalwheat_trainingUAE.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"]),'w') as f:
                f.write(str(dt))
        t2 = timer()
        for i in ['KS','CVM','MMD','LSDD']:
                myUAE.init_detector(detector_type='{}'.format(i),reference_data=globalwheat_train,detector_name='globalwheat_UAE_n_50_{}_{}'.format(size_training,i),save_dec=True)
                dt2 = timer() - t2
                with open('{}/track_time_globalwheat_{}_init_{}_p_train.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],i,size_training),'w') as f:
                        f.write(str(dt2))




    

        # rxrx1_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/96_by_96_transform/rxrx1_96_train_ds.npz')
        # rxrx1_train = rxrx1_train_comp['arr_0']

        # np.random.shuffle(rxrx1_train)

        # t = timer() # setup timer
        # myUAE = UntrainedAutoencoder(drift_detection_config)
        # dt = timer() - t
        # with open('{}/track_time_UAE_rxrx1_trainingUAE.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"]),'w') as f:
        #         f.write(str(dt))

        # size_training = rxrx1_train.shape[0]
        # for i in ['KS','CVM','MMD','LSDD']:
        #         t2 = timer()
        #         myUAE.init_detector(detector_type='{}'.format(i),reference_data=rxrx1_train,detector_name='rxrx1_UAE_{}_{}'.format(size_training,i),save_dec=True)
        #         dt2 = timer()-t2
        #         with open('{}/track_time_rxrx1_{}_init_{}_train.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],i,size_training),'w') as f:
        #                 f.write(str(dt2))


       

        # iwildcam_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/96_by_96_transform/iwildcam_96_train_ds.npz')
        # iwildcam_train = iwildcam_train_comp['arr_0']
        
        # np.random.shuffle(iwildcam_train)

        # t = timer() # setup timer
        # myUAE = UntrainedAutoencoder(drift_detection_config)
        # dt = timer() - t
        # with open('{}/track_time_UAE_iwildcam_trainingUAE.txt'.format(drift_detection_config['PATHS']['DETECTOR_DIR_PATH']),'w') as f:
        #         f.write(str(dt))
        # for i in ['KS','CVM','MMD','LSDD']:
        #         if i == 'MMD' or i == 'LSDD':
        #                 iwildcam_train_init = iwildcam_train[:int(len(iwildcam_train)*0.35)]
                        
        #         else:
        #                 iwildcam_train_init = iwildcam_train

        #         size_training = iwildcam_train_init.shape[0]
        #         t2 = timer()
        #         myUAE.init_detector(detector_type='{}'.format(i),reference_data=iwildcam_train_init,detector_name='iwildcam_UAE_{}_{}'.format(size_training,i),save_dec=True)
        #         dt2 = timer() - t2
        #         with open('{}/track_time_iwildcam_{}_init_{}_train.txt'.format(drift_detection_config['PATHS']['DETECTOR_DIR_PATH'],i,size_training),'w') as f:
        #                 f.write(str(dt2))


     

        # camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
        # camelyon_train = camelyon_train_comp['arr_0']


        # np.random.shuffle(camelyon_train)
        
        
        # t = timer() # setup timer
        # myUAE = UntrainedAutoencoder(drift_detection_config)
        # dt = timer() - t
        # with open('{}/track_time_UAE_camelyon_trainingUAE.txt'.format(drift_detection_config['PATHS']['DETECTOR_DIR_PATH']),'w') as f:
        #         f.write(str(dt))
        # for i in ['KS','CVM','MMD','LSDD']:     
        #         if i == 'MMD' or i == 'LSDD':
        #                 camelyon_train_init = camelyon_train[:int(len(camelyon_train)*0.15)]
        #         else:
        #                 camelyon_train_init = camelyon_train
        #         size_training = camelyon_train_init.shape[0]
        #         t2 = timer()
        #         myUAE.init_detector(detector_type='{}'.format(i),reference_data=camelyon_train_init,detector_name='camelyon_UAE_{}_{}'.format(size_training,i),save_dec=True)
        #         dt2 = timer() - t2
        #         with open('{}/track_time_camelyon_{}_init_{}_train.txt'.format(drift_detection_config['PATHS']['DETECTOR_DIR_PATH'],i,size_training),'w') as f:
        #                 f.write(str(dt2))
    
# ======================================================================================
# call
if __name__ == "__main__":
    main()