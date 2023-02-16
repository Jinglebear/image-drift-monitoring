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
import torch
import pandas as pd
from timeit import default_timer as timer
import time
def main():

        time.sleep(4000)
        # globalwheat
        with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
                drift_detection_config = json.load(config_file)

        globalwheat_train_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/96_by_96_transform/globalwheat_96_train_ds.npz')
        globalwheat_train_i = globalwheat_train_i_comp['arr_0']

        size_train = globalwheat_train_i.shape[0]

        if(size_train > 5000):
                print('over 5000')
                np.random.shuffle(globalwheat_train_i)
        else:
                np.random.shuffle(globalwheat_train_i)
                # do 50 50 split
                globalwheat_train_i_0_50 =  globalwheat_train_i[:int(globalwheat_train_i.shape[0]*0.5)]
                size_training_set = globalwheat_train_i_0_50.shape[0]

                print(size_training_set)
                
                globalwheat_train_i_50_100 = globalwheat_train_i[int(globalwheat_train_i.shape[0]*0.5):]
                size_remaining = globalwheat_train_i_50_100.shape[0]

                print(size_remaining)

                globalwheat_train_i_0_50 = torch.as_tensor(globalwheat_train_i_0_50)
                
                globalwheat_train_i_0_50_dl = DataLoader(TensorDataset(globalwheat_train_i_0_50,globalwheat_train_i_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)
                
                t = timer() # setup timer
                myTAE = TrainedAutoencoder(drift_detection_config)
                myTAE.init_default_pt_autoencoder(globalwheat_train_i_0_50_dl)
                dt = timer() - t
                with open('/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/TAE/track_time_tae_globalwheat_trainingAE.txt','w') as f:
                        f.write(str(dt))
                t2 = timer()
                for i in ['KS','CVM','MMD','LSDD']:
                        myTAE.init_detector(detector_type='{}'.format(i),reference_data=globalwheat_train_i_50_100,detector_name='globalwheat_TAE_{}_{}_{}'.format(size_training_set,size_remaining,i),save_dec=True)
                        dt2 = timer() - t2
                        with open('/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/TAE/track_time_globalwheat_{}_init_50_p_train.txt'.format(i),'w') as f:
                                f.write(str(dt2))



        globalwheat_train_i_comp = None
        globalwheat_train_i = None
        globalwheat_train_i_0_50 = None 
        globalwheat_train_i_50_100 = None
        globalwheat_train_i_dl = None
        myTAE = None
       

        rxrx1_train_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/96_by_96_transform/rxrx1_96_train_ds.npz')
        rxrx1_train_i = rxrx1_train_i_comp['arr_0']

        size_train = rxrx1_train_i.shape[0]

        if(size_train > 5000):
                print('over 5000')
                np.random.shuffle(rxrx1_train_i)
                
                # do 50 50 split
                rxrx1_train_i_0_50 =  rxrx1_train_i[:5000]
                size_training_set = rxrx1_train_i_0_50.shape[0]

                print(size_training_set)
                
                rxrx1_train_i_50_100 = rxrx1_train_i[5000:]
                size_remaining = rxrx1_train_i_50_100.shape[0]

                print(size_remaining)

                rxrx1_train_i_0_50 = torch.as_tensor(rxrx1_train_i_0_50)
                
                rxrx1_train_i_0_50_dl = DataLoader(TensorDataset(rxrx1_train_i_0_50,rxrx1_train_i_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)
                t = timer() # setup timer
                myTAE = TrainedAutoencoder(drift_detection_config)
                myTAE.init_default_pt_autoencoder(rxrx1_train_i_0_50_dl)
                dt = timer() - t
                with open('/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/TAE/track_time_tae_rxrx1_trainingAE.txt','w') as f:
                        f.write(str(dt))
                for i in ['KS','CVM','MMD','LSDD']:
                        t2 = timer()
                        myTAE.init_detector(detector_type='{}'.format(i),reference_data=rxrx1_train_i_50_100,detector_name='rxrx1_TAE_{}_{}_{}'.format(size_training_set,size_remaining,i),save_dec=True)
                        dt2 = timer()-t2
                        with open('/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/TAE/track_time_rxrx1_{}_init_50_p_train.txt'.format(i),'w') as f:
                                f.write(str(dt2))



        rxrx1_train_i_comp = None
        rxrx1_train_i = None
        rxrx1_train_i_0_50 = None 
        rxrx1_train_i_50_100 = None
        rxrx1_train_i_dl = None
        myTAE = None

        iwildcam_train_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/96_by_96_transform/iwildcam_96_train_ds.npz')
        iwildcam_train_i = iwildcam_train_i_comp['arr_0']

        size_train = iwildcam_train_i.shape[0]

        if(size_train > 5000):
                print('over 5000')
                np.random.shuffle(iwildcam_train_i)
                # do 50 50 split
                iwildcam_train_i_0_50 =  iwildcam_train_i[:5000]
                size_training_set = iwildcam_train_i_0_50.shape[0]

                print(size_training_set)
                
                iwildcam_train_i_50_100 = iwildcam_train_i[5000:]
                size_remaining = iwildcam_train_i_50_100.shape[0]

                print(size_remaining)

                iwildcam_train_i_0_50 = torch.as_tensor(iwildcam_train_i_0_50)
                
                iwildcam_train_i_0_50_dl = DataLoader(TensorDataset(iwildcam_train_i_0_50,iwildcam_train_i_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)
                t = timer() # setup timer
                myTAE = TrainedAutoencoder(drift_detection_config)
                myTAE.init_default_pt_autoencoder(iwildcam_train_i_0_50_dl)
                with open('/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/TAE/track_time_tae_iwildcam_trainingAE.txt','w') as f:
                        f.write(str(dt))
                for i in ['KS','CVM','MMD','LSDD']:
                        t2 = timer()
                        myTAE.init_detector(detector_type='{}'.format(i),reference_data=iwildcam_train_i_50_100,detector_name='iwildcam_TAE_{}_{}_{}'.format(size_training_set,size_remaining,i),save_dec=True)
                        dt2 = timer - t2
                        with open('/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/TAE/track_time_iwildcam_{}_init_50_p_train.txt'.format(i),'w') as f:
                                f.write(str(dt2))


        iwildcam_train_i_comp = None
        iwildcam_train_i = None
        iwildcam_train_i_0_50 = None 
        iwildcam_train_i_50_100 = None
        iwildcam_train_i_dl = None
        myTAE = None


# ======================================================================================
# call
if __name__ == "__main__":
    main()