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
def main():
 
        data = {
        "15":["{}".format(i) for i in range(1,21,1)],
        "10" : ["{}".format(i) for i in range(1,21,1)],
        "5" : ["{}".format(i) for i in range(1,21,1)]
        }
        df_new = pd.DataFrame(data,index=["{} OOD Bilder".format(i) for i in range(10,210,10)])

        for j in range(5,20,5):
                myUAE = UntrainedAutoencoder()
                myUAE.import_detector(path='/home/ubuntu/image-drift-monitoring/config/detectors/Camelyon/UAE/MMD/Camelyon_UAE_{}_MMD'.format(j),detector_type='MMD')
                for i in range(10,210,10):
                        test_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/drifted_data/sudden_drift/camelyon_test_{}.npz'.format(i))
                        test_i = test_i_comp['arr_0']
                        res = myUAE.make_prediction(target_data=test_i, detector_type='MMD')
                        df_new.loc['{} OOD Bilder'.format(i)]['{}'.format(j)] = res['data']['is_drift']
        df_new.to_excel('camelyon_uae_mmd_results_sudden.xlsx')

        data = {
        "15":["{}".format(i) for i in range(1,21,1)],
        "10" : ["{}".format(i) for i in range(1,21,1)],
        "5" : ["{}".format(i) for i in range(1,21,1)]
        }
        df_new = pd.DataFrame(data,index=["{} OOD Bilder".format(i) for i in range(10,210,10)])

        for j in range(5,20,5):
                myUAE = UntrainedAutoencoder()
                myUAE.import_detector(path='/home/ubuntu/image-drift-monitoring/config/detectors/Camelyon/UAE/LSDD/Camelyon_UAE_{}_LSDD'.format(j),detector_type='LSDD')
                for i in range(10,210,10):
                        test_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/drifted_data/sudden_drift/camelyon_test_{}.npz'.format(i))
                        test_i = test_i_comp['arr_0']
                        res = myUAE.make_prediction(target_data=test_i, detector_type='LSDD')
                        df_new.loc['{} OOD Bilder'.format(i)]['{}'.format(j)] = res['data']['is_drift']
        df_new.to_excel('camelyon_uae_lsdd_results_sudden.xlsx')

# ======================================================================================
# call
if __name__ == "__main__":
    main()