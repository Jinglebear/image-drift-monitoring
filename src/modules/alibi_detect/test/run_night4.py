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
        

        data = {
        "100":["{}".format(i) for i in range(1,21,1)],
        "80" : ["{}".format(i) for i in range(1,21,1)],
        "60" : ["{}".format(i) for i in range(1,21,1)],
        "40" : ["{}".format(i) for i in range(1,21,1)],
        "20" : ["{}".format(i) for i in range(1,21,1)],
        }
        df_new = pd.DataFrame(data,index=["{}% OOD Bilder".format(i) for i in range(5,105,5)])


        for j in range(20,120,20):
                myUAE = UntrainedAutoencoder()
                myUAE.import_detector(path='/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/UAE/KS/globalwheat_UAE_{}_KS'.format(j),detector_type='KS')
                for i in range(5,105,5):
                        test_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/drifted_data/incremental_drift/globalwheat_test_incremental_{}.npz'.format(i))
                        test_i = test_i_comp['arr_0']
                        res = myUAE.make_prediction(target_data=test_i, detector_type='KS')
                        df_new.loc['{}% OOD Bilder'.format(i)]['{}'.format(j)] = res['data']['is_drift']
        df_new.to_excel('globalwheat_uae_ks_results_incremental.xlsx')

        for j in range(20,120,20):
                myUAE = UntrainedAutoencoder()
                myUAE.import_detector(path='/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/UAE/CVM/globalwheat_UAE_{}_CVM'.format(j),detector_type='CVM')
                for i in range(5,105,5):
                        test_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/drifted_data/incremental_drift/globalwheat_test_incremental_{}.npz'.format(i))
                        test_i = test_i_comp['arr_0']
                        res = myUAE.make_prediction(target_data=test_i, detector_type='CVM')
                        df_new.loc['{}% OOD Bilder'.format(i)]['{}'.format(j)] = res['data']['is_drift']
        df_new.to_excel('globalwheat_uae_cvm_results_incremental.xlsx')

        for j in range(20,120,20):
                myUAE = UntrainedAutoencoder()
                myUAE.import_detector(path='/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/UAE/MMD/globalwheat_UAE_{}_MMD'.format(j),detector_type='MMD')
                for i in range(5,105,5):
                        test_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/drifted_data/incremental_drift/globalwheat_test_incremental_{}.npz'.format(i))
                        test_i = test_i_comp['arr_0']
                        res = myUAE.make_prediction(target_data=test_i, detector_type='MMD')
                        df_new.loc['{}% OOD Bilder'.format(i)]['{}'.format(j)] = res['data']['is_drift']
        df_new.to_excel('globalwheat_uae_mmd_results_incremental.xlsx')

        for j in range(20,120,20):
                myUAE = UntrainedAutoencoder()
                myUAE.import_detector(path='/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/UAE/LSDD/globalwheat_UAE_{}_LSDD'.format(j),detector_type='LSDD')
                for i in range(5,105,5):
                        test_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/drifted_data/incremental_drift/globalwheat_test_incremental_{}.npz'.format(i))
                        test_i = test_i_comp['arr_0']
                        res = myUAE.make_prediction(target_data=test_i, detector_type='LSDD')
                        df_new.loc['{}% OOD Bilder'.format(i)]['{}'.format(j)] = res['data']['is_drift']
        df_new.to_excel('globalwheat_uae_lsdd_results_incremental.xlsx')


# ======================================================================================
# call
if __name__ == "__main__":
    main()