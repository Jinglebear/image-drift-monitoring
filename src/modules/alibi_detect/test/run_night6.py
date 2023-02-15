import numpy as np    
import json 
import os
import sys
sys.path.append('/home/ubuntu/image-drift-monitoring/src')
sys.path.append('/home/ubuntu/image-drift-monitoring')
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
            "5000":["{}".format(i) for i in range(1,21,1)],
            }
    df_new = pd.DataFrame(data,index=["{} OOD Bilder".format(i) for i in range(10,210,10)])

    myTAE = TrainedAutoencoder()
    myTAE.import_detector(path='/home/ubuntu/image-drift-monitoring/config/detectors/Camelyon/TAE/MMD/camelyon_TAE_5000_10121_MMD',detector_type='MMD')

    for i in range(10,210,10):
        test_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/drifted_data/sudden_drift/camelyon_test_{}.npz'.format(i))
        test_i = test_i_comp['arr_0']
        res = myTAE.make_prediction(target_data=test_i, detector_type='MMD')
        df_new.loc['{} OOD Bilder'.format(i)]['{}'.format(5000)] = res['data']['is_drift']

    df_new.to_excel('camelyon_tae_mmd_results_sudden.xlsx')

# ======================================================================================
# call
if __name__ == "__main__":
    main()