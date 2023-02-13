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

    data = {
            "100":["{}".format(i) for i in range(1,21,1)],
            "95" : ["{}".format(i) for i in range(1,21,1)],
            "90" : ["{}".format(i) for i in range(1,21,1)],
            "85" : ["{}".format(i) for i in range(1,21,1)],
            "80" : ["{}".format(i) for i in range(1,21,1)],
            "75" : ["{}".format(i) for i in range(1,21,1)],
            "70" : ["{}".format(i) for i in range(1,21,1)],
            "65" : ["{}".format(i) for i in range(1,21,1)],
            "60" : ["{}".format(i) for i in range(1,21,1)],
            "55" : ["{}".format(i) for i in range(1,21,1)],
            "50" : ["{}".format(i) for i in range(1,21,1)],
            "45" : ["{}".format(i) for i in range(1,21,1)],
            "40" : ["{}".format(i) for i in range(1,21,1)],
            "35" : ["{}".format(i) for i in range(1,21,1)],
            "30" : ["{}".format(i) for i in range(1,21,1)],
            "25" : ["{}".format(i) for i in range(1,21,1)],
            "20" : ["{}".format(i) for i in range(1,21,1)],
            "15" : ["{}".format(i) for i in range(1,21,1)],
            "10" : ["{}".format(i) for i in range(1,21,1)],
            "5" : ["{}".format(i) for i in range(1,21,1)],
            }
    df_new = pd.DataFrame(data,index=["Size {}".format(i) for i in range(10,210,10)])


    for i in range(10,210,10):
        test_i_comp = np.load('/home/ubuntu/image-drift-monitoring/data/globalwheat_v1.1/drifted_data/sudden_drift/globalwheat_test_{}.npz'.format(i))
        test_i = test_i_comp['arr_0']
        for j in range(5,105,5):
            myUAE = UntrainedAutoencoder()
            myUAE.import_detector(path='/home/ubuntu/image-drift-monitoring/config/detectors/GlobalWheat/UAE/LSDD/globalwheat_UAE_{}_LSDD'.format(j),detector_type='LSDD')
            res = myUAE.make_prediction(target_data=test_i, detector_type='LSDD')

            df_new.loc['Size {}'.format(i)]['{}'.format(j)] = res['data']['is_drift']
    df_new.to_excel('globalwheat_uae_lsdd_results.xlsx')
# ======================================================================================
# call
if __name__ == "__main__":
    main()