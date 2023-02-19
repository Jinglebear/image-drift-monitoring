import numpy as np    
import json 
import os
import sys
sys.path.append('/home/ubuntu/image-drift-monitoring/src')
import numpy as np
from modules.alibi_detect.untrained_encoder import UntrainedAutoencoder

import json 

import numpy as np

from torch.utils.data import TensorDataset, DataLoader
import json 
from timeit import default_timer as timer
import pandas as pd
import torch
def main():
        """ RECURRING """
        DATASET_NAME = 'iwildcam'
        
        with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
                drift_detection_config = json.load(config_file)

        data = {
        "50% train 50% init":["{}".format(i+333) for i in range(1,21,1)],
        }
        df_new = pd.DataFrame(data,index=["{}".format(i) for i in range(1,21,1)])

        for i in ['KS','CVM','MMD','LSDD']:
                t = timer()
                myUAE = UntrainedAutoencoder()
                myUAE.import_detector(path='{}/{}_uae_{}'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME,i),detector_type='{}'.format(i))
                for j in range(1,21,1):
                        test_j_comp = np.load('{}/{}_test_recurring_{}.npz'.format(drift_detection_config["PATHS"]["DATA_DIR_PATH"],DATASET_NAME,j))
                        test_j = test_j_comp['arr_0']
                        if j == 10:
                                res = myUAE.make_prediction(target_data=test_j, detector_type='{}'.format(i))
                                dt = timer() - t
                                with open('{}/track_time_UAE_{}_run_test_{}.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME,i),'w') as f:
                                        f.write(str(dt))
                        else:
                                res = myUAE.make_prediction(target_data=test_j, detector_type='{}'.format(i))
                        df_new.loc['{}'.format(j)]['{}'.format("50% train 50% init")] = res['data']['is_drift']
                        
                        print(len(res['data']['distance']))
                        
                df_new.to_excel('{}_uae_{}_results_recurring.xlsx'.format(DATASET_NAME,i))

        # """ INCREMENTAL """
        # DATASET_NAME = 'iwildcam'
        
        # with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
        #         drift_detection_config = json.load(config_file)

        # data = {
        # "50% train 50% init":["{}".format(i+333) for i in range(1,21,1)],
        # }
        # df_new = pd.DataFrame(data,index=["{}".format(i) for i in range(5,105,5)])

        # for i in ['KS','CVM','MMD','LSDD']:
        #         t = timer()
        #         myUAE = UntrainedAutoencoder()
        #         myUAE.import_detector(path='{}/{}_uae_{}'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME,i),detector_type='{}'.format(i))
        #         for j in range(5,105,5):
        #                 test_j_comp = np.load('{}/{}_test_incremental_{}.npz'.format(drift_detection_config["PATHS"]["DATA_DIR_PATH"],DATASET_NAME,j))
        #                 test_j = test_j_comp['arr_0']
        #                 if j == 5:
        #                         res = myUAE.make_prediction(target_data=test_j, detector_type='{}'.format(i))
        #                         dt = timer() - t
        #                         with open('{}/track_time_UAE_{}_run_test_{}.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME,i),'w') as f:
        #                                 f.write(str(dt))
        #                 else:
        #                         res = myUAE.make_prediction(target_data=test_j, detector_type='{}'.format(i))
        #                 df_new.loc['{}'.format(j)]['{}'.format("50% train 50% init")] = res['data']['is_drift']
                        
        #                 print(len(res['data']['distance']))
                        
        #         df_new.to_excel('{}_uae_{}_results_incremental.xlsx'.format(DATASET_NAME,i))


        # """ SUDDEN """
        # DATASET_NAME = 'iwildcam'
        
        # with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
        #         drift_detection_config = json.load(config_file)

        # data = {
        # "50% train 50% init":["{}".format(i+333) for i in range(1,21,1)],
        # }
        # df_new = pd.DataFrame(data,index=["{}".format(i) for i in range(10,210,10)])

        # for i in ['KS','CVM','MMD','LSDD']:
        #         t = timer()
        #         myUAE = UntrainedAutoencoder()
        #         myUAE.import_detector(path='{}/{}_uae_{}'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME,i),detector_type='{}'.format(i))
        #         for j in range(10,210,10):
        #                 test_j_comp = np.load('{}/{}_test_{}.npz'.format(drift_detection_config["PATHS"]["DATA_DIR_PATH"],DATASET_NAME,j))
        #                 test_j = test_j_comp['arr_0']
        #                 if j == 10:
        #                         res = myUAE.make_prediction(target_data=test_j, detector_type='{}'.format(i))
        #                         dt = timer() - t
        #                         with open('{}/track_time_UAE_{}_run_test_{}.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME,i),'w') as f:
        #                                 f.write(str(dt))
        #                 else:
        #                         res = myUAE.make_prediction(target_data=test_j, detector_type='{}'.format(i))
        #                 df_new.loc['{}'.format(j)]['{}'.format("50% train 50% init")] = res['data']['is_drift']
                        
        #                 print(len(res['data']['distance']))
                        
        #         df_new.to_excel('{}_uae_{}_results_sudden.xlsx'.format(DATASET_NAME,i))








        


# ======================================================================================
# call
if __name__ == "__main__":
    main()