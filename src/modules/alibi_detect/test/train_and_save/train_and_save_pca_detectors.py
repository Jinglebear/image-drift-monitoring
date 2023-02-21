import json
import sys

import numpy as np

sys.path.append('/home/ubuntu/image-drift-monitoring/src')
import json
from timeit import default_timer as timer

from modules.alibi_detect.principal_component_analysis import \
    PrincipalComponentAnalysis


def main():



        with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
                drift_detection_config = json.load(config_file)

        DATASET_NAME =''


        data_train_comp = np.load(drift_detection_config["PATHS"]["DATA_DIR_PATH"])
        data_train = data_train_comp['arr_0']
      
        np.random.shuffle(data_train)
        # do 50 50 split
        data_train_0_50 =  data_train[:int(data_train.shape[0]*0.5)]
        size_training_set = data_train_0_50.shape[0]

        print(size_training_set)
        
        data_train_50_100 = data_train[int(data_train.shape[0]*0.5):]
        size_remaining = data_train_50_100.shape[0]

        print(size_remaining)

        t = timer() # setup timer
        myPCA = PrincipalComponentAnalysis(drift_detection_config)
        myPCA.init_pca(data_train_0_50)
        dt = timer() - t
        with open('{}/track_time_pca_{}_trainingPCA.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME),'w') as f:
                f.write(str(dt))
        t2 = timer()

        for i in ['KS','CVM','MMD','LSDD']:
                # 50 / 15 split 
                if DATASET_NAME == 'camelyon' and (i == 'MMD' or i == 'LSDD'):
                        data_train_50_100 = data_train[int(data_train.shape[0]*0.5):int(data_train.shape[0]*0.65)]
                        size_remaining = data_train_50_100.shape[0]
                # 50 / 35 split
                if DATASET_NAME == 'iwildcam' and (i == 'MMD' or i == 'LSDD'):
                        data_train_50_100 = data_train[int(data_train.shape[0]*0.5):int(data_train.shape[0]*0.85)]
                        size_remaining = data_train_50_100.shape[0]
                myPCA.init_detector(detector_type='{}'.format(i),reference_data=data_train_50_100,detector_name='{}_PCA_n_32_{}_{}_{}'.format(DATASET_NAME,size_training_set,size_remaining,i),save_dec=True)
                dt2 = timer() - t2
                with open('{}/track_time_{}_{}_init_{}_p_train.txt'.format(drift_detection_config["PATHS"]["DETECTOR_DIR_PATH"],DATASET_NAME,i,size_remaining),'w') as f:
                        f.write(str(dt2))

# ======================================================================================
# call
if __name__ == "__main__":
    main()