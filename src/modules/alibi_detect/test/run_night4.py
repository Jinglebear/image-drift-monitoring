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
def main():
    with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
            drift_detection_config = json.load(config_file)

    for i in range(5,105,5):
        if(i == 100):
            camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
        else: 
            camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_{}_ds.npz'.format(i))
        
        camelyon_train = camelyon_train_comp['arr_0']

        camelyon_train_0_50   = camelyon_train[:int(len(camelyon_train)*0.5)]
        camelyon_train_50_100 = camelyon_train[ int(len(camelyon_train)*0.5):]

        myPCA = PrincipalComponentAnalysis(drift_detection_config)
        myPCA.init_pca(x_ref=camelyon_train_0_50)
        myPCA.init_detector(detector_type='KS',reference_data=camelyon_train_50_100,detector_name='camelyon_PCA_{}_KS'.format(i),save_dec=True)

        camelyon_train_comp = None
        camelyon_train = None
        camelyon_train_0_50 = None
        camelyon_train_50_100 =None
        myPCA = None

    for i in range(5,105,5):
        if(i == 100):
                camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
        else: 
                camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_{}_ds.npz'.format(i))
        
        camelyon_train = camelyon_train_comp['arr_0']

        camelyon_train_0_50   = camelyon_train[:int(len(camelyon_train)*0.5)]
        camelyon_train_50_100 = camelyon_train[ int(len(camelyon_train)*0.5):]

        myPCA = PrincipalComponentAnalysis(drift_detection_config)
        myPCA.init_pca(x_ref=camelyon_train_0_50)
        myPCA.init_detector(detector_type='CVM',reference_data=camelyon_train_50_100,detector_name='camelyon_PCA_{}_CVM'.format(i),save_dec=True)

        camelyon_train_comp = None
        camelyon_train = None
        camelyon_train_0_50 = None
        camelyon_train_50_100 =None
        myPCA = None


    for i in range(5,105,5):
        if(i == 100):
                camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
        else: 
                camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_{}_ds.npz'.format(i))

        camelyon_train = camelyon_train_comp['arr_0']

        camelyon_train_0_50   = camelyon_train[:int(len(camelyon_train)*0.5)]
        camelyon_train_50_100 = camelyon_train[ int(len(camelyon_train)*0.5):]

        myPCA = PrincipalComponentAnalysis(drift_detection_config)
        myPCA.init_pca(x_ref=camelyon_train_0_50)
        myPCA.init_detector(detector_type='MMD',reference_data=camelyon_train_50_100,detector_name='camelyon_PCA_{}_MMD'.format(i),save_dec=True)

        camelyon_train_comp = None
        camelyon_train = None
        camelyon_train_0_50 = None
        camelyon_train_50_100 =None
        myPCA = None
    for i in range(5,105,5):
        if(i == 100):
                camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
        else: 
                camelyon_train_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_{}_ds.npz'.format(i))
        camelyon_train = camelyon_train_comp['arr_0']

        camelyon_train_0_50   = camelyon_train[:int(len(camelyon_train)*0.5)]
        camelyon_train_50_100 = camelyon_train[ int(len(camelyon_train)*0.5):]

        myPCA = PrincipalComponentAnalysis(drift_detection_config)
        myPCA.init_pca(x_ref=camelyon_train_0_50)
        myPCA.init_detector(detector_type='LSDD',reference_data=camelyon_train_50_100,detector_name='camelyon_PCA_{}_LSDD'.format(i),save_dec=True)

        camelyon_train_comp = None
        camelyon_train = None
        camelyon_train_0_50 = None
        camelyon_train_50_100 =None
        myPCA = None



# ======================================================================================
# call
if __name__ == "__main__":
    main()