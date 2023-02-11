import numpy as np    
import json 
import os
import sys
sys.path.append('/home/ubuntu/image-drift-monitoring/src')
import numpy as np
from modules.alibi_detect.untrained_encoder import UntrainedAutoencoder
import json 

import numpy as np
from modules.alibi_detect.trained_autoencoder import TrainedAutoencoder
from torch.utils.data import TensorDataset, DataLoader
import json 
import torch
def main():
    



    with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
            drift_detection_config = json.load(config_file)



    iwildcam_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_5_ds.npz')
    iwildcam_train_5 = iwildcam_train_5_comp['arr_0']

    iwildcam_train_5.shape
    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_5.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_5,detector_name='iwildcam_UAE_5_KS',save_dec=True)

    iwildcam_train_5_comp = None
    iwildcam_train_5 = None
    myUAE = None


    iwildcam_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_10_ds.npz')
    iwildcam_train_10 = iwildcam_train_10_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_10.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_10,detector_name='iwildcam_UAE_10_KS',save_dec=True)

    iwildcam_train_10_comp = None
    iwildcam_train_10 = None
    myUAE = None

    iwildcam_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_15_ds.npz')
    iwildcam_train_15 = iwildcam_train_15_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_15.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_15,detector_name='iwildcam_UAE_15_KS',save_dec=True)

    iwildcam_train_15_comp = None
    iwildcam_train_15 = None
    myUAE = None

    iwildcam_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_20_ds.npz')
    iwildcam_train_20 = iwildcam_train_20_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_20.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_20,detector_name='iwildcam_UAE_20_KS',save_dec=True)

    iwildcam_train_20_comp = None
    iwildcam_train_20 = None
    myUAE = None

    iwildcam_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_25_ds.npz')
    iwildcam_train_25 = iwildcam_train_25_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_25.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_25,detector_name='iwildcam_UAE_25_KS',save_dec=True)

    iwildcam_train_25_comp = None
    iwildcam_train_25 = None
    myUAE = None

    iwildcam_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_30_ds.npz')
    iwildcam_train_30 = iwildcam_train_30_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_30.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_30,detector_name='iwildcam_UAE_30_KS',save_dec=True)

    iwildcam_train_30_comp = None
    iwildcam_train_30 = None
    myUAE = None

    iwildcam_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_35_ds.npz')
    iwildcam_train_35 = iwildcam_train_35_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_35.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_35,detector_name='iwildcam_UAE_35_KS',save_dec=True)

    iwildcam_train_35_comp = None
    iwildcam_train_35 = None
    myUAE = None

    iwildcam_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_40_ds.npz')
    iwildcam_train_40 = iwildcam_train_40_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_40.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_40,detector_name='iwildcam_UAE_40_KS',save_dec=True)

    iwildcam_train_40_comp = None
    iwildcam_train_40 = None
    myUAE = None

    iwildcam_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_45_ds.npz')
    iwildcam_train_45 = iwildcam_train_45_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_45.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_45,detector_name='iwildcam_UAE_45_KS',save_dec=True)

    iwildcam_train_45_comp = None
    iwildcam_train_45 = None
    myUAE = None

    iwildcam_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_50_ds.npz')
    iwildcam_train_50 = iwildcam_train_50_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_50.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_50,detector_name='iwildcam_UAE_50_KS',save_dec=True)

    iwildcam_train_50_comp = None
    iwildcam_train_50 = None
    myUAE = None

    iwildcam_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_55_ds.npz')
    iwildcam_train_55 = iwildcam_train_55_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_55.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_55,detector_name='iwildcam_UAE_55_KS',save_dec=True)

    iwildcam_train_55_comp = None
    iwildcam_train_55 = None
    myUAE = None


    iwildcam_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_60_ds.npz')
    iwildcam_train_60 = iwildcam_train_60_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_60.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_60,detector_name='iwildcam_UAE_60_KS',save_dec=True)

    iwildcam_train_60_comp = None
    iwildcam_train_60 = None
    myUAE = None

    iwildcam_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_65_ds.npz')
    iwildcam_train_65 = iwildcam_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_65.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_65,detector_name='iwildcam_UAE_65_KS',save_dec=True)

    iwildcam_train_65_comp = None
    iwildcam_train_65 = None
    myUAE = None

    iwildcam_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_70_ds.npz')
    iwildcam_train_70 = iwildcam_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_70.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_70,detector_name='iwildcam_UAE_70_KS',save_dec=True)

    iwildcam_train_70_comp = None
    iwildcam_train_70 = None
    myUAE = None

    iwildcam_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_75_ds.npz')
    iwildcam_train_75 = iwildcam_train_75_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_75.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_75,detector_name='iwildcam_UAE_75_KS',save_dec=True)

    iwildcam_train_75_comp = None
    iwildcam_train_75 = None
    myUAE = None

    iwildcam_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_80_ds.npz')
    iwildcam_train_80 = iwildcam_train_80_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_80.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_80,detector_name='iwildcam_UAE_80_KS',save_dec=True)

    iwildcam_train_80_comp = None
    iwildcam_train_80 = None
    myUAE = None

    iwildcam_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_85_ds.npz')
    iwildcam_train_85 = iwildcam_train_85_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_85.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_85,detector_name='iwildcam_UAE_85_KS',save_dec=True)

    iwildcam_train_85_comp = None
    iwildcam_train_85 = None
    myUAE = None

    iwildcam_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_90_ds.npz')
    iwildcam_train_90 = iwildcam_train_90_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_90.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_90,detector_name='iwildcam_UAE_90_KS',save_dec=True)

    iwildcam_train_90_comp = None
    iwildcam_train_90 = None
    myUAE = None

    iwildcam_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_95_ds.npz')
    iwildcam_train_95 = iwildcam_train_95_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_95.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_95,detector_name='iwildcam_UAE_95_KS',save_dec=True)

    iwildcam_train_95_comp = None
    iwildcam_train_95 = None
    myUAE = None

    iwildcam_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_ds.npz')
    iwildcam_train_100 = iwildcam_train_100_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_100.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=iwildcam_train_100,detector_name='iwildcam_UAE_100_KS',save_dec=True)

    iwildcam_train_100_comp = None
    iwildcam_train_100 = None
    myUAE = None


    iwildcam_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_5_ds.npz')
    iwildcam_train_5 = iwildcam_train_5_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_5.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_5,detector_name='iwildcam_UAE_5_CVM',save_dec=True)

    iwildcam_train_5_comp = None
    iwildcam_train_5 = None
    myUAE = None


    iwildcam_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_10_ds.npz')
    iwildcam_train_10 = iwildcam_train_10_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_10.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_10,detector_name='iwildcam_UAE_10_CVM',save_dec=True)

    iwildcam_train_10_comp = None
    iwildcam_train_10 = None
    myUAE = None

    iwildcam_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_15_ds.npz')
    iwildcam_train_15 = iwildcam_train_15_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_15.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_15,detector_name='iwildcam_UAE_15_CVM',save_dec=True)

    iwildcam_train_15_comp = None
    iwildcam_train_15 = None
    myUAE = None

    iwildcam_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_20_ds.npz')
    iwildcam_train_20 = iwildcam_train_20_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_20.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_20,detector_name='iwildcam_UAE_20_CVM',save_dec=True)

    iwildcam_train_20_comp = None
    iwildcam_train_20 = None
    myUAE = None

    iwildcam_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_25_ds.npz')
    iwildcam_train_25 = iwildcam_train_25_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_25.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_25,detector_name='iwildcam_UAE_25_CVM',save_dec=True)

    iwildcam_train_25_comp = None
    iwildcam_train_25 = None
    myUAE = None

    iwildcam_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_30_ds.npz')
    iwildcam_train_30 = iwildcam_train_30_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_30.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_30,detector_name='iwildcam_UAE_30_CVM',save_dec=True)

    iwildcam_train_30_comp = None
    iwildcam_train_30 = None
    myUAE = None

    iwildcam_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_35_ds.npz')
    iwildcam_train_35 = iwildcam_train_35_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_35.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_35,detector_name='iwildcam_UAE_35_CVM',save_dec=True)

    iwildcam_train_35_comp = None
    iwildcam_train_35 = None
    myUAE = None

    iwildcam_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_40_ds.npz')
    iwildcam_train_40 = iwildcam_train_40_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_40.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_40,detector_name='iwildcam_UAE_40_CVM',save_dec=True)

    iwildcam_train_40_comp = None
    iwildcam_train_40 = None
    myUAE = None

    iwildcam_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_45_ds.npz')
    iwildcam_train_45 = iwildcam_train_45_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_45.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_45,detector_name='iwildcam_UAE_45_CVM',save_dec=True)

    iwildcam_train_45_comp = None
    iwildcam_train_45 = None
    myUAE = None

    iwildcam_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_50_ds.npz')
    iwildcam_train_50 = iwildcam_train_50_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_50.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_50,detector_name='iwildcam_UAE_50_CVM',save_dec=True)

    iwildcam_train_50_comp = None
    iwildcam_train_50 = None
    myUAE = None

    iwildcam_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_55_ds.npz')
    iwildcam_train_55 = iwildcam_train_55_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_55.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_55,detector_name='iwildcam_UAE_55_CVM',save_dec=True)

    iwildcam_train_55_comp = None
    iwildcam_train_55 = None
    myUAE = None


    iwildcam_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_60_ds.npz')
    iwildcam_train_60 = iwildcam_train_60_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_60.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_60,detector_name='iwildcam_UAE_60_CVM',save_dec=True)

    iwildcam_train_60_comp = None
    iwildcam_train_60 = None
    myUAE = None

    iwildcam_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_65_ds.npz')
    iwildcam_train_65 = iwildcam_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_65.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_65,detector_name='iwildcam_UAE_65_CVM',save_dec=True)

    iwildcam_train_65_comp = None
    iwildcam_train_65 = None
    myUAE = None

    iwildcam_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_70_ds.npz')
    iwildcam_train_70 = iwildcam_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_70.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_70,detector_name='iwildcam_UAE_70_CVM',save_dec=True)

    iwildcam_train_70_comp = None
    iwildcam_train_70 = None
    myUAE = None

    iwildcam_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_75_ds.npz')
    iwildcam_train_75 = iwildcam_train_75_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_75.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_75,detector_name='iwildcam_UAE_75_CVM',save_dec=True)

    iwildcam_train_75_comp = None
    iwildcam_train_75 = None
    myUAE = None

    iwildcam_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_80_ds.npz')
    iwildcam_train_80 = iwildcam_train_80_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_80.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_80,detector_name='iwildcam_UAE_80_CVM',save_dec=True)

    iwildcam_train_80_comp = None
    iwildcam_train_80 = None
    myUAE = None

    iwildcam_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_85_ds.npz')
    iwildcam_train_85 = iwildcam_train_85_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_85.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_85,detector_name='iwildcam_UAE_85_CVM',save_dec=True)

    iwildcam_train_85_comp = None
    iwildcam_train_85 = None
    myUAE = None

    iwildcam_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_90_ds.npz')
    iwildcam_train_90 = iwildcam_train_90_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_90.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_90,detector_name='iwildcam_UAE_90_CVM',save_dec=True)

    iwildcam_train_90_comp = None
    iwildcam_train_90 = None
    myUAE = None

    iwildcam_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_95_ds.npz')
    iwildcam_train_95 = iwildcam_train_95_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_95.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_95,detector_name='iwildcam_UAE_95_CVM',save_dec=True)

    iwildcam_train_95_comp = None
    iwildcam_train_95 = None
    myUAE = None

    iwildcam_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_ds.npz')
    iwildcam_train_100 = iwildcam_train_100_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_100.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=iwildcam_train_100,detector_name='iwildcam_UAE_100_CVM',save_dec=True)

    iwildcam_train_100_comp = None
    iwildcam_train_100 = None
    myUAE = None


    iwildcam_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_5_ds.npz')
    iwildcam_train_5 = iwildcam_train_5_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_5.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_5,detector_name='iwildcam_UAE_5_MMD',save_dec=True)

    iwildcam_train_5_comp = None
    iwildcam_train_5 = None
    myUAE = None


    iwildcam_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_10_ds.npz')
    iwildcam_train_10 = iwildcam_train_10_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_10.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_10,detector_name='iwildcam_UAE_10_MMD',save_dec=True)

    iwildcam_train_10_comp = None

    myUAE = None

    iwildcam_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_15_ds.npz')
    iwildcam_train_15 = iwildcam_train_15_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_10.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_15,detector_name='iwildcam_UAE_15_MMD',save_dec=True)

    iwildcam_train_15_comp = None
    iwildcam_train_15 = None
    myUAE = None

    iwildcam_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_20_ds.npz')
    iwildcam_train_20 = iwildcam_train_20_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_20.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_20,detector_name='iwildcam_UAE_20_MMD',save_dec=True)

    iwildcam_train_20_comp = None
    iwildcam_train_20 = None
    myUAE = None

    iwildcam_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_25_ds.npz')
    iwildcam_train_25 = iwildcam_train_25_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_25.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_25,detector_name='iwildcam_UAE_25_MMD',save_dec=True)

    iwildcam_train_25_comp = None
    iwildcam_train_25 = None
    myUAE = None

    iwildcam_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_30_ds.npz')
    iwildcam_train_30 = iwildcam_train_30_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_30.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_30,detector_name='iwildcam_UAE_30_MMD',save_dec=True)

    iwildcam_train_30_comp = None
    iwildcam_train_30 = None
    myUAE = None

    iwildcam_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_35_ds.npz')
    iwildcam_train_35 = iwildcam_train_35_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_35.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_35,detector_name='iwildcam_UAE_35_MMD',save_dec=True)

    iwildcam_train_35_comp = None
    iwildcam_train_35 = None
    myUAE = None

    iwildcam_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_40_ds.npz')
    iwildcam_train_40 = iwildcam_train_40_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_40.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_40,detector_name='iwildcam_UAE_40_MMD',save_dec=True)

    iwildcam_train_40_comp = None
    iwildcam_train_40 = None
    myUAE = None

    iwildcam_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_45_ds.npz')
    iwildcam_train_45 = iwildcam_train_45_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_45.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_45,detector_name='iwildcam_UAE_45_MMD',save_dec=True)

    iwildcam_train_45_comp = None
    iwildcam_train_45 = None
    myUAE = None

    iwildcam_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_50_ds.npz')
    iwildcam_train_50 = iwildcam_train_50_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_50.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_50,detector_name='iwildcam_UAE_50_MMD',save_dec=True)

    iwildcam_train_50_comp = None
    iwildcam_train_50 = None
    myUAE = None

    iwildcam_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_55_ds.npz')
    iwildcam_train_55 = iwildcam_train_55_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_55.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_55,detector_name='iwildcam_UAE_55_MMD',save_dec=True)

    iwildcam_train_55_comp = None
    iwildcam_train_55 = None
    myUAE = None


    iwildcam_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_60_ds.npz')
    iwildcam_train_60 = iwildcam_train_60_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_60.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_60,detector_name='iwildcam_UAE_60_MMD',save_dec=True)

    iwildcam_train_60_comp = None
    iwildcam_train_60 = None
    myUAE = None

    iwildcam_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_65_ds.npz')
    iwildcam_train_65 = iwildcam_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_65.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_65,detector_name='iwildcam_UAE_65_MMD',save_dec=True)

    iwildcam_train_65_comp = None
    iwildcam_train_65 = None
    myUAE = None

    iwildcam_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_70_ds.npz')
    iwildcam_train_70 = iwildcam_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_70.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_70,detector_name='iwildcam_UAE_70_MMD',save_dec=True)

    iwildcam_train_70_comp = None
    iwildcam_train_70 = None
    myUAE = None

    iwildcam_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_75_ds.npz')
    iwildcam_train_75 = iwildcam_train_75_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_75.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_75,detector_name='iwildcam_UAE_75_MMD',save_dec=True)

    iwildcam_train_75_comp = None
    iwildcam_train_75 = None
    myUAE = None

    iwildcam_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_80_ds.npz')
    iwildcam_train_80 = iwildcam_train_80_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_80.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_80,detector_name='iwildcam_UAE_80_MMD',save_dec=True)

    iwildcam_train_80_comp = None
    iwildcam_train_80 = None
    myUAE = None

    iwildcam_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_85_ds.npz')
    iwildcam_train_85 = iwildcam_train_85_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_85.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_85,detector_name='iwildcam_UAE_85_MMD',save_dec=True)

    iwildcam_train_85_comp = None
    iwildcam_train_85 = None
    myUAE = None

    iwildcam_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_90_ds.npz')
    iwildcam_train_90 = iwildcam_train_90_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_90.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_90,detector_name='iwildcam_UAE_90_MMD',save_dec=True)

    iwildcam_train_90_comp = None
    iwildcam_train_90 = None
    myUAE = None

    iwildcam_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_95_ds.npz')
    iwildcam_train_95 = iwildcam_train_95_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_95.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_95,detector_name='iwildcam_UAE_95_MMD',save_dec=True)

    iwildcam_train_95_comp = None
    iwildcam_train_95 = None
    myUAE = None

    iwildcam_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_ds.npz')
    iwildcam_train_100 = iwildcam_train_100_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_100.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=iwildcam_train_100,detector_name='iwildcam_UAE_100_MMD',save_dec=True)

    iwildcam_train_100_comp = None
    iwildcam_train_100 = None
    myUAE = None



    iwildcam_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_5_ds.npz')
    iwildcam_train_5 = iwildcam_train_5_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_5.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_5,detector_name='iwildcam_UAE_5_LSDD',save_dec=True)

    iwildcam_train_5_comp = None
    iwildcam_train_5 = None
    myUAE = None


    iwildcam_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_10_ds.npz')
    iwildcam_train_10 = iwildcam_train_10_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_10.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_10,detector_name='iwildcam_UAE_10_LSDD',save_dec=True)

    iwildcam_train_10_comp = None
    iwildcam_train_10 = None
    myUAE = None

    iwildcam_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_15_ds.npz')
    iwildcam_train_15 = iwildcam_train_15_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_15.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_15,detector_name='iwildcam_UAE_15_LSDD',save_dec=True)

    iwildcam_train_15_comp = None
    iwildcam_train_15 = None
    myUAE = None

    iwildcam_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_20_ds.npz')
    iwildcam_train_20 = iwildcam_train_20_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_20.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_20,detector_name='iwildcam_UAE_20_LSDD',save_dec=True)

    iwildcam_train_20_comp = None
    iwildcam_train_20 = None
    myUAE = None

    iwildcam_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_25_ds.npz')
    iwildcam_train_25 = iwildcam_train_25_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_25.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_25,detector_name='iwildcam_UAE_25_LSDD',save_dec=True)

    iwildcam_train_25_comp = None
    iwildcam_train_25 = None
    myUAE = None

    iwildcam_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_30_ds.npz')
    iwildcam_train_30 = iwildcam_train_30_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_30.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_30,detector_name='iwildcam_UAE_30_LSDD',save_dec=True)

    iwildcam_train_30_comp = None
    iwildcam_train_30 = None
    myUAE = None

    iwildcam_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_35_ds.npz')
    iwildcam_train_35 = iwildcam_train_35_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_35.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_35,detector_name='iwildcam_UAE_35_LSDD',save_dec=True)

    iwildcam_train_35_comp = None
    iwildcam_train_35 = None
    myUAE = None

    iwildcam_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_40_ds.npz')
    iwildcam_train_40 = iwildcam_train_40_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_40.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_40,detector_name='iwildcam_UAE_40_LSDD',save_dec=True)

    iwildcam_train_40_comp = None
    iwildcam_train_40 = None
    myUAE = None

    iwildcam_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_45_ds.npz')
    iwildcam_train_45 = iwildcam_train_45_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_45.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_45,detector_name='iwildcam_UAE_45_LSDD',save_dec=True)

    iwildcam_train_45_comp = None
    iwildcam_train_45 = None
    myUAE = None

    iwildcam_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_50_ds.npz')
    iwildcam_train_50 = iwildcam_train_50_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_50.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_50,detector_name='iwildcam_UAE_50_LSDD',save_dec=True)

    iwildcam_train_50_comp = None
    iwildcam_train_50 = None
    myUAE = None

    iwildcam_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_55_ds.npz')
    iwildcam_train_55 = iwildcam_train_55_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_55.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_55,detector_name='iwildcam_UAE_55_LSDD',save_dec=True)

    iwildcam_train_55_comp = None
    iwildcam_train_55 = None
    myUAE = None


    iwildcam_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_60_ds.npz')
    iwildcam_train_60 = iwildcam_train_60_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_60.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_60,detector_name='iwildcam_UAE_60_LSDD',save_dec=True)

    iwildcam_train_60_comp = None
    iwildcam_train_60 = None
    myUAE = None

    iwildcam_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_65_ds.npz')
    iwildcam_train_65 = iwildcam_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_65.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_65,detector_name='iwildcam_UAE_65_LSDD',save_dec=True)

    iwildcam_train_65_comp = None
    iwildcam_train_65 = None
    myUAE = None

    iwildcam_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_70_ds.npz')
    iwildcam_train_70 = iwildcam_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_70.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_70,detector_name='iwildcam_UAE_70_LSDD',save_dec=True)

    iwildcam_train_70_comp = None
    iwildcam_train_70 = None
    myUAE = None

    iwildcam_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_75_ds.npz')
    iwildcam_train_75 = iwildcam_train_75_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_75.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_75,detector_name='iwildcam_UAE_75_LSDD',save_dec=True)

    iwildcam_train_75_comp = None
    iwildcam_train_75 = None
    myUAE = None

    iwildcam_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_80_ds.npz')
    iwildcam_train_80 = iwildcam_train_80_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_80.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_80,detector_name='iwildcam_UAE_80_LSDD',save_dec=True)

    iwildcam_train_80_comp = None
    iwildcam_train_80 = None
    myUAE = None

    iwildcam_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_85_ds.npz')
    iwildcam_train_85 = iwildcam_train_85_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_85.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_85,detector_name='iwildcam_UAE_85_LSDD',save_dec=True)

    iwildcam_train_85_comp = None
    iwildcam_train_85 = None
    myUAE = None

    iwildcam_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_90_ds.npz')
    iwildcam_train_90 = iwildcam_train_90_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_90.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_90,detector_name='iwildcam_UAE_90_LSDD',save_dec=True)

    iwildcam_train_90_comp = None
    iwildcam_train_90 = None
    myUAE = None

    iwildcam_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_95_ds.npz')
    iwildcam_train_95 = iwildcam_train_95_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_95.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_95,detector_name='iwildcam_UAE_95_LSDD',save_dec=True)

    iwildcam_train_95_comp = None
    iwildcam_train_95 = None
    myUAE = None

    iwildcam_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/iwildcam_v2.0/iwildcam_train_ds.npz')
    iwildcam_train_100 = iwildcam_train_100_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=iwildcam_train_100.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=iwildcam_train_100,detector_name='iwildcam_UAE_100_MMD_LSDD',save_dec=True)

    iwildcam_train_100_comp = None
    iwildcam_train_100 = None
    myUAE = None


    with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
        drift_detection_config = json.load(config_file)



    camelyon_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_5_ds.npz')
    camelyon_train_5 = camelyon_train_5_comp['arr_0']
    camelyon_train_5_0_50 = camelyon_train_5[:int(len(camelyon_train_5)*0.5)]
    camelyon_train_5_50_100 = camelyon_train_5[int(len(camelyon_train_5)*0.5):]

    camelyon_train_5_0_50 = torch.as_tensor(camelyon_train_5_0_50)
    camelyon_train_5_0_50_dl = DataLoader(TensorDataset(camelyon_train_5_0_50,camelyon_train_5_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

    myTAE = TrainedAutoencoder(drift_detection_config)
    myTAE.init_default_pt_autoencoder(camelyon_train_5_0_50_dl)
    myTAE.init_detector(detector_type='KS',reference_data=camelyon_train_5_50_100,detector_name='camelyon_TAE_5_KS',save_dec=True)

    camelyon_train_5_comp= None
    camelyon_train_5 = None
    camelyon_train_5_0_50 = None 
    camelyon_train_5_50_100 = None
    camelyon_train_5_0_50_dl = None
    myTAE = None 


# ======================================================================================
# call
if __name__ == "__main__":
    main()