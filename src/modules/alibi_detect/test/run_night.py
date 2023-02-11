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


    rxrx1_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_5_ds.npz')
    rxrx1_train_5 = rxrx1_train_5_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_5.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_5,detector_name='rxrx1_UAE_5_LSDD',save_dec=True)

    rxrx1_train_5_comp = None
    rxrx1_train_5 = None
    myUAE = None


    rxrx1_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_10_ds.npz')
    rxrx1_train_10 = rxrx1_train_10_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_10.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_10,detector_name='rxrx1_UAE_10_LSDD',save_dec=True)

    rxrx1_train_10_comp = None
    rxrx1_train_10 = None
    myUAE = None

    rxrx1_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_15_ds.npz')
    rxrx1_train_15 = rxrx1_train_15_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_15.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_15,detector_name='rxrx1_UAE_15_LSDD',save_dec=True)

    rxrx1_train_15_comp = None
    rxrx1_train_15 = None
    myUAE = None

    rxrx1_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_20_ds.npz')
    rxrx1_train_20 = rxrx1_train_20_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_20.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_20,detector_name='rxrx1_UAE_20_LSDD',save_dec=True)

    rxrx1_train_20_comp = None
    rxrx1_train_20 = None
    myUAE = None

    rxrx1_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_25_ds.npz')
    rxrx1_train_25 = rxrx1_train_25_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_25.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_25,detector_name='rxrx1_UAE_25_LSDD',save_dec=True)

    rxrx1_train_25_comp = None
    rxrx1_train_25 = None
    myUAE = None

    rxrx1_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_30_ds.npz')
    rxrx1_train_30 = rxrx1_train_30_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_30.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_30,detector_name='rxrx1_UAE_30_LSDD',save_dec=True)

    rxrx1_train_30_comp = None
    rxrx1_train_30 = None
    myUAE = None

    rxrx1_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_35_ds.npz')
    rxrx1_train_35 = rxrx1_train_35_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_35.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_35,detector_name='rxrx1_UAE_35_LSDD',save_dec=True)

    rxrx1_train_35_comp = None
    rxrx1_train_35 = None
    myUAE = None

    rxrx1_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_40_ds.npz')
    rxrx1_train_40 = rxrx1_train_40_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_40.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_40,detector_name='rxrx1_UAE_40_LSDD',save_dec=True)

    rxrx1_train_40_comp = None
    rxrx1_train_40 = None
    myUAE = None

    rxrx1_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_45_ds.npz')
    rxrx1_train_45 = rxrx1_train_45_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_45.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_45,detector_name='rxrx1_UAE_45_LSDD',save_dec=True)

    rxrx1_train_45_comp = None
    rxrx1_train_45 = None
    myUAE = None

    rxrx1_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_50_ds.npz')
    rxrx1_train_50 = rxrx1_train_50_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_50.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_50,detector_name='rxrx1_UAE_50_LSDD',save_dec=True)

    rxrx1_train_50_comp = None
    rxrx1_train_50 = None
    myUAE = None

    rxrx1_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_55_ds.npz')
    rxrx1_train_55 = rxrx1_train_55_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_55.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_55,detector_name='rxrx1_UAE_55_LSDD',save_dec=True)

    rxrx1_train_55_comp = None
    rxrx1_train_55 = None
    myUAE = None


    rxrx1_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_60_ds.npz')
    rxrx1_train_60 = rxrx1_train_60_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_60.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_60,detector_name='rxrx1_UAE_60_LSDD',save_dec=True)

    rxrx1_train_60_comp = None
    rxrx1_train_60 = None
    myUAE = None

    rxrx1_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_65_ds.npz')
    rxrx1_train_65 = rxrx1_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_65.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_65,detector_name='rxrx1_UAE_65_LSDD',save_dec=True)

    rxrx1_train_65_comp = None
    rxrx1_train_65 = None
    myUAE = None

    rxrx1_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_70_ds.npz')
    rxrx1_train_70 = rxrx1_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_70.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_70,detector_name='rxrx1_UAE_70_LSDD',save_dec=True)

    rxrx1_train_70_comp = None
    rxrx1_train_70 = None
    myUAE = None

    rxrx1_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_75_ds.npz')
    rxrx1_train_75 = rxrx1_train_75_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_75.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_75,detector_name='rxrx1_UAE_75_LSDD',save_dec=True)

    rxrx1_train_75_comp = None
    rxrx1_train_75 = None
    myUAE = None

    rxrx1_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_80_ds.npz')
    rxrx1_train_80 = rxrx1_train_80_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_80.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_80,detector_name='rxrx1_UAE_80_LSDD',save_dec=True)

    rxrx1_train_80_comp = None
    rxrx1_train_80 = None
    myUAE = None

    rxrx1_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_85_ds.npz')
    rxrx1_train_85 = rxrx1_train_85_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_85.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_85,detector_name='rxrx1_UAE_85_LSDD',save_dec=True)

    rxrx1_train_85_comp = None
    rxrx1_train_85 = None
    myUAE = None

    rxrx1_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_90_ds.npz')
    rxrx1_train_90 = rxrx1_train_90_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_90.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_90,detector_name='rxrx1_UAE_90_LSDD',save_dec=True)

    rxrx1_train_90_comp = None
    rxrx1_train_90 = None
    myUAE = None

    rxrx1_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_95_ds.npz')
    rxrx1_train_95 = rxrx1_train_95_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_95.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_95,detector_name='rxrx1_UAE_95_LSDD',save_dec=True)

    rxrx1_train_95_comp = None
    rxrx1_train_95 = None
    myUAE = None

    rxrx1_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/rxrx1_v1.0/rxrx1_train_ds.npz')
    rxrx1_train_100 = rxrx1_train_100_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=100,input_shape=rxrx1_train_100.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=rxrx1_train_100,detector_name='rxrx1_UAE_100_MMD_LSDD',save_dec=True)

    rxrx1_train_100_comp = None
    rxrx1_train_100 = None
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

    camelyon_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_10_ds.npz')
    camelyon_train_10 = camelyon_train_10_comp['arr_0']
    camelyon_train_10_0_50 = camelyon_train_10[:int(len(camelyon_train_10)*0.5)]
    camelyon_train_10_50_100 = camelyon_train_10[int(len(camelyon_train_10)*0.5):]

    camelyon_train_10_0_50 = torch.as_tensor(camelyon_train_10_0_50)
    camelyon_train_10_0_50_dl = DataLoader(TensorDataset(camelyon_train_10_0_50,camelyon_train_10_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

    myTAE = TrainedAutoencoder(drift_detection_config)
    myTAE.init_default_pt_autoencoder(camelyon_train_10_0_50_dl)
    myTAE.init_detector(detector_type='KS',reference_data=camelyon_train_10_50_100,detector_name='camelyon_TAE_10_KS',save_dec=True)

    camelyon_train_10_comp= None
    camelyon_train_10 = None
    camelyon_train_10_0_50 = None 
    camelyon_train_10_50_100 = None
    camelyon_train_10_0_50_dl = None
    myTAE = None 

    camelyon_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_15_ds.npz')
    camelyon_train_15 = camelyon_train_15_comp['arr_0']
    camelyon_train_15_0_50 = camelyon_train_15[:int(len(camelyon_train_15)*0.5)]
    camelyon_train_15_50_100 = camelyon_train_15[int(len(camelyon_train_15)*0.5):]

    camelyon_train_15_0_50 = torch.as_tensor(camelyon_train_15_0_50)
    camelyon_train_15_0_50_dl = DataLoader(TensorDataset(camelyon_train_15_0_50,camelyon_train_15_0_50),drift_detection_config['TAE']['BATCH_SIZE'],shuffle=True)

    myTAE = TrainedAutoencoder(drift_detection_config)
    myTAE.init_default_pt_autoencoder(camelyon_train_15_0_50_dl)
    myTAE.init_detector(detector_type='KS',reference_data=camelyon_train_15_50_100,detector_name='camelyon_TAE_15_KS',save_dec=True)

    camelyon_train_15_comp= None
    camelyon_train_15 = None
    camelyon_train_15_0_50 = None 
    camelyon_train_15_50_100 = None
    camelyon_train_15_0_50_dl = None
    myTAE = None


# ======================================================================================
# call
if __name__ == "__main__":
    main()