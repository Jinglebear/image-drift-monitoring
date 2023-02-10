import numpy as np    
import json 
import os
import sys
def main():
    # %cd  '/home/ubuntu/image-drift-monitoring'
    # %pwd
    sys.path.append('/home/ubuntu/image-drift-monitoring/src')
    from modules.alibi_detect.untrained_encoder import UntrainedAutoencoder



    with open('/home/ubuntu/image-drift-monitoring/config/common/drift_detection_config.json') as config_file:
        drift_detection_config = json.load(config_file)


    """ KS """

    # camelyon_train_5_comp = np.load('camelyon_train_5_ds.npz')
    # camelyon_train_5 = camelyon_train_5_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_5,backend='pytorch',detector_name='Camelyon_UAE_5_MMD_torch',save_dec=True)

    # camelyon_train_5_comp = None
    # camelyon_train_5 = None
    # myUAE = None


    # camelyon_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_10_ds.npz')
    # camelyon_train_10 = camelyon_train_10_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)

    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_10,detector_name='Camelyon_UAE_10_KS_torch',save_dec=True)

    # camelyon_train_10_comp = None

    # myUAE = None

    # camelyon_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_15_ds.npz')
    # camelyon_train_15 = camelyon_train_15_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_15,detector_name='Camelyon_UAE_15_KS_torch',save_dec=True)

    # camelyon_train_15_comp = None
    # camelyon_train_15 = None
    # myUAE = None

    # camelyon_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_20_ds.npz')
    # camelyon_train_20 = camelyon_train_20_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_20,detector_name='Camelyon_UAE_20_KS_torch',save_dec=True)

    # camelyon_train_20_comp = None
    # camelyon_train_20 = None
    # myUAE = None

    # camelyon_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_25_ds.npz')
    # camelyon_train_25 = camelyon_train_25_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_25,detector_name='Camelyon_UAE_25_KS_torch',save_dec=True)

    # camelyon_train_25_comp = None
    # camelyon_train_25 = None
    # myUAE = None

    # camelyon_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_30_ds.npz')
    # camelyon_train_30 = camelyon_train_30_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_30,detector_name='Camelyon_UAE_30_KS_torch',save_dec=True)

    # camelyon_train_30_comp = None
    # camelyon_train_30 = None
    # myUAE = None

    # camelyon_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_35_ds.npz')
    # camelyon_train_35 = camelyon_train_35_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_35,detector_name='Camelyon_UAE_35_KS_torch',save_dec=True)

    # camelyon_train_35_comp = None
    # camelyon_train_35 = None
    # myUAE = None

    # camelyon_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_40_ds.npz')
    # camelyon_train_40 = camelyon_train_40_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_40,detector_name='Camelyon_UAE_40_KS_torch',save_dec=True)

    # camelyon_train_40_comp = None
    # camelyon_train_40 = None
    # myUAE = None

    # camelyon_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_45_ds.npz')
    # camelyon_train_45 = camelyon_train_45_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_45,detector_name='Camelyon_UAE_45_KS_torch',save_dec=True)

    # camelyon_train_45_comp = None
    # camelyon_train_45 = None
    # myUAE = None

    # camelyon_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_50_ds.npz')
    # camelyon_train_50 = camelyon_train_50_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_50,detector_name='Camelyon_UAE_50_KS_torch',save_dec=True)

    # camelyon_train_50_comp = None
    # camelyon_train_50 = None
    # myUAE = None

    # camelyon_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_55_ds.npz')
    # camelyon_train_55 = camelyon_train_55_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_55,detector_name='Camelyon_UAE_55_KS_torch',save_dec=True)

    # camelyon_train_55_comp = None
    # camelyon_train_55 = None
    # myUAE = None


    # camelyon_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_60_ds.npz')
    # camelyon_train_60 = camelyon_train_60_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_60.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_60,detector_name='Camelyon_UAE_60_KS',save_dec=True)

    # camelyon_train_60_comp = None
    # camelyon_train_60 = None
    # myUAE = None

    camelyon_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_65_ds.npz')
    camelyon_train_65 = camelyon_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_65.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_65,detector_name='Camelyon_UAE_65_KS',save_dec=True)

    camelyon_train_65_comp = None
    camelyon_train_65 = None
    myUAE = None

    camelyon_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_70_ds.npz')
    camelyon_train_70 = camelyon_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_70.shape[1:])
    myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_70,detector_name='Camelyon_UAE_70_KS',save_dec=True)

    camelyon_train_70_comp = None
    camelyon_train_70 = None
    myUAE = None

    # camelyon_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_75_ds.npz')
    # camelyon_train_75 = camelyon_train_75_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_75.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_75,detector_name='Camelyon_UAE_75_KS',save_dec=True)

    # camelyon_train_75_comp = None
    # camelyon_train_75 = None
    # myUAE = None

    # camelyon_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_80_ds.npz')
    # camelyon_train_80 = camelyon_train_80_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_80.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_80,detector_name='Camelyon_UAE_80_KS',save_dec=True)

    # camelyon_train_80_comp = None
    # camelyon_train_80 = None
    # myUAE = None

    # camelyon_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_85_ds.npz')
    # camelyon_train_85 = camelyon_train_85_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_85.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_85,detector_name='Camelyon_UAE_85_KS',save_dec=True)

    # camelyon_train_85_comp = None
    # camelyon_train_85 = None
    # myUAE = None

    # camelyon_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_90_ds.npz')
    # camelyon_train_90 = camelyon_train_90_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_90.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_90,detector_name='Camelyon_UAE_90_KS',save_dec=True)

    # camelyon_train_90_comp = None
    # camelyon_train_90 = None
    # myUAE = None

    # camelyon_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_95_ds.npz')
    # camelyon_train_95 = camelyon_train_95_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_95.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_95,detector_name='Camelyon_UAE_95_KS',save_dec=True)

    # camelyon_train_95_comp = None
    # camelyon_train_95 = None
    # myUAE = None

    # camelyon_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
    # camelyon_train_100 = camelyon_train_100_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_100.shape[1:])
    # myUAE.init_detector(detector_type='KS',reference_data=camelyon_train_100,detector_name='Camelyon_UAE_100_KS',save_dec=True)

    # camelyon_train_100_comp = None
    # camelyon_train_100 = None
    # myUAE = None

    """ CVM """
    # camelyon_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_5_ds.npz')
    # camelyon_train_5 = camelyon_train_5_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_5.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_5,detector_name='Camelyon_UAE_5_CVM',save_dec=True)

    # camelyon_train_5_comp = None
    # camelyon_train_5 = None
    # myUAE = None


    # camelyon_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_10_ds.npz')
    # camelyon_train_10 = camelyon_train_10_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_10,detector_name='Camelyon_UAE_10_CVM',save_dec=True)

    # camelyon_train_10_comp = None

    # myUAE = None

    # camelyon_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_15_ds.npz')
    # camelyon_train_15 = camelyon_train_15_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_15,detector_name='Camelyon_UAE_15_CVM',save_dec=True)

    # camelyon_train_15_comp = None
    # camelyon_train_15 = None
    # myUAE = None

    # camelyon_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_20_ds.npz')
    # camelyon_train_20 = camelyon_train_20_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_20,detector_name='Camelyon_UAE_20_CVM',save_dec=True)

    # camelyon_train_20_comp = None
    # camelyon_train_20 = None
    # myUAE = None

    # camelyon_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_25_ds.npz')
    # camelyon_train_25 = camelyon_train_25_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_25,detector_name='Camelyon_UAE_25_CVM',save_dec=True)

    # camelyon_train_25_comp = None
    # camelyon_train_25 = None
    # myUAE = None

    # camelyon_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_30_ds.npz')
    # camelyon_train_30 = camelyon_train_30_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_30,detector_name='Camelyon_UAE_30_CVM',save_dec=True)

    # camelyon_train_30_comp = None
    # camelyon_train_30 = None
    # myUAE = None

    # camelyon_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_35_ds.npz')
    # camelyon_train_35 = camelyon_train_35_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_35,detector_name='Camelyon_UAE_35_CVM',save_dec=True)

    # camelyon_train_35_comp = None
    # camelyon_train_35 = None
    # myUAE = None

    # camelyon_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_40_ds.npz')
    # camelyon_train_40 = camelyon_train_40_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_40,detector_name='Camelyon_UAE_40_CVM',save_dec=True)

    # camelyon_train_40_comp = None
    # camelyon_train_40 = None
    # myUAE = None

    # camelyon_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_45_ds.npz')
    # camelyon_train_45 = camelyon_train_45_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_45,detector_name='Camelyon_UAE_45_CVM',save_dec=True)

    # camelyon_train_45_comp = None
    # camelyon_train_45 = None
    # myUAE = None

    # camelyon_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_50_ds.npz')
    # camelyon_train_50 = camelyon_train_50_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_50,detector_name='Camelyon_UAE_50_CVM',save_dec=True)

    # camelyon_train_50_comp = None
    # camelyon_train_50 = None
    # myUAE = None

    # camelyon_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_55_ds.npz')
    # camelyon_train_55 = camelyon_train_55_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_55,detector_name='Camelyon_UAE_55_CVM',save_dec=True)

    # camelyon_train_55_comp = None
    # camelyon_train_55 = None
    # myUAE = None


    camelyon_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_60_ds.npz')
    camelyon_train_60 = camelyon_train_60_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_60.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_60,detector_name='Camelyon_UAE_60_CVM',save_dec=True)

    camelyon_train_60_comp = None
    camelyon_train_60 = None
    myUAE = None

    camelyon_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_65_ds.npz')
    camelyon_train_65 = camelyon_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_65.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_65,detector_name='Camelyon_UAE_65_CVM',save_dec=True)

    camelyon_train_65_comp = None
    camelyon_train_65 = None
    myUAE = None

    camelyon_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_70_ds.npz')
    camelyon_train_70 = camelyon_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_70.shape[1:])
    myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_70,detector_name='Camelyon_UAE_70_CVM',save_dec=True)

    camelyon_train_70_comp = None
    camelyon_train_70 = None
    myUAE = None

    # camelyon_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_75_ds.npz')
    # camelyon_train_75 = camelyon_train_75_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_75.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_75,detector_name='Camelyon_UAE_75_CVM',save_dec=True)

    # camelyon_train_75_comp = None
    # camelyon_train_75 = None
    # myUAE = None

    # camelyon_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_80_ds.npz')
    # camelyon_train_80 = camelyon_train_80_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_80.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_80,detector_name='Camelyon_UAE_80_CVM',save_dec=True)

    # camelyon_train_80_comp = None
    # camelyon_train_80 = None
    # myUAE = None

    # camelyon_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_85_ds.npz')
    # camelyon_train_85 = camelyon_train_85_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_85.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_85,detector_name='Camelyon_UAE_85_CVM',save_dec=True)

    # camelyon_train_85_comp = None
    # camelyon_train_85 = None
    # myUAE = None

    # camelyon_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_90_ds.npz')
    # camelyon_train_90 = camelyon_train_90_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_90.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_90,detector_name='Camelyon_UAE_90_CVM',save_dec=True)

    # camelyon_train_90_comp = None
    # camelyon_train_90 = None
    # myUAE = None

    # camelyon_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_95_ds.npz')
    # camelyon_train_95 = camelyon_train_95_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_95.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_95,detector_name='Camelyon_UAE_95_CVM',save_dec=True)

    # camelyon_train_95_comp = None
    # camelyon_train_95 = None
    # myUAE = None

    # camelyon_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
    # camelyon_train_100 = camelyon_train_100_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_100.shape[1:])
    # myUAE.init_detector(detector_type='CVM',reference_data=camelyon_train_100,detector_name='Camelyon_UAE_100_CVM',save_dec=True)

    # camelyon_train_100_comp = None
    # camelyon_train_100 = None
    # myUAE = None
    """ MMD """
    # camelyon_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_5_ds.npz')
    # camelyon_train_5 = camelyon_train_5_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_5.shape[1:])
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_5,detector_name='Camelyon_UAE_5_MMD',save_dec=True)

    # camelyon_train_5_comp = None
    # camelyon_train_5 = None
    # myUAE = None


    # camelyon_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_10_ds.npz')
    # camelyon_train_10 = camelyon_train_10_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_10,detector_name='Camelyon_UAE_10_MMD',save_dec=True)

    # camelyon_train_10_comp = None

    # myUAE = None

    # camelyon_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_15_ds.npz')
    # camelyon_train_15 = camelyon_train_15_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_15,detector_name='Camelyon_UAE_15_MMD',save_dec=True)

    # camelyon_train_15_comp = None
    # camelyon_train_15 = None
    # myUAE = None

    camelyon_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_20_ds.npz')
    camelyon_train_20 = camelyon_train_20_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_20.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_20,detector_name='Camelyon_UAE_20_MMD',save_dec=True)

    camelyon_train_20_comp = None
    camelyon_train_20 = None
    myUAE = None

    camelyon_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_25_ds.npz')
    camelyon_train_25 = camelyon_train_25_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_25.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_25,detector_name='Camelyon_UAE_25_MMD',save_dec=True)

    camelyon_train_25_comp = None
    camelyon_train_25 = None
    myUAE = None

    camelyon_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_30_ds.npz')
    camelyon_train_30 = camelyon_train_30_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_30.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_30,detector_name='Camelyon_UAE_30_MMD',save_dec=True)

    camelyon_train_30_comp = None
    camelyon_train_30 = None
    myUAE = None

    camelyon_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_35_ds.npz')
    camelyon_train_35 = camelyon_train_35_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_35.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_35,detector_name='Camelyon_UAE_35_MMD',save_dec=True)

    camelyon_train_35_comp = None
    camelyon_train_35 = None
    myUAE = None

    camelyon_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_40_ds.npz')
    camelyon_train_40 = camelyon_train_40_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_40.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_40,detector_name='Camelyon_UAE_40_MMD',save_dec=True)

    camelyon_train_40_comp = None
    camelyon_train_40 = None
    myUAE = None

    camelyon_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_45_ds.npz')
    camelyon_train_45 = camelyon_train_45_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_45.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_45,detector_name='Camelyon_UAE_45_MMD',save_dec=True)

    camelyon_train_45_comp = None
    camelyon_train_45 = None
    myUAE = None

    camelyon_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_50_ds.npz')
    camelyon_train_50 = camelyon_train_50_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_50.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_50,detector_name='Camelyon_UAE_50_MMD',save_dec=True)

    camelyon_train_50_comp = None
    camelyon_train_50 = None
    myUAE = None

    camelyon_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_55_ds.npz')
    camelyon_train_55 = camelyon_train_55_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_55.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_55,detector_name='Camelyon_UAE_55_MMD',save_dec=True)

    camelyon_train_55_comp = None
    camelyon_train_55 = None
    myUAE = None


    camelyon_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_60_ds.npz')
    camelyon_train_60 = camelyon_train_60_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_60.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_60,backend='pytorch',detector_name='Camelyon_UAE_60_MMD',save_dec=True)

    camelyon_train_60_comp = None
    camelyon_train_60 = None
    myUAE = None

    camelyon_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_65_ds.npz')
    camelyon_train_65 = camelyon_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_65.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_65,backend='pytorch',detector_name='Camelyon_UAE_65_MMD',save_dec=True)

    camelyon_train_65_comp = None
    camelyon_train_65 = None
    myUAE = None

    camelyon_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_70_ds.npz')
    camelyon_train_70 = camelyon_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_70.shape[1:])
    myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_70,backend='pytorch',detector_name='Camelyon_UAE_70_MMD',save_dec=True)

    camelyon_train_70_comp = None
    camelyon_train_70 = None
    myUAE = None

    # camelyon_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_75_ds.npz')
    # camelyon_train_75 = camelyon_train_75_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_75,backend='pytorch',detector_name='Camelyon_UAE_75_MMD_torch',save_dec=True)

    # camelyon_train_75_comp = None
    # camelyon_train_75 = None
    # myUAE = None

    # camelyon_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_80_ds.npz')
    # camelyon_train_80 = camelyon_train_80_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_80,backend='pytorch',detector_name='Camelyon_UAE_80_MMD_torch',save_dec=True)

    # camelyon_train_80_comp = None
    # camelyon_train_80 = None
    # myUAE = None

    # camelyon_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_85_ds.npz')
    # camelyon_train_85 = camelyon_train_85_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_85,backend='pytorch',detector_name='Camelyon_UAE_85_MMD_torch',save_dec=True)

    # camelyon_train_85_comp = None
    # camelyon_train_85 = None
    # myUAE = None

    # camelyon_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_90_ds.npz')
    # camelyon_train_90 = camelyon_train_90_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_90,backend='pytorch',detector_name='Camelyon_UAE_90_MMD_torch',save_dec=True)

    # camelyon_train_90_comp = None
    # camelyon_train_90 = None
    # myUAE = None

    # camelyon_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_95_ds.npz')
    # camelyon_train_95 = camelyon_train_95_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_95,backend='pytorch',detector_name='Camelyon_UAE_95_MMD_torch',save_dec=True)

    # camelyon_train_95_comp = None
    # camelyon_train_95 = None
    # myUAE = None

    # camelyon_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
    # camelyon_train_100 = camelyon_train_100_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='MMD',reference_data=camelyon_train_100,backend='pytorch',detector_name='Camelyon_UAE_100_MMD_torch',save_dec=True)

    # camelyon_train_100_comp = None
    # camelyon_train_100 = None
    # myUAE = None

    """ LSDD """
    camelyon_train_5_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_5_ds.npz')
    camelyon_train_5 = camelyon_train_5_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_5.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_5,detector_name='Camelyon_UAE_5_LSDD',save_dec=True)

    camelyon_train_5_comp = None
    camelyon_train_5 = None
    myUAE = None


    camelyon_train_10_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_10_ds.npz')
    camelyon_train_10 = camelyon_train_10_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_10.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_10,detector_name='Camelyon_UAE_10_LSDD',save_dec=True)

    camelyon_train_10_comp = None
    camelyon_train_10 = None
    myUAE = None

    camelyon_train_15_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_15_ds.npz')
    camelyon_train_15 = camelyon_train_15_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_15.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_15,detector_name='Camelyon_UAE_15_LSDD',save_dec=True)

    camelyon_train_15_comp = None
    camelyon_train_15 = None
    myUAE = None

    camelyon_train_20_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_20_ds.npz')
    camelyon_train_20 = camelyon_train_20_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_20.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_20,detector_name='Camelyon_UAE_20_LSDD',save_dec=True)

    camelyon_train_20_comp = None
    camelyon_train_20 = None
    myUAE = None

    camelyon_train_25_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_25_ds.npz')
    camelyon_train_25 = camelyon_train_25_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_25.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_25,detector_name='Camelyon_UAE_25_LSDD',save_dec=True)

    camelyon_train_25_comp = None
    camelyon_train_25 = None
    myUAE = None

    camelyon_train_30_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_30_ds.npz')
    camelyon_train_30 = camelyon_train_30_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_30.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_30,detector_name='Camelyon_UAE_30_LSDD',save_dec=True)

    camelyon_train_30_comp = None
    camelyon_train_30 = None
    myUAE = None

    camelyon_train_35_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_35_ds.npz')
    camelyon_train_35 = camelyon_train_35_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_35.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_35,detector_name='Camelyon_UAE_35_LSDD',save_dec=True)

    camelyon_train_35_comp = None
    camelyon_train_35 = None
    myUAE = None

    camelyon_train_40_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_40_ds.npz')
    camelyon_train_40 = camelyon_train_40_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_40.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_40,detector_name='Camelyon_UAE_40_LSDD',save_dec=True)

    camelyon_train_40_comp = None
    camelyon_train_40 = None
    myUAE = None

    camelyon_train_45_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_45_ds.npz')
    camelyon_train_45 = camelyon_train_45_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_45.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_45,detector_name='Camelyon_UAE_45_LSDD',save_dec=True)

    camelyon_train_45_comp = None
    camelyon_train_45 = None
    myUAE = None

    camelyon_train_50_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_50_ds.npz')
    camelyon_train_50 = camelyon_train_50_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_50.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_50,detector_name='Camelyon_UAE_50_LSDD',save_dec=True)

    camelyon_train_50_comp = None
    camelyon_train_50 = None
    myUAE = None

    camelyon_train_55_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_55_ds.npz')
    camelyon_train_55 = camelyon_train_55_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_55.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_55,detector_name='Camelyon_UAE_55_LSDD',save_dec=True)

    camelyon_train_55_comp = None
    camelyon_train_55 = None
    myUAE = None


    camelyon_train_60_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_60_ds.npz')
    camelyon_train_60 = camelyon_train_60_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_60.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_60,detector_name='Camelyon_UAE_60_LSDD',save_dec=True)

    camelyon_train_60_comp = None
    camelyon_train_60 = None
    myUAE = None

    camelyon_train_65_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_65_ds.npz')
    camelyon_train_65 = camelyon_train_65_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_65.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_65,detector_name='Camelyon_UAE_65_LSDD',save_dec=True)

    camelyon_train_65_comp = None
    camelyon_train_65 = None
    myUAE = None

    camelyon_train_70_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_70_ds.npz')
    camelyon_train_70 = camelyon_train_70_comp['arr_0']

    myUAE = UntrainedAutoencoder(drift_detection_config)
    myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_70.shape[1:])
    myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_70,detector_name='Camelyon_UAE_70_LSDD',save_dec=True)

    camelyon_train_70_comp = None
    camelyon_train_70 = None
    myUAE = None

    # camelyon_train_75_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_75_ds.npz')
    # camelyon_train_75 = camelyon_train_75_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_tf_encoder(encoding_dim=96,input_shape=camelyon_train_75.shape[1:])
    # myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_75,detector_name='Camelyon_UAE_75_LSDD',save_dec=True)

    # camelyon_train_75_comp = None
    # camelyon_train_75 = None
    # myUAE = None

    # camelyon_train_80_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_80_ds.npz')
    # camelyon_train_80 = camelyon_train_80_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_80,detector_name='Camelyon_UAE_80_LSDD',save_dec=True)

    # camelyon_train_80_comp = None
    # camelyon_train_80 = None
    # myUAE = None

    # camelyon_train_85_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_85_ds.npz')
    # camelyon_train_85 = camelyon_train_85_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_85,detector_name='Camelyon_UAE_85_LSDD',save_dec=True)

    # camelyon_train_85_comp = None
    # camelyon_train_85 = None
    # myUAE = None

    # camelyon_train_90_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_90_ds.npz')
    # camelyon_train_90 = camelyon_train_90_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_90,detector_name='Camelyon_UAE_90_LSDD',save_dec=True)

    # camelyon_train_90_comp = None
    # camelyon_train_90 = None
    # myUAE = None

    # camelyon_train_95_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_95_ds.npz')
    # camelyon_train_95 = camelyon_train_95_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_95,detector_name='Camelyon_UAE_95_LSDD',save_dec=True)

    # camelyon_train_95_comp = None
    # camelyon_train_95 = None
    # myUAE = None

    # camelyon_train_100_comp = np.load('/home/ubuntu/image-drift-monitoring/data/camelyon17_v1.0/camelyon_train_ds.npz')
    # camelyon_train_100 = camelyon_train_100_comp['arr_0']

    # myUAE = UntrainedAutoencoder(drift_detection_config)
    # myUAE.init_default_py_encoder()
    # myUAE.init_detector(detector_type='LSDD',reference_data=camelyon_train_100,detector_name='Camelyon_UAE_100_MMD_LSDD',save_dec=True)

    # camelyon_train_100_comp = None
    # camelyon_train_100 = None
    # myUAE = None
# ======================================================================================
# call
if __name__ == "__main__":
    main()