# MISC imports
import numpy as np
import logging
from typing import Dict, Tuple, Union
from functools import partial
# tensorflow imports
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, InputLayer
# pytorch imports
import torch
import torch.nn as nn
# alibi-detect imports
from alibi_detect.cd import KSDrift, MMDDrift, LSDDDrift, CVMDrift
from alibi_detect.saving import save_detector, load_detector
from alibi_detect.cd.tensorflow import preprocess_drift


class UntrainedAutoencoder():
    def __init__(self, config=None) -> None:
        """ init config """
        self.config = config

        """ init logger """
        my_format = logging.Formatter(
            '%(asctime)s [%(levelname)s]  %(message)s')
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(my_format)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

        """ detectors """
        self.detectorKS: KSDrift = None
        self.detectorMMD: MMDDrift = None
        self.detectorLSDD: LSDDDrift = None
        self.detectorCVM: CVMDrift = None

        """ encoder """
        self.encoder_fn: partial = None

    # tensorflow encoder
    def init_default_tf_encoder(self, encoding_dim: int, input_shape: Tuple[int, int, int]):
        tf.random.set_seed(0)  # random
        # self.encoding_dim = encoding_dim # check later
        encoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(encoding_dim*2, 4, strides=2,
                       padding='same', activation=tf.nn.relu),
                Conv2D(encoding_dim*4, 4, strides=2,
                       padding='same', activation=tf.nn.relu),
                Conv2D(encoding_dim*16, 4, strides=2,
                       padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(encoding_dim)
            ]
        )
        self.encoder_fn = partial(
            preprocess_drift, model=encoder_net, batch_size=self.config['UAE']['BATCH_SIZE'])
        
    # pytorch encode (enc_dim=32)
    def init_default_py_encoder(self):
        encoder_net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 512, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 32)
        ).to(torch.device('cpu')).eval()
        self.encoder_fn = partial(
            preprocess_drift, model=encoder_net, batch_size=self.config['UAE']['BATCH_SIZE'])

    # init various types of detectors
    def init_detector(self, detector_type: str, reference_data: np.ndarray, backend='tensorflow', detector_name: str = None, save_dec: bool = False):
        # try:
        if detector_type == 'KS':
            detector = KSDrift(
                reference_data, p_val=self.config['GENERAL']['P_VAL'], preprocess_fn=self.encoder_fn)
            self.detectorKS = detector
        elif detector_type == 'MMD':
            detector = MMDDrift(
                x_ref=reference_data, p_val=self.config['GENERAL']['P_VAL'], backend=backend, preprocess_fn=self.encoder_fn)
            self.detectorMMD = detector
        elif detector_type == 'CVM':
            detector = CVMDrift(
                x_ref=reference_data, p_val=self.config['GENERAL']['P_VAL'], preprocess_fn=self.encoder_fn)
            self.detectorCVM = detector
        elif detector_type == 'LSDD':
            detector = LSDDDrift(
                x_ref=reference_data, p_val=self.config['GENERAL']['P_VAL'], preprocess_fn=self.encoder_fn)
            self.detectorLSDD = detector
        else:
            raise ValueError('Invalid Detector Type')
        self.logger.info('{} Detector initialized'.format(detector_type))
        if(save_dec and detector_name):
            try:
                save_detector(
                    detector, "{}/{}/".format(self.config['PATHS']['DETECTOR_DIR_PATH'], detector_name))
            except Exception as e:
                self.logger.info('Error in init_detector({}:{}): Error Saving Detector'.format(
                    detector_type, detector_name), e)
        # except Exception as e:
            # self.logger.exception('Error in init_detector({}): Error Initializing Detector'.format(detector_type),e)

    # import detector
    def import_detector(self, path: str, detector_type: str):
        try:
            if detector_type == 'KS':
                self.detectorKS = load_detector(path)
                self.logger.info('KS Detector imported')
            elif detector_type == 'MMD':
                self.detectorMMD = load_detector(path)
                self.logger.info('MMD Detector imported')
            elif detector_type == 'CVM':
                self.detectorCVM = load_detector(path)
                self.logger.info('CVM Detector imported')
            elif detector_type == 'LSDD':
                self.detectorLSDD = load_detector(path)
                self.logger.info('LSDD Detector imported')
            else:
                raise ValueError('Invalid Detector Type')
        except Exception as e:
            self.logger.exception(
                'Error in import_KS_detector(): Error Importing Detector', e)

    # make prediction
    def make_prediction(self, target_data: np.ndarray, detector_type: str) -> Dict[Dict[str, str], Dict[str, Union[np.ndarray, int, float]]]:
        labels = ['No!', 'Yes!']
        if detector_type == 'KS' and self.detectorKS is not None:
            preds = self.detectorKS.predict(x=target_data)
        elif detector_type == 'MMD' and self.detectorMMD is not None:
            preds = self.detectorMMD.predict(x=target_data)
        elif detector_type == 'CVM' and self.detectorCVM is not None:
            preds = self.detectorCVM.predict(x=target_data)
        elif detector_type == 'LSDD' and self.detectorLSDD is not None:
            preds = self.detectorLSDD.predict(x=target_data)
        else:
            raise ValueError(
                'Wrong Detector Type / No {} detector initialized'.format(detector_type))

        # print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        # print('Feature-wise p-values:')
        # print(preds['data']['p_val'])
        # print('len:{}'.format(int(len(preds['data']['p_val']))))

        return preds
