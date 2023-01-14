# alibi
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import logging
from alibi_detect.cd import KSDrift
from alibi_detect.models.tensorflow import scale_by_instance
from alibi_detect.utils.fetching import fetch_tf_model, fetch_detector
from alibi_detect.saving import save_detector, load_detector


from typing import Dict, Tuple, Union

from functools import partial
from tensorflow import keras
from keras.layers import Conv2D, Dense, Flatten, InputLayer, Reshape
from alibi_detect.cd.tensorflow import preprocess_drift

class Untrained_Autoencoder():
    def __init__(self,config=None) -> None:
        """ init config """
        self.config = config

        """ init logger """
        my_format = logging.Formatter('%(asctime)s [%(levelname)s]  %(message)s')
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(my_format)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)

        """ detector """
        self.detector: KSDrift = None

    def init_encoder(self,encoding_dim :int,input_shape : Tuple[int,int,int],batch_size :int):
        tf.random.set_seed(0) # random
        # self.encoding_dim = encoding_dim # check later
        encoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=input_shape),
                Conv2D(encoding_dim*2,4,strides=2,padding='same',activation=tf.nn.relu),
                Conv2D(encoding_dim*4,4,strides=2,padding='same',activation=tf.nn.relu),
                Conv2D(encoding_dim*16,4,strides=2,padding='same',activation=tf.nn.relu),
                Flatten(),
                Dense(encoding_dim)
            ]
        )
        return partial(preprocess_drift,model=encoder_net,batch_size=batch_size)
    
    def init_detector(self,reference_data: np.ndarray,encoder_fn : partial, p_val: float = 0.05, path:str = None, save_dec :bool =False):
        detector = KSDrift(reference_data,p_val=p_val,preprocess_fn=encoder_fn)
        # assert detector.p_val / detector.n_features == p_val / self.encoding_dim #check later
        if(save_dec and path):
            try:
                save_detector(detector,path)
            except Exception as e:
                self.logger.info('Error in init_detector(): Error Saving Detector',e)
        self.detector = detector
        self.logger.info('Detector initialized')

    def import_detector(self,path:str):
        try:
            self.detector = load_detector(path) #load drift detector
            self.logger.info('Detector imported and initialized')
        except Exception as e:
                self.logger.exception('Error in import_detector(): Error Saving Detector',e) 
    def make_prediction(self,target_data:np.ndarray) ->Dict[Dict[str, str], Dict[str, Union[np.ndarray,int,float] ]]:
        labels = ['No!', 'Yes!']
        if self.detector is None:
            self.logger.exception('No Detector initialized')
        else:
            preds = self.detector.predict(x=target_data) # predict wether a batch of data has drifted from reference data
            print('Drift? {}'.format(labels[preds['data']['is_drift']]))
            print('Feature-wise p-values:')
            print(preds['data']['p_val'])
            print('len:{}'.format(len(preds['data']['p_val'])))
            return preds
