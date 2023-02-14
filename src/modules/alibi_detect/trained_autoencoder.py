# MISC imports
import numpy as np
import logging
from typing import Dict, Tuple, Union
from functools import partial
# tensorflow imports
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, InputLayer, Conv2DTranspose, Reshape
# pytorch imports 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# alibi-detect imports
from alibi_detect.cd import KSDrift, MMDDrift, LSDDDrift, CVMDrift
from alibi_detect.saving import save_detector, load_detector
from alibi_detect.cd.tensorflow import preprocess_drift
from alibi_detect.models.pytorch import trainer

class TrainedAutoencoder():
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

        """ detectors """
        self.detectorKS: KSDrift = None
        self.detectorMMD : MMDDrift = None
        self.dectectorLSDD : LSDDDrift = None
        self.detectorCVM : CVMDrift = None

        """ Autoencoder """
        self.encoder = None
        self.decoder = None

    # tensorflow encoder 
    def init_default_tf_autoencoder(self,encoding_dim :int,input_shape : Tuple[int,int,int],batch_size :int):
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
        decoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=input_shape),
                Dense(encoding_dim),
                Reshape(),
                Conv2DTranspose(filters=encoding_dim*16,kernel_size=4,strides=4,padding='same',activation=tf.nn.relu),
                Conv2DTranspose(filters=encoding_dim*4,kernel_size=4,strides=4,padding='same',activation=tf.nn.relu),
                Conv2DTranspose(filters=encoding_dim*4,kernel_size=4,strides=4,padding='same',activation=tf.nn.relu),
            ]
        )
        return partial(preprocess_drift,model=encoder_net,batch_size=batch_size)

    def init_default_pt_autoencoder(self,dl : DataLoader):
        ENC_DIM = self.config['TAE']['ENC_DIM']
        BATCH_SIZE = self.config['TAE']['BATCH_SIZE']
        EPOCHS = self.config['TAE']['EPOCHS']
        LEARNING_RATE = self.config['TAE']['LEARNING_RATE']

        self.encoder = nn.Sequential(
            nn.Conv2d(3,8,5,stride=3,padding=1), # (batch, 8, 32, 32)
            nn.ReLU(),
            nn.Conv2d(8,12,4,stride=2,padding=1), # (batch,12,16,16)
            nn.ReLU(),
            nn.Conv2d(12,16,4,stride=2,padding=1), # (batch,16,8,8)
            nn.ReLU(),
            nn.Conv2d(16,20,4,stride=2,padding=1), # (batch,20,4,4)
            nn.ReLU(),
            nn.Conv2d(20,ENC_DIM,4,stride=1,padding=0), #(batch,enc_dim,1,1)
            nn.Flatten(),   
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1,(ENC_DIM,1,1)),
            nn.ConvTranspose2d(ENC_DIM,20,4,stride=1,padding=0), # [batch,20,4,4]
            nn.ReLU(),
            nn.ConvTranspose2d(20,16,4,stride=2,padding=1), # (batch,16,8,8)
            nn.ReLU(),
            nn.ConvTranspose2d(16,12,4,stride=2,padding=1), # (batch,12,16,16)
            nn.ReLU(),
            nn.ConvTranspose2d(12,8,4,stride=2,padding=1),  # (batch,8,32,32)
            nn.ReLU(),
            nn.ConvTranspose2d(8,3,5,stride=3,padding=1),  # (batch,3,96,96)
            nn.Sigmoid()
        )
        device = torch.device('cpu')
        ae = nn.Sequential(self.encoder,self.decoder).to(device)
        trainer(ae,nn.MSELoss(),dl,device,learning_rate=LEARNING_RATE,epochs=EPOCHS)

    def encoder_fn(self,x: np.ndarray) -> np.array:
        x = torch.as_tensor(x).to(device=torch.device('cpu'))
        with torch.no_grad():
            x_proj = self.encoder(x)
        return x_proj.cpu().numpy()

    # init various types of detectors
    def init_detector(self,detector_type:str, reference_data:np.ndarray, detector_name:str =None, save_dec:bool = False):
        # try:
            if detector_type == 'KS':
                detector = KSDrift(reference_data,p_val=self.config['GENERAL']['P_VAL'],preprocess_fn=self.encoder_fn)
                self.detectorKS = detector
            elif detector_type == 'MMD':
                detector = MMDDrift(x_ref=reference_data,p_val=self.config['GENERAL']['P_VAL'],preprocess_fn=self.encoder_fn)
                self.detectorMMD = detector
            elif detector_type == 'CVM':
                detector = CVMDrift(x_ref=reference_data,p_val=self.config['GENERAL']['P_VAL'],preprocess_fn=self.encoder_fn)
                self.detectorCVM = detector
            elif detector_type == 'LSDD':
                detector = LSDDDrift(x_ref=reference_data,p_val=self.config['GENERAL']['P_VAL'],preprocess_fn=self.encoder_fn)
                self.dectectorLSDD = detector
            else:
                raise ValueError('Invalid Detector Type')
            self.logger.info('{} Detector initialized'.format(detector_type))
            if(save_dec and detector_name):
                try:
                    save_detector(detector,"{}/{}/".format(self.config['PATHS']['DETECTOR_DIR_PATH'],detector_name))
                except Exception as e:
                    self.logger.info('Error in init_detector({}:{}): Error Saving Detector'.format(detector_type,detector_name),e)

        # except Exception as e:
        #         self.logger.exception('Error in init_detector({}): Error Initializing Detector'.format(detector_type),e)


    # import detector
    def import_detector(self,path:str, detector_type: str):
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
                self.dectectorLSDD = load_detector(path)
                self.logger.info('LSDD Detector imported')
            else:
                raise ValueError('Invalid Detector Type')
        except Exception as e:
                self.logger.exception('Error in import_KS_detector(): Error Importing Detector',e)
                 
    # make prediction
    def make_prediction(self,target_data:np.ndarray, detector_type :str) ->Dict[Dict[str, str], Dict[str, Union[np.ndarray,int,float] ]]:
        labels = ['No!', 'Yes!']
        if detector_type == 'KS' and self.detectorKS is not None:
            preds = self.detectorKS.predict(x=target_data) 
        elif detector_type == 'MMD' and self.detectorMMD is not None:
            preds = self.detectorMMD.predict(x=target_data) 
        elif detector_type == 'CVM' and self.detectorCVM is not None:
            preds = self.detectorCVM.predict(x=target_data) 
        elif detector_type == 'LSDD' and self.dectectorLSDD is not None:
            preds = self.detectorCVM.predict(x=target_data) 
        else:
            raise ValueError('Wrong Detector Type / No {} detector initialized'.format(detector_type))

            
        # print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        # print('Feature-wise p-values:')
        # print(preds['data']['p_val'])
        # print('len:{}'.format(len(preds['data']['p_val']))) 
            
        return preds

    # TODO: install required package / delete this 
    # pytorch encoder expects images in (channels, height, width) format
    # def init_default_pt_encoder(self,encoding_dim :int,input_shape : Tuple[int,int,int],batch_size :int):
    #     torch.manual_seed(0)
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     encoder_net = nn.Sequential(
    #         nn.Conv2d(in_channels=input_shape[0],out_channels=input_shape[1]*2,kernel_size=4,stride=2,padding=0),
    #         nn.ReLU(),
    #         nn.Conv2d(in_channels=input_shape[1]*2,out_channels=input_shape[1]*4,kernel_size=4,stride=2,padding=0),
    #         nn.ReLU(),
    #         nn.Conv2d(in_channels=input_shape[1]*4,out_channels=input_shape[1]*16,kernel_size=4,stride=2,padding=0),
    #         nn.ReLU(),
    #         nn.Flatten(),
    #         nn.Linear(in_features=input_shape[1]*64,out_features=encoding_dim)
    #     ).to(device=device).eval()
    #     return partial(preprocess_drift,model=encoder_net,batch_size=batch_size)