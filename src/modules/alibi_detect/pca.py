# alibi
import matplotlib.pyplot as plt
import numpy as np
import os
import logging

from alibi_detect.cd import KSDrift, MMDDrift

from alibi_detect.saving import save_detector, load_detector





from typing import Dict, Tuple, Union

from functools import partial




from sklearn.decomposition import PCA


class PCA():
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

    # pca init 
    def init_pca(self,x_ref: np.ndarray, n_comp: int) -> PCA:
        try:
            shape = x_ref.shape
            # reshape for pca
            # TODO : fix 
            x_ref = np.reshape(x_ref,(x_ref[0],int(shape[1]*shape[2]*shape[3]))) 
            pca = PCA(n_comp)
            pca.fit(x_ref)
            self.logger.info('init_pca(): fitted x_ref to pca')
        except Exception as e:
            self.logger.exception('Error in init pca()')
        return pca

    def init_ks_detector(self,reference_data: np.ndarray, pca : PCA, p_val: float = 0.05, path : str = None, save_dec : bool = False):
        detector = KSDrift(reference_data,p_val=p_val,preprocess_fn=pca.transform)
        # assert detector.p_val / detector.n_features == p_val / self.encoding_dim #check later
        if(save_dec and path):
            try:
                save_detector(detector,path)
            except Exception as e:
                self.logger.info('Error in init_ks_detector(): Error Saving Detector',e)
        self.detectorKS = detector
        self.logger.info('KS Detector initialized')


    def init_mmd_detector(self,reference_data: np.ndarray, encoder_fn : partial, n_permutations : int, p_val: float =0.05, path: str =None, save_dec : bool = False, backend : str = 'tensorflow'):
        detector = MMDDrift(x_ref=reference_data,backend=backend,p_val=p_val,preprocess_fn=encoder_fn,n_permutations=n_permutations)
        if(save_dec and path):
            try:
                save_detector(detector,path)
            except Exception as e:
                self.logger.info('Error in init_mmd_detector(): Error Saving Detector',e)
        self.detectorMMD = detector
        self.logger.info('MMD Detector initialized')


    def import_detector(self,path:str, type: str):
        try:
            if type == 'KS':
                self.detectorKS = load_detector(path) #load drift detector
                self.logger.info('KS Detector imported')
            elif type == 'MMD':
                self.detectorMMD = load_detector(path)
                self.logger.info('MMD Detector imported')
            else:
                raise ValueError('Invalid Detector Type')
        except Exception as e:
                self.logger.exception('Error in import_KS_detector(): Error Importing Detector',e)
                 

    def make_prediction(self,target_data:np.ndarray, type :str) ->Dict[Dict[str, str], Dict[str, Union[np.ndarray,int,float] ]]:
        labels = ['No!', 'Yes!']
        if type == 'KS':
            if self.detectorKS is None:
                self.logger.exception('No Detector initialized')
            else:
                preds = self.detectorKS.predict(x=target_data) # predict wether a batch of data has drifted from reference data
                print('Drift? {}'.format(labels[preds['data']['is_drift']]))
                print('Feature-wise p-values:')
                print(preds['data']['p_val'])
                print('len:{}'.format(len(preds['data']['p_val'])))
                return preds
        elif type == 'MMD':
            if self.detectorMMD is None:
                self.logger.exception('No Detector initialized')
            else:
                preds = self.detectorMMD.predict(x=target_data) # predict wether a batch of data has drifted from reference data
                print('Drift? {}'.format(labels[preds['data']['is_drift']]))
                print('Feature-wise p-values:')
                print(preds['data']['p_val'])
                print('len:{}'.format(len(preds['data']['p_val'])))
                return preds
        else:
            raise ValueError('Invalid Detector Type')
