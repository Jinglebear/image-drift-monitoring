# MISC imports
import numpy as np
from typing import Dict,Union
import logging
from functools import partial
# alibi-detect imports
from alibi_detect.cd import KSDrift, MMDDrift, CVMDrift, LSDDDrift
from alibi_detect.saving import save_detector, load_detector
# sklearn imports
from sklearn.decomposition import PCA

class PrincipalComponentAnalysis():
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

        """ pca model """
        self.pca : PCA = None 

    # pca init 
    def init_pca(self,x_ref: np.ndarray,) -> PCA:
        try:
            shape = x_ref.shape
            x_ref = np.reshape(x_ref,(x_ref[0],int(shape[1]*shape[2]*shape[3]))) 
            pca = PCA(self.config['PCA_N_COMPONENTS'])
            pca.fit(x_ref) # fitting x_ref to pca 
            self.logger.info('init_pca(): fitted x_ref to pca')
        except Exception as e:
            self.logger.exception('Error in init pca()',e)
        return pca

    # init various types of detectors
    def init_detector(self,detector_type:str, reference_data:np.ndarray,detector_name:str =None, save_dec:bool = False):
        try:
            shape = reference_data.shape
            reference_data = np.reshape(reference_data,(reference_data[0],int(shape[1]*shape[2]*shape[3])))
            if detector_type == 'KS' and self.pca is not None:
                detector = KSDrift(reference_data,p_val=self.config['P_VAL'],preprocess_fn=self.pca.transform)
                self.detectorKS = detector
            elif detector_type == 'MMD' and self.pca is not None:
                detector = MMDDrift(x_ref=reference_data,p_val=self.config['P_VAL'],preprocess_fn=self.pca.transform)
                self.detectorMMD = detector
            elif detector_type == 'CVM' and self.pca is not None:
                detector = CVMDrift(x_ref=reference_data,p_val=self.config['P_VAL'],preprocess_fn=self.pca.transform)
                self.detectorCVM = detector
            elif detector_type == 'LSDD' and self.pca is not None:
                detector = LSDDDrift(x_ref=reference_data,p_val=self.config['P_VAL'],preprocess_fn=self.pca.transform)
                self.dectectorLSDD = detector
            else:
                raise ValueError('Invalid Detector Type / PCA not initialized')
            self.logger.info('{} Detector initialized'.format(detector_type))
            if(save_dec and detector_name):
                try:
                    save_detector(detector,"{}/{}".format(self.config['DETECTOR_DIR_PATH'],detector_name))
                except Exception as e:
                    self.logger.info('Error in init_detector({}:{}): Error Saving Detector'.format(detector_type,detector_name),e)
        except Exception as e:
                self.logger.exception('Error in init_detector({}): Error Initializing Detector'.format(detector_type),e)


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
        shape = target_data.shape
        target_data = np.reshape(target_data,(target_data[0],int(shape[1]*shape[2]*shape[3])))
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

            
        print('Drift? {}'.format(labels[preds['data']['is_drift']]))
        print('Feature-wise p-values:')
        print(preds['data']['p_val'])
        print('len:{}'.format(len(preds['data']['p_val']))) 
            
        return preds
