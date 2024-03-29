# MISC imports
import logging
from typing import Dict, Union
import numpy as np

# alibi-detect imports
from alibi_detect.cd import CVMDrift, KSDrift, LSDDDrift, MMDDrift
from alibi_detect.saving import load_detector, save_detector

# sklearn imports
from sklearn.decomposition import PCA


class PrincipalComponentAnalysis():
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

        """ pca model """
        self.pca: PCA = None

    # pca init
    def init_pca(self, x_ref: np.ndarray,) -> PCA:
        try:
            shape = x_ref.shape
            x_ref = np.reshape(
                x_ref, (shape[0], int(shape[1]*shape[2]*shape[3])))
            print(x_ref.shape)
            self.pca = PCA(int(self.config['PCA']['PCA_N_COMPONENTS']))
            self.pca.fit(x_ref)  # fitting x_ref to pca
            self.logger.info('init_pca(): fitted x_ref to pca')
        except Exception as e:
            self.logger.exception('Error in init pca()', e)
        return self.pca

    # init various types of detectors
    def init_detector(self, detector_type: str, reference_data: np.ndarray, detector_name: str = None, save_dec: bool = False):
        try:
            shape = reference_data.shape
            reference_data = np.reshape(
                reference_data, (shape[0], int(shape[1]*shape[2]*shape[3])))
            if detector_type == 'KS' and self.pca is not None:
                detector = KSDrift(
                    reference_data, p_val=self.config['GENERAL']['P_VAL'], preprocess_fn=self.pca.transform)
                self.detectorKS = detector
            elif detector_type == 'MMD' and self.pca is not None:
                detector = MMDDrift(
                    x_ref=reference_data, p_val=self.config['GENERAL']['P_VAL'], preprocess_fn=self.pca.transform)
                self.detectorMMD = detector
            elif detector_type == 'CVM' and self.pca is not None:
                detector = CVMDrift(
                    x_ref=reference_data, p_val=self.config['GENERAL']['P_VAL'], preprocess_fn=self.pca.transform)
                self.detectorCVM = detector
            elif detector_type == 'LSDD' and self.pca is not None:
                detector = LSDDDrift(
                    x_ref=reference_data, p_val=self.config['GENERAL']['P_VAL'], preprocess_fn=self.pca.transform)
                self.detectorLSDD = detector
            else:
                raise ValueError('Invalid Detector Type / PCA not initialized')
            self.logger.info('{} Detector initialized'.format(detector_type))
            if(save_dec and detector_name):
                try:
                    save_detector(
                        detector, "{}/{}".format(self.config['PATHS']['DETECTOR_DIR_PATH'], detector_name))
                except Exception as e:
                    self.logger.info('Error in init_detector({}:{}): Error Saving Detector'.format(
                        detector_type, detector_name), e)
        except Exception as e:
            self.logger.exception(
                'Error in init_detector({}): Error Initializing Detector'.format(detector_type), e)

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
        shape = target_data.shape
        target_data = np.reshape(
            target_data, (shape[0], int(shape[1]*shape[2]*shape[3])))
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

        return preds
