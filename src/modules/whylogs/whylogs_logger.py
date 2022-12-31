""" standard imports """
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional,List
import pytz
import random
import pandas as pd
from PIL import Image
from PIL.Image import Image as PIL_IMAGE
from progressbar import progressbar
import pathlib

""" whylogs imports  """

from whylogs.core.view.dataset_profile_view import DatasetProfileView
from whylogs.extras.image_metric import log_image
from whylogs.migration import uncompound
from whylogs.viz import NotebookProfileVisualizer
from whylogs.viz.utils.profile_viz_calculations import (
    OverallStats, add_feature_statistics, add_overall_statistics,
    frequent_items_from_view, generate_profile_summary, generate_summaries,
    histogram_from_view)

class Whylogs_Logger():
    def __init__(self, config = None) -> None:
        
        """ init output paths """
        self.OUTPUT_PATH = '/whylogs_output'
        self.OUTPUT_BIN = '/whylogs_output/bins'
        self.PROFILE_SUMMARIES = '/whylogs_output/profile_summaries'
        self.PROFILE_COMPARE = '/whylogs_output/profile_compare'
        """ init config """
        self.config = config

        """ init datetime, logger """
        self.my_datetime = datetime.now(pytz.timezone('Europe/Berlin'))

        """ init logger """
        my_format = logging.Formatter('%(asctime)s [%(levelname)s]  %(message)s')
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(my_format)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(handler)


    """ create output directories """
    def create_output_dir(self,path :str) -> None:
        pathlib.Path(path + self.OUTPUT_PATH).mkdir(parents=True,exist_ok=True)
        pathlib.Path(path + self.OUTPUT_BIN).mkdir(parents=True,exist_ok=True)
        pathlib.Path(path + self.PROFILE_SUMMARIES).mkdir(parents=True,exist_ok=True)
        pathlib.Path(path + self.PROFILE_COMPARE).mkdir(parents=True,exist_ok=True)


    """ log basic images from directory to a profile, set logging date """
    def log_basic_image_data_from_dir(self, data_directory_path: str,sub_dir_path: str, batch_size: int = 0, shuffle: bool = False) -> DatasetProfileView:
        try:
            self.create_output_dir(data_directory_path) # create output dirs 
            self.logger.info('Started logging image data from: {}'.format(sub_dir_path))
            profile = None
            data_directory_list = os.listdir(sub_dir_path)
            if shuffle and batch_size > 0:
                random.shuffle(data_directory_list)  # shuffle images
                data_directory_list = data_directory_list[0:batch_size]
            for img_name in progressbar(data_directory_list):
                img = Image.open(sub_dir_path + img_name)  # read in image
                _ = log_image(img).profile()  # generate profile
                _.set_dataset_timestamp(self.my_datetime) # optionally set dataset_timestamp
                _view = _.view()  # extract mergeable profile view
                if profile is None:
                    profile = _view
                else:
                    profile = profile.merge(_view) # merge each profile while looping
            self.logger.info('Logged image data from: {} with batch size {}'.format(sub_dir_path, profile.to_pandas()['image/Brightness.mean:counts/n']['image'])) #  get batchsize from output data
            return profile
        except Exception as e:
            self.logger.exception('Exception in log_data(): {}'.format(e))
    
    """ log pil images from a list to a profile, set logging date """
    def log_pil_images_data_from_list(self,data_directory_path:str,pil_data_arr: List[PIL_IMAGE],batch_size: int = 0, shuffle: bool = False) -> DatasetProfileView:
        try:
            self.create_output_dir(data_directory_path) # create output dirs 
            self.logger.info('Started logging PIL image data from array')
            profile = None
            if shuffle and batch_size > 0:
                random.shuffle(pil_data_arr)  # shuffle images
                pil_data_arr = pil_data_arr[0:batch_size]
            for img in progressbar(pil_data_arr):
                _ = log_image(img).profile()  # generate profile
                _.set_dataset_timestamp(self.my_datetime) # optionally set dataset_timestamp
                _view = _.view()  # extract mergeable profile view
                if profile is None:
                    profile = _view
                else:
                    profile = profile.merge(_view) # merge each profile while looping
            self.logger.info('Logged image data from PIL image data with batch size {}'.format(profile.to_pandas()['image/Brightness.mean:counts/n']['image'])) #  get batchsize from output data
            return profile
        except Exception as e:
            self.logger.exception('Exception in log_data(): {}'.format(e))

    def log_pil_image_to_profile(self,image : PIL_IMAGE, profile : DatasetProfileView) -> DatasetProfileView:
        try:
            _ = log_image(image).profile()
            _.set_dataset_timestamp(self.my_datetime) # optionally set dataset_timestamp
            _view = _.view()
            if profile is None:
                profile = _view
            else:
                profile = profile.merge(_view)
            return profile
        except Exception as e:
            self.logger.exception('Exception in log_pil_image_to_profile: {}'.format(e))

    def merge_profiles(self,profiles : List[DatasetProfileView]) -> DatasetProfileView:
        first = profiles[0] 
        for profile in profiles[1:]:
            first = first.merge(profile)
        return first



    """ serialize a profile to a .bin """
    def serialize_profile(self,profile: DatasetProfileView, binary_name: str, data_directory_path:str) -> str:
        try:
            serialize_profile_bin_path = '{}{}/{}.bin'.format(data_directory_path,self.OUTPUT_BIN,binary_name)
            profile.write(serialize_profile_bin_path)
            return serialize_profile_bin_path
        except Exception as e:
            self.logger.exception('Exception in serialize_profile(): {}'.format(e))


    """ deserialize a profile from a .bin """
    def deserialize_profile(self,data_directory_path: str, binary_name:str) -> DatasetProfileView:
        try:
            return DatasetProfileView.read('{}{}/{}.bin'.format(data_directory_path,self.OUTPUT_BIN,binary_name))
        except Exception as e:
            self.logger.exception('Exception in deserialize_profile(): {}'.format(e))


    """ create profile summary json"""
    def create_profile_summary_json(self,profile: DatasetProfileView, data_directory_path: str, summary_name : str) -> str:
        try:
            profile_summary_json_path = '{}{}/{}.json'.format(data_directory_path,self.PROFILE_SUMMARIES,summary_name)
            with open(profile_summary_json_path, 'w') as output_file:
                output_file.write(str(generate_profile_summary(target_view=uncompound._uncompound_dataset_profile(profile), config=None)['profile_from_whylogs']))
            self.logger.info('Created profile summary json for {}'.format(summary_name))
            return profile_summary_json_path
        except Exception as e:
            self.logger.exception('Exception in create_profile_summary(): {}'.format(e))


    """ create profile comparison json """
    def create_profile_compare_summary_json(self,target_profile: DatasetProfileView, ref_profile: DatasetProfileView, data_directory_path: str, compare_summary_name: str) -> str:
        try:
            compare_summary_json_path = '{}{}/{}.json'.format(data_directory_path,self.PROFILE_COMPARE,compare_summary_name)
            with open(compare_summary_json_path, 'w') as output_file:
                target_uncompound = uncompound._uncompound_dataset_profile(target_profile)
                ref_uncompound = uncompound._uncompound_dataset_profile(ref_profile)
                output_file.write(str(generate_summaries(target_view=target_uncompound,ref_view=ref_uncompound, config=None)['reference_profile_from_whylogs']))
            return compare_summary_json_path
        except Exception as e:
            self.logger.exception('Exception in create_profile_compare_summary_json(): {}'.format(e))


    """ create metric and p_val dataframes """
    def create_drift_metric_df_from_comp_summary_json(self,data_directory_path: str, compare_summary_name : str) -> pd.DataFrame:
        try:
            with open('{}{}/{}.json'.format(data_directory_path,self.PROFILE_COMPARE,compare_summary_name), 'r') as input:
                data = json.load(input)
            tmp = {"metric": [], "p_val": [], "rating": []}
            for metric in data['columns']:
                if metric != 'image.Colorspace':
                    tmp['metric'].append(metric)
                    p_val = data['columns'][metric]['drift_from_ref']
                    tmp['p_val'].append(p_val)
                    if p_val <= 0.05:
                        tmp["rating"].append('Detected Drift (0.00 - 0.05)')
                    elif p_val <= 0.15:
                        tmp["rating"].append('Possible Drift (0.05 - 0.15)')
                    elif p_val <= 1.0:
                        tmp['rating'].append('No Drift (0.15 - 1.00)')
            df = pd.DataFrame.from_dict(tmp)
            df.sort_values(by=['metric'], inplace=True)
            df.to_csv('{}{}/{}.csv'.format(data_directory_path,self.PROFILE_COMPARE,compare_summary_name))
            return df
        except Exception as e:
            self.logger.exception('Exception in create_drift_metric_df_from_comp_summary_json(): {}'.format(e))


    """ create visualization  """
    def create_visualization(self,data_directory_path:str, viz_name:str, target_profile: DatasetProfileView, referece_profile: DatasetProfileView) -> str:
        try:
            _viz = NotebookProfileVisualizer()
            _viz.set_profiles(target_profile_view=target_profile,reference_profile_view=referece_profile)
            html_viz = _viz.summary_drift_report()
            html_viz_path = '{}{}/{}.html'.format(data_directory_path,self.PROFILE_COMPARE,viz_name)
            with open(html_viz_path, 'w') as output:
                output.write(html_viz.data)
            return html_viz_path
        except Exception as e:
            self.logger.exception('Exception in create_metrics(): {}'.format(e))