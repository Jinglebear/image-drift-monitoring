import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import pytz
from PIL import Image
from progressbar import progressbar
from whylogs.core.view.dataset_profile_view import DatasetProfileView
from whylogs.extras.image_metric import log_image
from whylogs.migration import uncompound
from whylogs.viz import NotebookProfileVisualizer
from whylogs.viz.utils.profile_viz_calculations import (
    OverallStats, add_feature_statistics, add_overall_statistics,
    frequent_items_from_view, generate_profile_summary, generate_summaries,
    histogram_from_view)

""" log data from directory to a profile, set logging date """
""" ****************************************************** """
def log_data(data_directory : str,datetime : datetime, logger : logging.Logger) -> DatasetProfileView:
    try:
        logger.info('Started logging image data from: {}'.format(data_directory))
        profile = None

        for img_name in progressbar(os.listdir(data_directory)):

            img = Image.open(data_directory + img_name) # read in image
            _ = log_image(img).profile() # generate profile  
            _.set_dataset_timestamp(datetime) # optionally set dataset_timestamp
            _view = _.view() # extract mergeable profile view
            # merge each profile while looping
            if profile is None:
                profile = _view
            else:
                profile = profile.merge(_view)
        logger.info('Logged image data from: {} with batch size {}'.format(data_directory,profile.to_pandas()['image/Brightness.mean:counts/n']['image']))
        return profile
    except Exception as e:
        logger.exception('Exception in log_data(): {}'.format(e))
""" serialize a profile to a .bin """
""" ****************************************************** """
def serialize_profile(profile : DatasetProfileView, binary_name : str,logger : logging.Logger) -> str:
    try:
        return profile.write('{}.bin'.format(binary_name))
    except Exception as e:
        logger.exception('Exception in serialize_profile(): {}'.format(e))

""" deserialize a profile from a .bin """
""" ****************************************************** """
def deserialize_profile(path : str,logger : logging.Logger) -> DatasetProfileView:
    try:
        return DatasetProfileView.read(path)
    except Exception as e:
        logger.exception('Exception in deserialize_profile(): {}'.format(e))
        
        

""" create profile viz """
""" ****************** """
def create_metrics(target_profile: DatasetProfileView, referece_profile: DatasetProfileView,logger : logging.Logger) -> NotebookProfileVisualizer:
    try:
        _viz = NotebookProfileVisualizer
        _viz.set_profiles(target_profile_view=target_profile,reference_profile_view=referece_profile)
        return _viz
    except Exception as e:
        logger.exception('Exception in create_metrics(): {}'.format(e))

""" create profile summary json"""
""" ************************** """
def create_profile_summary_json(profile: DatasetProfileView,path :str, logger : logging.Logger) ->  str:
    try:
        with open('{}.json'.format(path), 'w') as output_file:
            output_file.write(str(generate_profile_summary(target_view=uncompound._uncompound_dataset_profile(profile),config=None)['profile_from_whylogs']))
        logger.info('Created profile summary json for {}'.format(path))
    except Exception as e:
        logger.exception('Exception in create_profile_summary(): {}'.format(e))

        

def main():
    
    """ init datetime, logger """
    my_datetime = datetime.now(pytz.timezone('Europe/Berlin'))
    my_format = logging.Formatter('%(asctime)s [%(levelname)s]  %(message)s')
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(my_format)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    LANDSCAPE_DATA_RAW_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/landscape_data_raw'
    
    LANDSCAPE_BINS_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/bins'

    LANDSCAPE_JSON_PROFILES_PATH = '/home/jinglewsl/evoila/sandbox/whylogs_v1/image-drift/output/landscape/profile_summaries'


    """ log  & serialize landscape dataset  """
    profile_baseline = log_data('{}/landscape_baseline/baseline/'.format(LANDSCAPE_DATA_RAW_PATH),my_datetime,logger)
    serialize_profile(profile_baseline,'{}/baseline'.format(LANDSCAPE_BINS_PATH),logger)

    profile_landscape = log_data('{}/landscape_images/landscape/'.format(LANDSCAPE_DATA_RAW_PATH),my_datetime,logger)
    serialize_profile(profile_landscape,'{}/landscape'.format(LANDSCAPE_BINS_PATH),logger)

    profile_camera = log_data('{}/camera_images/camera/'.format(LANDSCAPE_DATA_RAW_PATH),my_datetime,logger)
    serialize_profile(profile_camera,'{}/camera'.format(LANDSCAPE_BINS_PATH),logger)


    """ load & deserialize landscape dataset """
    profile_baseline = deserialize_profile('{}/baseline.bin'.format(LANDSCAPE_BINS_PATH),logger)
    profile_landscape = deserialize_profile('{}/landscape.bin'.format(LANDSCAPE_BINS_PATH),logger)
    profile_camera = deserialize_profile('{}/camera.bin'.format(LANDSCAPE_BINS_PATH),logger)


    """ create summary json's """
    create_profile_summary_json(profile_baseline,'{}/baseline'.format(LANDSCAPE_JSON_PROFILES_PATH),logger)
    create_profile_summary_json(profile_landscape,'{}/landscape'.format(LANDSCAPE_JSON_PROFILES_PATH),logger)
    create_profile_summary_json(profile_camera,'{}/camera'.format(LANDSCAPE_JSON_PROFILES_PATH),logger)
    
    

if __name__ == "__main__":
    main()
